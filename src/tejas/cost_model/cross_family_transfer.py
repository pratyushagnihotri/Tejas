"""Cross-family transfer: train the energy/quality ladders on one DGX
family (Qwen or Llama), evaluate zero-shot on the other. Restricted to the
two DGX experiments — both share measurement_scope="gpu_die_only", so
transferring absolute-J targets between them doesn't violate the
measurement-boundary guard in energy_model.py (Mac's different physical
measurement boundary rules it out for this leg — see deployment_features.py
and RESEARCH_LOG.md's measurement-scope finding).

Two protocols, since DGX-Qwen and DGX-Llama share the same 40 task IDs:
- Protocol A (matched-workload): train on ALL of one family's 40 tasks,
  evaluate on ALL of the other's. Isolates deployment/model-stack transfer
  while holding workload constant — does NOT test generalization to unseen
  workloads, only to a new model stack for workloads already seen.
- Protocol B (unseen-workload): task IDs are split BEFORE crossing
  deployments (grouped, repeated), so evaluated tasks were seen by neither
  source nor target training data — the harder, more honest test.

Family, parameter range, and GPU count (1 vs 2) all change simultaneously
between DGX-Qwen and DGX-Llama — this isolates "does a cost model trained
on one deployment transfer to another," not "model family" alone.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from tejas.cost_model.dataset import feature_columns, load_experiment_frame
from tejas.cost_model.energy_model import ENERGY_MODEL_KINDS, EnergyModel
from tejas.cost_model.evaluate_models import FEATURE_GROUPS, _calibration_diagnostics, _per_task_energy_diagnostics
from tejas.cost_model.quality_model import QUALITY_MODEL_KINDS, QualityModel
from tejas.paths import RESULTS_DIR

QWEN_CSV = os.path.join(str(RESULTS_DIR), "raw_results_dgx_qwen.csv")
LLAMA_CSV = os.path.join(str(RESULTS_DIR), "raw_results_dgx_llama.csv")
N_SPLITS = 5
N_REPEATS = 3


def _fit_eval_energy(train_df, test_df, cat, num):
    rows = []
    for kind in ENERGY_MODEL_KINDS:
        m = EnergyModel(kind, cat, num).fit(train_df, train_df["energy_j"])
        pred = m.predict(test_df)
        mae = float(np.mean(np.abs(pred - test_df["energy_j"])))
        rmse_log = float(np.sqrt(np.mean((np.log(pred) - np.log(test_df["energy_j"])) ** 2)))
        avg_spearman, min_tier_hit = _per_task_energy_diagnostics(test_df, pred)
        rows.append({"kind": kind, "mae_j": mae, "rmse_log": rmse_log,
                      "avg_spearman": avg_spearman, "min_tier_hit_rate": min_tier_hit})
    return rows


def _fit_eval_quality(train_df, test_df, cat, num):
    rows = []
    for kind in QUALITY_MODEL_KINDS:
        m = QualityModel(kind, cat, num).fit(train_df, train_df["correct"], groups=train_df["task_id"])
        pred = m.predict_proba(test_df)
        accuracy = float(np.mean((pred >= 0.5) == test_df["correct"]))
        cal = _calibration_diagnostics(test_df["correct"].values, pred)
        rows.append({"kind": kind, "accuracy": accuracy, "brier": cal["brier"], "log_loss": cal["log_loss"]})
    return rows


def protocol_a(source_csv, target_csv, source_name, target_name):
    """Matched-workload: train on all 40 source tasks, test on all 40
    target tasks (same task IDs, different deployment)."""
    source_df, _, _ = load_experiment_frame(source_csv)
    target_df, _, _ = load_experiment_frame(target_csv)
    cat, num = feature_columns(FEATURE_GROUPS)

    energy_rows = _fit_eval_energy(source_df, target_df, cat, num)
    quality_rows = _fit_eval_quality(source_df, target_df, cat, num)
    for r in energy_rows:
        r.update(protocol="A_matched_workload", source=source_name, target=target_name)
    for r in quality_rows:
        r.update(protocol="A_matched_workload", source=source_name, target=target_name)
    return pd.DataFrame(energy_rows), pd.DataFrame(quality_rows)


def protocol_b(source_csv, target_csv, source_name, target_name):
    """Unseen-workload: task IDs split before crossing deployments —
    source trains only on TRAIN tasks, target is evaluated only on TEST
    tasks, so no task_id is shared between what source trained on and what
    target is scored on."""
    source_df, _, _ = load_experiment_frame(source_csv)
    target_df, _, _ = load_experiment_frame(target_csv)
    cat, num = feature_columns(FEATURE_GROUPS)
    task_type_tier_by_task = source_df.drop_duplicates("task_id").set_index("task_id")["task_type_tier"]
    task_ids = source_df["task_id"].unique()

    energy_all, quality_all = [], []
    for repeat in range(N_REPEATS):
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=repeat)
        strat_y = task_type_tier_by_task.loc[task_ids].values
        for fold_idx, (train_pos, test_pos) in enumerate(splitter.split(task_ids, strat_y, groups=task_ids)):
            train_tasks = set(task_ids[train_pos])
            test_tasks = set(task_ids[test_pos])
            assert train_tasks.isdisjoint(test_tasks)
            train_df = source_df[source_df["task_id"].isin(train_tasks)]
            test_df = target_df[target_df["task_id"].isin(test_tasks)]

            for r in _fit_eval_energy(train_df, test_df, cat, num):
                r.update(repeat=repeat, fold=fold_idx)
                energy_all.append(r)
            for r in _fit_eval_quality(train_df, test_df, cat, num):
                r.update(repeat=repeat, fold=fold_idx)
                quality_all.append(r)

    energy_df = pd.DataFrame(energy_all)
    quality_df = pd.DataFrame(quality_all)
    energy_summary = energy_df.groupby("kind").agg(
        mae_j_mean=("mae_j", "mean"), mae_j_std=("mae_j", "std"),
        avg_spearman_mean=("avg_spearman", "mean"),
        min_tier_hit_rate_mean=("min_tier_hit_rate", "mean"),
    ).reset_index()
    energy_summary["protocol"] = "B_unseen_workload"
    energy_summary["source"], energy_summary["target"] = source_name, target_name
    quality_summary = quality_df.groupby("kind").agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        brier_mean=("brier", "mean"), log_loss_mean=("log_loss", "mean"),
    ).reset_index()
    quality_summary["protocol"] = "B_unseen_workload"
    quality_summary["source"], quality_summary["target"] = source_name, target_name
    return energy_summary, quality_summary


def main():
    print("=== Protocol A: matched-workload transfer (same 40 tasks, different deployment) ===")
    for source_csv, target_csv, sname, tname in [
        (QWEN_CSV, LLAMA_CSV, "dgx_qwen", "dgx_llama"),
        (LLAMA_CSV, QWEN_CSV, "dgx_llama", "dgx_qwen"),
    ]:
        e_df, q_df = protocol_a(source_csv, target_csv, sname, tname)
        print(f"\n  {sname} -> {tname}:")
        print("  energy:")
        print(e_df.to_string(index=False))
        print("  quality:")
        print(q_df.to_string(index=False))
        e_df.to_csv(os.path.join(str(RESULTS_DIR), f"transfer_A_energy_{sname}_to_{tname}.csv"), index=False)
        q_df.to_csv(os.path.join(str(RESULTS_DIR), f"transfer_A_quality_{sname}_to_{tname}.csv"), index=False)

    print(f"\n=== Protocol B: unseen-workload transfer ({N_SPLITS}x{N_REPEATS} grouped splits) ===")
    for source_csv, target_csv, sname, tname in [
        (QWEN_CSV, LLAMA_CSV, "dgx_qwen", "dgx_llama"),
        (LLAMA_CSV, QWEN_CSV, "dgx_llama", "dgx_qwen"),
    ]:
        e_df, q_df = protocol_b(source_csv, target_csv, sname, tname)
        print(f"\n  {sname} -> {tname}:")
        print("  energy:")
        print(e_df.to_string(index=False))
        print("  quality:")
        print(q_df.to_string(index=False))
        e_df.to_csv(os.path.join(str(RESULTS_DIR), f"transfer_B_energy_{sname}_to_{tname}.csv"), index=False)
        q_df.to_csv(os.path.join(str(RESULTS_DIR), f"transfer_B_quality_{sname}_to_{tname}.csv"), index=False)


if __name__ == "__main__":
    main()
