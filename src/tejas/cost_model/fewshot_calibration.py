"""Few-shot deployment calibration: given a cost model trained on one DGX
family (the "source"), how many tasks from a *new* deployment (the
"target") are needed before it's useful there?

Two approaches compared at k in {1, 5, 10} calibration tasks (20 random
subsets per k, evaluated on the remaining 40-k target tasks each time):

- **full refit**: retrain the energy/quality models on source-40-tasks +
  target-k-tasks combined.
- **linear calibration**: freeze the source-trained model, and fit a
  simple correction using only the k calibration tasks' rows —
  `logE_target = a + b*logE_hat_source` (least squares; also a fold-safe
  logit-space Platt-style correction for quality). Distinguishes "the
  target deployment is basically a scale shift" (linear calibration
  should already recover most of the benefit) from "it needs genuinely
  new task-model interaction data" (only full refit helps).

Implementation note: even at k=1, the single calibration task still
contributes 3 tiers x 4 runs = 12 rows, so the linear correction always
fits both `a` and `b` via least squares rather than an intercept-only
special case — a real zero-shot deployment might only have 1 row per
task, which would need the intercept-only fallback; this dataset has more
signal per task than that minimal scenario, and this is flagged here
rather than silently assumed away.

Restricted to the two DGX experiments (shared measurement_scope), same as
cross_family_transfer.py.
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from tejas.cost_model.dataset import feature_columns, load_experiment_frame
from tejas.cost_model.energy_model import EnergyModel
from tejas.cost_model.evaluate_models import FEATURE_GROUPS
from tejas.cost_model.quality_model import QualityModel
from tejas.paths import RESULTS_DIR

QWEN_CSV = os.path.join(str(RESULTS_DIR), "raw_results_dgx_qwen.csv")
LLAMA_CSV = os.path.join(str(RESULTS_DIR), "raw_results_dgx_llama.csv")
K_VALUES = (1, 5, 10)
N_REPEATS = 20
ENERGY_KIND = "histgb"
QUALITY_KIND = "logreg"


def _split_calib_eval(target_df, k, seed):
    rng = np.random.default_rng(seed)
    task_ids = target_df["task_id"].unique()
    calib_tasks = set(rng.choice(task_ids, size=k, replace=False))
    eval_tasks = [t for t in task_ids if t not in calib_tasks]
    return (target_df[target_df["task_id"].isin(calib_tasks)],
            target_df[target_df["task_id"].isin(eval_tasks)])


def _eval_metrics(pred_e, eval_df, pred_q):
    mae = float(np.mean(np.abs(pred_e - eval_df["energy_j"])))
    accuracy = float(np.mean((pred_q >= 0.5) == eval_df["correct"]))
    p = np.clip(pred_q, 1e-6, 1 - 1e-6)
    ll = float(log_loss(eval_df["correct"], p, labels=[0, 1]))
    return mae, accuracy, ll


def full_refit(source_df, calib_df, eval_df, cat, num):
    train_df = pd.concat([source_df, calib_df], ignore_index=True)
    em = EnergyModel(ENERGY_KIND, cat, num).fit(train_df, train_df["energy_j"])
    qm = QualityModel(QUALITY_KIND, cat, num).fit(train_df, train_df["correct"], groups=train_df["task_id"])
    return _eval_metrics(em.predict(eval_df), eval_df, qm.predict_proba(eval_df))


def linear_calibration(source_df, calib_df, eval_df, cat, num):
    em_source = EnergyModel(ENERGY_KIND, cat, num).fit(source_df, source_df["energy_j"])
    qm_source = QualityModel(QUALITY_KIND, cat, num).fit(source_df, source_df["correct"], groups=source_df["task_id"])

    log_pred_calib = np.log(em_source.predict(calib_df))
    log_actual_calib = np.log(calib_df["energy_j"].values)
    A = np.vstack([np.ones_like(log_pred_calib), log_pred_calib]).T
    (a, b), *_ = np.linalg.lstsq(A, log_actual_calib, rcond=None)
    log_pred_eval = np.log(em_source.predict(eval_df))
    pred_e_eval = np.exp(a + b * log_pred_eval)

    raw_q_calib = np.clip(qm_source.predict_proba(calib_df), 1e-6, 1 - 1e-6)
    logit_calib = np.log(raw_q_calib / (1 - raw_q_calib)).reshape(-1, 1)
    raw_q_eval = np.clip(qm_source.predict_proba(eval_df), 1e-6, 1 - 1e-6)
    logit_eval = np.log(raw_q_eval / (1 - raw_q_eval)).reshape(-1, 1)
    calib_labels = calib_df["correct"].values.astype(int)
    if len(np.unique(calib_labels)) < 2:
        # A small k can land on a calibration set that's all-correct (or
        # all-wrong) — Platt scaling needs both classes to fit a slope, so
        # fall back to the uncalibrated source-model probabilities rather
        # than crashing. Small-k instability like this is itself part of
        # what few-shot calibration is supposed to measure, not hidden.
        pred_q_eval = raw_q_eval
    else:
        platt = LogisticRegression().fit(logit_calib, calib_labels)
        pred_q_eval = platt.predict_proba(logit_eval)[:, 1]

    return _eval_metrics(pred_e_eval, eval_df, pred_q_eval)


def no_calibration_baseline(source_df, eval_df, cat, num):
    """Source-trained model applied to the target with zero adaptation —
    the floor both calibration approaches should beat."""
    em = EnergyModel(ENERGY_KIND, cat, num).fit(source_df, source_df["energy_j"])
    qm = QualityModel(QUALITY_KIND, cat, num).fit(source_df, source_df["correct"], groups=source_df["task_id"])
    return _eval_metrics(em.predict(eval_df), eval_df, qm.predict_proba(eval_df))


def run_direction(source_csv, target_csv, source_name, target_name):
    source_df, _, _ = load_experiment_frame(source_csv)
    target_df, _, _ = load_experiment_frame(target_csv)
    cat, num = feature_columns(FEATURE_GROUPS)

    zero_shot = no_calibration_baseline(source_df, target_df, cat, num)
    rows = [{"k": 0, "method": "zero-shot (no calibration)",
             "mae_j": zero_shot[0], "accuracy": zero_shot[1], "log_loss": zero_shot[2]}]

    for k in K_VALUES:
        for method_name, method_fn in [("full refit", full_refit), ("linear calibration", linear_calibration)]:
            trials = []
            for seed in range(N_REPEATS):
                calib_df, eval_df = _split_calib_eval(target_df, k, seed)
                trials.append(method_fn(source_df, calib_df, eval_df, cat, num))
            trials = np.array(trials)
            rows.append({
                "k": k, "method": method_name,
                "mae_j": trials[:, 0].mean(), "mae_j_std": trials[:, 0].std(),
                "accuracy": trials[:, 1].mean(), "accuracy_std": trials[:, 1].std(),
                "log_loss": trials[:, 2].mean(), "log_loss_std": trials[:, 2].std(),
            })

    df = pd.DataFrame(rows)
    df["source"], df["target"] = source_name, target_name
    return df


def main():
    all_results = []
    for source_csv, target_csv, sname, tname in [
        (QWEN_CSV, LLAMA_CSV, "dgx_qwen", "dgx_llama"),
        (LLAMA_CSV, QWEN_CSV, "dgx_llama", "dgx_qwen"),
    ]:
        print(f"\n=== few-shot calibration: {sname} -> {tname} ===")
        df = run_direction(source_csv, target_csv, sname, tname)
        print(df.to_string(index=False))
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    out_path = os.path.join(str(RESULTS_DIR), "fewshot_calibration.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
