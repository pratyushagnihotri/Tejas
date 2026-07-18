"""Repeated nested grouped CV: reports PREDICTION-quality metrics for the
energy/quality ladders (not routing outcomes — that's evaluate_policy.py).

Outer 5-fold x 5 repeats (StratifiedGroupKFold, stratified by
task_type_tier so no fold starves a workload class — 16/40 tasks are
semantic_hard), inner 4-fold plain GroupKFold on each outer-train split
selects the best ladder member (by inner out-of-fold MAE for energy,
inner out-of-fold log loss for quality). A task's rows (3 tiers x 4 runs)
never cross a fold boundary at either level.
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from tejas.cost_model.dataset import feature_columns, load_experiment_frame
from tejas.cost_model.deployment_features import CSV_TO_DEPLOYMENT
from tejas.cost_model.energy_model import ENERGY_MODEL_KINDS, EnergyModel
from tejas.cost_model.quality_model import QUALITY_MODEL_KINDS, QualityModel
from tejas.paths import RESULTS_DIR

N_OUTER_SPLITS = 5
N_OUTER_REPEATS = 5
N_INNER_SPLITS = 4
FEATURE_GROUPS = ("base", "sql_semantic", "interactions")

# 3 of the 6 task_type_tier classes have only 3-8 members (concept_easy=3,
# concept_hard=3, sql_easy=5, sql_hard=5) — StratifiedGroupKFold(5) can't
# stratify those perfectly and warns every time. Expected given the fixed
# 40-task set, not something to act on.
warnings.filterwarnings("ignore", message="The least populated class in y has only")


def outer_splits(df, task_type_tier_by_task=None):
    """Yields (repeat, fold_idx, train_df, test_df) for N_OUTER_REPEATS x
    N_OUTER_SPLITS repeated nested-CV outer folds, group-safe on task_id and
    stratified by task_type_tier. Shared by evaluate_policy.py so both
    modules see identical fold structure."""
    if task_type_tier_by_task is None:
        task_type_tier_by_task = df.drop_duplicates("task_id").set_index("task_id")["task_type_tier"]
    task_ids = df["task_id"].unique()
    task_strat_y = task_type_tier_by_task.loc[task_ids].values
    for repeat in range(N_OUTER_REPEATS):
        splitter = StratifiedGroupKFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=repeat)
        for fold_idx, (train_task_pos, test_task_pos) in enumerate(
            splitter.split(task_ids, task_strat_y, groups=task_ids)
        ):
            train_tasks = set(task_ids[train_task_pos])
            test_tasks = set(task_ids[test_task_pos])
            assert train_tasks.isdisjoint(test_tasks)
            train_df = df[df["task_id"].isin(train_tasks)]
            test_df = df[df["task_id"].isin(test_tasks)]
            yield repeat, fold_idx, train_df, test_df


def _inner_oof(kind, is_quality, train_df, cat, num):
    inner = GroupKFold(n_splits=N_INNER_SPLITS)
    oof = np.full(len(train_df), np.nan)
    for itr_pos, ite_pos in inner.split(train_df, groups=train_df["task_id"]):
        itr, ite = train_df.iloc[itr_pos], train_df.iloc[ite_pos]
        if is_quality:
            m = QualityModel(kind, cat, num).fit(itr, itr["correct"], groups=itr["task_id"])
            pred = m.predict_proba(ite)
        else:
            m = EnergyModel(kind, cat, num).fit(itr, itr["energy_j"])
            pred = m.predict(ite)
        oof[ite_pos] = pred
    assert not np.isnan(oof).any()
    return oof


def _select_best_energy_kind(train_df, cat, num):
    scores = {}
    for kind in ENERGY_MODEL_KINDS:
        oof = _inner_oof(kind, False, train_df, cat, num)
        scores[kind] = float(np.mean(np.abs(oof - train_df["energy_j"])))  # MAE
    return min(scores, key=scores.get), scores


def _select_best_quality_kind(train_df, cat, num):
    scores = {}
    for kind in QUALITY_MODEL_KINDS:
        oof = _inner_oof(kind, True, train_df, cat, num)
        p = np.clip(oof, 1e-6, 1 - 1e-6)
        scores[kind] = float(log_loss(train_df["correct"], p, labels=[0, 1]))
    return min(scores, key=scores.get), scores


def _per_task_energy_diagnostics(test_df, pred_energy):
    """Per-task Spearman correlation between predicted/actual energy across
    that task's tiers (avg'd), and whether the (quality-blind) min-energy
    tier was correctly identified."""
    d = test_df.copy()
    d["pred_energy"] = pred_energy
    spearmans, hit = [], []
    for task_id, g in d.groupby("task_id"):
        if g["model_tier"].nunique() < 2:
            continue
        rho, _ = spearmanr(g["pred_energy"], g["energy_j"])
        if not np.isnan(rho):
            spearmans.append(rho)
        actual_min = g.loc[g["energy_j"].idxmin(), "model_tier"]
        pred_min = g.loc[g["pred_energy"].idxmin(), "model_tier"]
        hit.append(actual_min == pred_min)
    return float(np.mean(spearmans)) if spearmans else float("nan"), float(np.mean(hit))


def _calibration_diagnostics(y_true, p_pred):
    p = np.clip(p_pred, 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        cal = LogisticRegression().fit(logit_p, y_true)
        slope, intercept = float(cal.coef_[0][0]), float(cal.intercept_[0])
    except ValueError:
        slope, intercept = float("nan"), float("nan")  # e.g. all-one-class fold
    frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=5, strategy="quantile")
    return {
        "brier": float(brier_score_loss(y_true, p)),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "reliability_bins": list(zip(mean_pred.tolist(), frac_pos.tolist())),
    }


def _delta_q_error(test_df, pred_quality):
    """Mean absolute error between predicted and actual quality *difference
    from the reference tier* — closer to what policy_optimizer.py actually
    consumes than raw probability accuracy, since routing decisions compare
    Q_hat(m) - Q_hat(reference) against delta, not Q_hat(m) in isolation."""
    d = test_df.copy()
    d["pred_quality"] = pred_quality
    g = d.groupby(["task_id", "model_tier"]).agg(
        measured_quality=("correct", "mean"),
        pred_quality=("pred_quality", "mean"),
        reference_tier=("reference_tier", "first"),
    ).reset_index()

    errors = []
    for task_id, tg in g.groupby("task_id"):
        ref_tier = tg["reference_tier"].iloc[0]
        ref_row = tg[tg["model_tier"] == ref_tier].iloc[0]
        for _, row in tg.iterrows():
            if row["model_tier"] == ref_tier:
                continue
            actual_diff = row["measured_quality"] - ref_row["measured_quality"]
            pred_diff = row["pred_quality"] - ref_row["pred_quality"]
            errors.append(abs(actual_diff - pred_diff))
    return float(np.mean(errors)) if errors else float("nan")


def run_nested_cv(csv_path):
    df, task_tier, runs = load_experiment_frame(csv_path)
    cat, num = feature_columns(FEATURE_GROUPS)

    outer_results = []
    oof_records = []
    for repeat, fold_idx, train_df, test_df in outer_splits(df):
        test_tasks = set(test_df["task_id"])
        best_e_kind, e_scores = _select_best_energy_kind(train_df, cat, num)
        best_q_kind, q_scores = _select_best_quality_kind(train_df, cat, num)

        em = EnergyModel(best_e_kind, cat, num).fit(train_df, train_df["energy_j"])
        qm = QualityModel(best_q_kind, cat, num).fit(train_df, train_df["correct"], groups=train_df["task_id"])

        pred_energy = em.predict(test_df)
        pred_quality = qm.predict_proba(test_df)

        mae = float(np.mean(np.abs(pred_energy - test_df["energy_j"])))
        mdape = float(np.median(np.abs(pred_energy - test_df["energy_j"]) / test_df["energy_j"]) * 100)
        rmse_log = float(np.sqrt(np.mean((np.log(pred_energy) - np.log(test_df["energy_j"])) ** 2)))
        ss_res = np.sum((pred_energy - test_df["energy_j"]) ** 2)
        ss_tot = np.sum((test_df["energy_j"] - test_df["energy_j"].mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        avg_spearman, min_tier_hit_rate = _per_task_energy_diagnostics(test_df, pred_energy)

        accuracy = float(np.mean((pred_quality >= 0.5) == test_df["correct"]))
        cal = _calibration_diagnostics(test_df["correct"].values, pred_quality)
        delta_q_error = _delta_q_error(test_df, pred_quality)

        outer_results.append({
            "repeat": repeat, "fold": fold_idx,
            "best_energy_kind": best_e_kind, "best_quality_kind": best_q_kind,
            "energy_mae_j": mae, "energy_mdape_pct": mdape, "energy_rmse_log": rmse_log,
            "energy_r2": r2, "energy_avg_spearman": avg_spearman,
            "energy_min_tier_hit_rate": min_tier_hit_rate,
            "quality_accuracy": accuracy, "quality_brier": cal["brier"],
            "quality_log_loss": cal["log_loss"],
            "quality_calibration_slope": cal["calibration_slope"],
            "quality_calibration_intercept": cal["calibration_intercept"],
            "quality_delta_q_error": delta_q_error,
            "n_test_tasks": len(test_tasks),
        })

        # Pooled out-of-fold predictions (one row per test row) — the raw
        # material for a predicted-vs-actual figure. These are genuinely
        # held-out (the fold's models never saw these rows in training),
        # unlike an in-sample scatter would be.
        oof_records.append(pd.DataFrame({
            "repeat": repeat, "fold": fold_idx,
            "task_id": test_df["task_id"].values, "model_tier": test_df["model_tier"].values,
            "task_type_tier": test_df["task_type_tier"].values,
            "actual_energy_j": test_df["energy_j"].values, "pred_energy_j": pred_energy,
            "actual_correct": test_df["correct"].values.astype(int), "pred_quality": pred_quality,
        }))

    return pd.DataFrame(outer_results), pd.concat(oof_records, ignore_index=True)


def summarize(results_df, experiment_name):
    print(f"\n=== {experiment_name}: prediction-quality metrics "
          f"({N_OUTER_SPLITS}x{N_OUTER_REPEATS} repeated nested grouped CV) ===")
    metric_cols = ["energy_mae_j", "energy_mdape_pct", "energy_rmse_log", "energy_r2",
                   "energy_avg_spearman", "energy_min_tier_hit_rate",
                   "quality_accuracy", "quality_brier", "quality_log_loss",
                   "quality_calibration_slope", "quality_calibration_intercept",
                   "quality_delta_q_error"]
    for col in metric_cols:
        vals = results_df[col].dropna()
        print(f"  {col:<28} mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"median={vals.median():.4f}  (n={len(vals)}/{len(results_df)})")
    print("  energy model selection frequency:")
    for kind, frac in results_df["best_energy_kind"].value_counts(normalize=True).items():
        print(f"    {kind:<12} {frac:.0%}")
    print("  quality model selection frequency:")
    for kind, frac in results_df["best_quality_kind"].value_counts(normalize=True).items():
        print(f"    {kind:<12} {frac:.0%}")


def main():
    for csv_name in CSV_TO_DEPLOYMENT:
        path = os.path.join(str(RESULTS_DIR), csv_name)
        results_df, oof_df = run_nested_cv(path)
        summarize(results_df, CSV_TO_DEPLOYMENT[csv_name])
        out_path = os.path.join(str(RESULTS_DIR), f"model_eval_{CSV_TO_DEPLOYMENT[csv_name]}.csv")
        results_df.to_csv(out_path, index=False)
        oof_path = os.path.join(str(RESULTS_DIR), f"oof_predictions_{CSV_TO_DEPLOYMENT[csv_name]}.csv")
        oof_df.to_csv(oof_path, index=False)
        print(f"  wrote {oof_path}")
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
