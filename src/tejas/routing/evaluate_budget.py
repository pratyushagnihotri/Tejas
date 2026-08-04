"""Budget-vs-quality frontier: what does the router do at 100/90/.../40% of
static-cost-router's energy, not just at the single delta the main routing
result (evaluate_policy.py) settles on? This is the bridge from "Tejas saves
energy within a fixed quality budget" to computation-by-energy's actual
claim that behavior should track *how much* energy is available.

Per outer fold, the energy/quality model kinds are chosen once by the same
standalone-accuracy procedure evaluate_models.py uses (inner-OOF MAE/log-loss)
— not re-selected per budget point, which would multiply the search space
for a question (how does behavior change with budget) that is orthogonal to
which model kind predicts best. Predictions are made once per outer-test
fold; the budget-constrained knapsack (oracle.budget_constrained_select) is
then swept cheaply over those fixed predictions for every budget fraction,
mirroring how evaluate_policy.py sweeps delta cheaply over fixed inner-OOF
predictions rather than refitting per delta.

The budget itself is defined in *measured* joules (a fraction of that fold's
measured static-cost-router total) since a real deployment's energy budget
is a real quantity, not a predicted one; the knapsack that decides which
tiers to use under that budget runs on *predicted* (energy, quality) pairs,
since a real deployment doesn't know measured outcomes in advance. The
achieved energy/quality reported back is always measured — so prediction
error can and does show up as the achieved point missing its nominal
budget, which is itself part of what this figure is for.
"""

import os

import numpy as np
import pandas as pd

from tejas.benchmarking.results_loader import ENERGY_AWARE_ROUTE
from tejas.cost_model.dataset import feature_columns, load_experiment_frame
from tejas.cost_model.deployment_features import CSV_TO_DEPLOYMENT
from tejas.cost_model.energy_model import EnergyModel
from tejas.cost_model.evaluate_models import (
    FEATURE_GROUPS, N_OUTER_REPEATS, _select_best_energy_kind, _select_best_quality_kind, outer_splits,
)
from tejas.cost_model.quality_model import QualityModel
from tejas.paths import RESULTS_DIR
from tejas.routing.oracle import _candidates_for_task, _task_tier_stats, budget_constrained_select, energy_budget_frontier

BUDGET_FRACS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
RNG_SEED = 0
BOOTSTRAP_N = 2000


def _reference_tiers(df):
    return df.drop_duplicates("task_id").set_index("task_id")["reference_tier"].to_dict()


def learned_budget_curve(csv_path):
    """Returns (per_repeat_df, pooled_df): per_repeat_df has one row per
    (budget_frac, repeat) with the achieved energy/quality after combining
    that repeat's 5 outer-test folds into the full 40-task assignment;
    pooled_df has one row per (budget_frac, repeat, task_id), the raw
    material for a bootstrap CI per budget point."""
    df, task_tier, runs = load_experiment_frame(csv_path)
    cat, num = feature_columns(FEATURE_GROUPS)
    measured_stats = _task_tier_stats(df)
    ref_tiers = _reference_tiers(df)
    all_task_ids = set(df["task_id"].unique())

    # {frac: {repeat: {task_id: chosen_tier}}}
    chosen_by_frac_repeat = {frac: {r: {} for r in range(N_OUTER_REPEATS)} for frac in BUDGET_FRACS}

    for repeat, fold_idx, train_df, test_df in outer_splits(df):
        best_e_kind, _ = _select_best_energy_kind(train_df, cat, num)
        best_q_kind, _ = _select_best_quality_kind(train_df, cat, num)
        em = EnergyModel(best_e_kind, cat, num).fit(train_df, train_df["energy_j"])
        qm = QualityModel(best_q_kind, cat, num).fit(train_df, train_df["correct"], groups=train_df["task_id"])

        test_tasks = test_df["task_id"].unique().tolist()
        per_task_options_pred = []
        for tid in test_tasks:
            g = test_df[test_df["task_id"] == tid]
            rows_by_tier = {t: g[g["model_tier"] == t].iloc[[0]] for t in g["model_tier"].unique()}
            _ref_tier, cand_tiers = _candidates_for_task(test_df, tid)
            options = []
            for t in cand_tiers:
                e_hat = float(em.predict(rows_by_tier[t])[0])
                q_hat = float(qm.predict_proba(rows_by_tier[t])[0])
                options.append((t, e_hat, q_hat))
            per_task_options_pred.append((tid, options))

        fold_ref_total = sum(measured_stats[(tid, ref_tiers[tid])]["energy"] for tid in test_tasks)

        for frac in BUDGET_FRACS:
            budget = frac * fold_ref_total
            chosen = budget_constrained_select(per_task_options_pred, budget)
            chosen_by_frac_repeat[frac][repeat].update(chosen)

    per_repeat_rows = []
    pooled_rows = []
    for frac in BUDGET_FRACS:
        for repeat in range(N_OUTER_REPEATS):
            chosen = chosen_by_frac_repeat[frac][repeat]
            assert set(chosen.keys()) == all_task_ids, \
                f"frac={frac} repeat={repeat}: outer-test folds didn't cover all 40 tasks exactly once"
            achieved_energy = sum(measured_stats[(tid, chosen[tid])]["energy"] for tid in chosen)
            achieved_quality = sum(measured_stats[(tid, chosen[tid])]["quality"] for tid in chosen) / len(chosen)
            ref_total = sum(measured_stats[(tid, ref_tiers[tid])]["energy"] for tid in chosen)
            ref_quality = sum(measured_stats[(tid, ref_tiers[tid])]["quality"] for tid in chosen) / len(chosen)
            per_repeat_rows.append({
                "budget_frac": frac, "repeat": repeat,
                "achieved_energy_j": achieved_energy, "achieved_energy_pct_of_static": 100 * achieved_energy / ref_total,
                "achieved_quality_pct": achieved_quality * 100,
                "quality_diff_pp_vs_static": (achieved_quality - ref_quality) * 100,
            })
            for tid, tier in chosen.items():
                pooled_rows.append({
                    "budget_frac": frac, "repeat": repeat, "task_id": tid,
                    "energy_chosen": measured_stats[(tid, tier)]["energy"],
                    "energy_ref": measured_stats[(tid, ref_tiers[tid])]["energy"],
                    "quality_chosen": measured_stats[(tid, tier)]["quality"],
                    "quality_ref": measured_stats[(tid, ref_tiers[tid])]["quality"],
                })

    return pd.DataFrame(per_repeat_rows), pd.DataFrame(pooled_rows)


def _bootstrap_ci(pooled_df, frac, rng):
    sub = pooled_df[pooled_df["budget_frac"] == frac]
    n = len(sub)
    idx = rng.integers(0, n, size=(BOOTSTRAP_N, n))
    e_chosen, e_ref = sub["energy_chosen"].values, sub["energy_ref"].values
    q_chosen, q_ref = sub["quality_chosen"].values, sub["quality_ref"].values
    boot_energy_pct = 100 * e_chosen[idx].sum(axis=1) / e_ref[idx].sum(axis=1)
    boot_quality_pct = 100 * q_chosen[idx].mean(axis=1)
    return {
        "energy_pct_ci": tuple(np.percentile(boot_energy_pct, [2.5, 97.5])),
        "quality_pct_ci": tuple(np.percentile(boot_quality_pct, [2.5, 97.5])),
    }


def scarce_point(df):
    """The fixed-downgrade policy has no budget knob — a single (energy%,
    quality%) point, not a curve, computed directly from measured stats
    (no CV needed for a deterministic rule)."""
    stats = _task_tier_stats(df)
    task_ids = df["task_id"].unique().tolist()
    task_tier_map = df.drop_duplicates("task_id").set_index("task_id")["task_tier"].to_dict()
    ref_tiers = _reference_tiers(df)
    chosen = {tid: ENERGY_AWARE_ROUTE["scarce"][task_tier_map[tid]] for tid in task_ids}
    achieved_energy = sum(stats[(tid, chosen[tid])]["energy"] for tid in task_ids)
    achieved_quality = sum(stats[(tid, chosen[tid])]["quality"] for tid in task_ids) / len(task_ids)
    ref_total = sum(stats[(tid, ref_tiers[tid])]["energy"] for tid in task_ids)
    return 100 * achieved_energy / ref_total, achieved_quality * 100


def run_budget_sweep(csv_path, deployment_key):
    df, _, _ = load_experiment_frame(csv_path)
    per_repeat_df, pooled_df = learned_budget_curve(csv_path)
    rng = np.random.default_rng(RNG_SEED)

    rows = []
    summary = per_repeat_df.groupby("budget_frac").agg(
        tejas_energy_pct_mean=("achieved_energy_pct_of_static", "mean"),
        tejas_energy_pct_std=("achieved_energy_pct_of_static", "std"),
        tejas_quality_pct_mean=("achieved_quality_pct", "mean"),
        tejas_quality_pct_std=("achieved_quality_pct", "std"),
    ).reset_index()
    for _, row in summary.iterrows():
        ci = _bootstrap_ci(pooled_df, row["budget_frac"], rng)
        rows.append({
            "budget_frac": row["budget_frac"],
            "tejas_energy_pct_mean": row["tejas_energy_pct_mean"],
            "tejas_energy_pct_std": row["tejas_energy_pct_std"],
            "tejas_quality_pct_mean": row["tejas_quality_pct_mean"],
            "tejas_quality_pct_std": row["tejas_quality_pct_std"],
            "tejas_energy_pct_ci_lo": ci["energy_pct_ci"][0], "tejas_energy_pct_ci_hi": ci["energy_pct_ci"][1],
            "tejas_quality_pct_ci_lo": ci["quality_pct_ci"][0], "tejas_quality_pct_ci_hi": ci["quality_pct_ci"][1],
        })
    curve_df = pd.DataFrame(rows)

    oracle_frontier, ref_total = energy_budget_frontier(df, BUDGET_FRACS)
    for i, frac in enumerate(BUDGET_FRACS):
        _, e, q = oracle_frontier[frac]
        curve_df.loc[curve_df["budget_frac"] == frac, "oracle_energy_pct"] = 100 * e / ref_total
        curve_df.loc[curve_df["budget_frac"] == frac, "oracle_quality_pct"] = q * 100

    curve_df["deployment"] = deployment_key
    scarce_energy_pct, scarce_quality_pct = scarce_point(df)

    out_path = os.path.join(str(RESULTS_DIR), f"budget_sweep_{deployment_key}.csv")
    curve_df.to_csv(out_path, index=False)

    scarce_path = os.path.join(str(RESULTS_DIR), f"budget_sweep_scarce_point_{deployment_key}.csv")
    pd.DataFrame([{"deployment": deployment_key, "energy_pct": scarce_energy_pct, "quality_pct": scarce_quality_pct}]).to_csv(
        scarce_path, index=False)

    print(f"\n=== {deployment_key}: budget-vs-quality frontier ===")
    print(curve_df.to_string(index=False))
    print(f"  scarce (fixed downgrade, single point): energy {scarce_energy_pct:.1f}% of static, "
          f"quality {scarce_quality_pct:.1f}%")
    print(f"  wrote {out_path}")
    print(f"  wrote {scarce_path}")
    return curve_df


def main():
    for csv_name, deployment_key in CSV_TO_DEPLOYMENT.items():
        path = os.path.join(str(RESULTS_DIR), csv_name)
        run_budget_sweep(path, deployment_key)


if __name__ == "__main__":
    main()
