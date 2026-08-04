"""Does the energy model contribute anything beyond the quality gate? Four
rows, same -2pp aggregate quality budget and same nested-CV methodology as
the main routing result (evaluate_policy.py), varying only which model(s)
decide the tier:

  Fixed downgrade        no energy model, no quality model (the `scarce`
                          policy from static_policies.py: always downgrade
                          one fixed tier).
  Smallest predicted-safe  quality model only: among tiers the quality
                          model predicts are safe within delta, pick the
                          smallest by parameter count. The energy model
                          plays no role in the choice.
  Tejas                  both models: among the same quality-safe tiers,
                          pick the one with minimum *predicted* energy.
  Oracle                 both, measured directly (no prediction) — the
                          upper bound.

If "Smallest predicted-safe" performs about as well as Tejas, the energy
model isn't earning its place — the quality gate plus "prefer cheaper" is
doing all the work, and a reviewer could reasonably say the smallest model
is just naturally the cheapest. Run after evaluate_policy.py, since this
reuses its exact per-repeat/pooled-bootstrap reporting shape.
"""

import os

import pandas as pd

from tejas.benchmarking.results_loader import ENERGY_AWARE_ROUTE, STATIC_ROUTE, load_raw_results
from tejas.cost_model.dataset import load_experiment_frame
from tejas.cost_model.deployment_features import CSV_TO_DEPLOYMENT
from tejas.paths import RESULTS_DIR
from tejas.routing.evaluate_policy import QUALITY_BUDGET_PP, run_policy_evaluation, summarize
from tejas.routing.oracle import _task_tier_stats, global_budget_oracle
from tejas.routing.static_policies import mean_std, score_policy_for_run


def _fixed_downgrade_row(csv_path):
    by_run_task_model, task_tier_map, runs = load_raw_results(csv_path)
    per_run = [score_policy_for_run(by_run_task_model, task_tier_map, r, lambda t: ENERGY_AWARE_ROUTE["scarce"][t])
               for r in runs]
    q_mean, _ = mean_std([r["quality_pct"] for r in per_run])
    e_mean, _ = mean_std([r["total_energy_j"] for r in per_run])
    static_per_run = [score_policy_for_run(by_run_task_model, task_tier_map, r, lambda t: STATIC_ROUTE[t])
                       for r in runs]
    static_q_mean, _ = mean_std([r["quality_pct"] for r in static_per_run])
    static_e_mean, _ = mean_std([r["total_energy_j"] for r in static_per_run])
    return {
        "policy": "Fixed downgrade", "energy_model": "No", "quality_model": "No",
        "quality_pct_change": q_mean - static_q_mean,
        "pct_energy_saved": 100 * (1 - e_mean / static_e_mean),
    }


def _from_run_policy_evaluation(csv_path, selection_rule, label, energy_model_label, quality_model_label):
    result = run_policy_evaluation(csv_path, selection_rule=selection_rule)
    per_repeat_df, _ = summarize(result, f"{label} ({selection_rule})")
    return {
        "policy": label, "energy_model": energy_model_label, "quality_model": quality_model_label,
        "quality_pct_change": per_repeat_df["quality_diff_pp_vs_static"].mean(),
        "pct_energy_saved": per_repeat_df["pct_energy_saved_vs_static"].mean(),
    }


def _oracle_row(csv_path):
    df, _, _ = load_experiment_frame(csv_path)
    ref_tiers = df.drop_duplicates("task_id").set_index("task_id")["reference_tier"].to_dict()
    task_ids = df["task_id"].unique().tolist()
    stats = _task_tier_stats(df)
    ref_quality = sum(stats[(tid, ref_tiers[tid])]["quality"] for tid in task_ids) / len(task_ids) * 100
    ref_energy = sum(stats[(tid, ref_tiers[tid])]["energy"] for tid in task_ids)
    _, budget_energy, budget_quality = global_budget_oracle(df, QUALITY_BUDGET_PP)
    return {
        "policy": "Oracle", "energy_model": "Measured", "quality_model": "Measured",
        "quality_pct_change": budget_quality * 100 - ref_quality,
        "pct_energy_saved": 100 * (1 - budget_energy / ref_energy),
    }


def run_ablation(csv_path, deployment_key):
    rows = [
        _fixed_downgrade_row(csv_path),
        _from_run_policy_evaluation(csv_path, "smallest_tier", "Smallest predicted-safe", "No", "Yes"),
        _from_run_policy_evaluation(csv_path, "min_energy", "Tejas", "Yes", "Yes"),
        _oracle_row(csv_path),
    ]
    table = pd.DataFrame(rows)
    table["deployment"] = deployment_key
    out_path = os.path.join(str(RESULTS_DIR), f"ablation_energy_model_{deployment_key}.csv")
    table.to_csv(out_path, index=False)
    print(f"\n=== {deployment_key}: energy-model ablation (at the -2pp budget) ===")
    print(table.to_string(index=False))
    print(f"  wrote {out_path}")
    return table


def main():
    for csv_name, deployment_key in CSV_TO_DEPLOYMENT.items():
        path = os.path.join(str(RESULTS_DIR), csv_name)
        run_ablation(path, deployment_key)


if __name__ == "__main__":
    main()
