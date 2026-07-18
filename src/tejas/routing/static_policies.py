"""Simulates 5 routing policies from results/raw_results.csv by table lookup
(no need to re-run anything: every model already answered every task, so a
policy is just "which model's row do we pick for each task"). Aggregates
across all runs found in the CSV's 'run' column, reporting mean +/- stdev
so results with --repeats > 1 get error bars.
"""

import argparse
import csv
import os
from collections import defaultdict

from tejas.benchmarking.results_loader import ENERGY_AWARE_ROUTE, STATIC_ROUTE, load_raw_results
from tejas.paths import RESULTS_DIR

DEFAULT_RAW = os.path.join(str(RESULTS_DIR), "raw_results.csv")


def calibrate_scarce_route(by_run_task_model, task_tier, runs, gap_threshold_pp=10.0):
    """Builds a 'calibrated scarce' route: for each tier, only take the
    downgrade ENERGY_AWARE_ROUTE['scarce'] proposes if the *measured* quality
    gap between the static tier and the downgraded tier is small (<=
    gap_threshold_pp). Otherwise it keeps the static (abundant) tier even
    under scarcity — no point burning less energy if it costs a lot of
    quality. Calibrated from this dataset's own data (retrospective/oracle,
    not a zero-shot deployable rule) — motivates a learned cost model
    (Option C) that could estimate this without needing to measure it first.
    """
    acc = defaultdict(lambda: [0, 0])  # (model_tier, task_tier) -> [correct, n]
    for (run, task_id, model_tier), row in by_run_task_model.items():
        tier = task_tier[task_id]
        acc[(model_tier, tier)][1] += 1
        if row["correct"] == "True":
            acc[(model_tier, tier)][0] += 1

    def pct(model_tier, tier):
        c, n = acc[(model_tier, tier)]
        return 100 * c / n if n else None

    calibrated = {}
    for tier in ("easy", "medium", "hard"):
        static_tier = STATIC_ROUTE[tier]
        scarce_tier = ENERGY_AWARE_ROUTE["scarce"][tier]
        if static_tier == scarce_tier:
            calibrated[tier] = static_tier
            continue
        static_acc = pct(static_tier, tier)
        scarce_acc = pct(scarce_tier, tier)
        gap = (static_acc - scarce_acc) if (static_acc is not None and scarce_acc is not None) else None
        if gap is not None and gap <= gap_threshold_pp:
            calibrated[tier] = scarce_tier  # downgrade is cheap enough — take it
        else:
            calibrated[tier] = static_tier  # downgrade too costly — stay put
    return calibrated


def score_policy_for_run(by_run_task_model, task_tier, run, route_fn):
    n = correct_n = 0
    energy_total = latency_total = 0.0
    missing_energy = False
    for task_id, tier in task_tier.items():
        chosen_tier = route_fn(tier)
        row = by_run_task_model[(run, task_id, chosen_tier)]
        n += 1
        correct_n += 1 if row["correct"] == "True" else 0
        if row["energy_j"]:
            energy_total += float(row["energy_j"])
        else:
            missing_energy = True
        latency_total += float(row["latency_s"])
    return {
        "quality_pct": 100 * correct_n / n,
        "total_energy_j": None if missing_energy else energy_total,
        "total_latency_s": latency_total,
    }


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, var ** 0.5


def fmt(mean, std, decimals=1):
    if mean is None:
        return "n/a"
    if std is None or std == 0.0:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_RAW,
                         help="path to a raw_results CSV (default: results/raw_results.csv)")
    parser.add_argument("--output", default=None,
                         help="where to write policy_summary CSV (default: alongside --input)")
    args = parser.parse_args()
    out_path = args.output or os.path.join(
        os.path.dirname(args.input),
        os.path.basename(args.input).replace("raw_results", "policy_summary"))

    by_run_task_model, task_tier, runs = load_raw_results(args.input)
    calibrated_route = calibrate_scarce_route(by_run_task_model, task_tier, runs)

    policies = {
        "always-largest": lambda t: "large",
        "always-smallest": lambda t: "small",
        "static-cost-router": lambda t: STATIC_ROUTE[t],
        "energy-aware (abundant)": lambda t: ENERGY_AWARE_ROUTE["abundant"][t],
        "energy-aware (scarce)": lambda t: ENERGY_AWARE_ROUTE["scarce"][t],
        "energy-aware (scarce, calibrated)": lambda t: calibrated_route[t],
    }
    print(f"Calibrated scarce route (only downgrade if measured gap <= 10pp): {calibrated_route}\n")

    print(f"Aggregating over {len(runs)} run(s): {', '.join(runs)}\n")

    summary = {}
    header = f"{'policy':<26}{'quality %':>18}{'energy J':>22}{'latency s':>18}"
    print(header)
    print("-" * len(header))
    for name, fn in policies.items():
        per_run = [score_policy_for_run(by_run_task_model, task_tier, r, fn) for r in runs]
        q_mean, q_std = mean_std([r["quality_pct"] for r in per_run])
        e_mean, e_std = mean_std([r["total_energy_j"] for r in per_run])
        l_mean, l_std = mean_std([r["total_latency_s"] for r in per_run])
        summary[name] = {
            "quality_pct_mean": q_mean, "quality_pct_std": q_std,
            "total_energy_j_mean": e_mean, "total_energy_j_std": e_std,
            "total_latency_s_mean": l_mean, "total_latency_s_std": l_std,
            "n_runs": len(per_run),
        }
        print(f"{name:<26}{fmt(q_mean, q_std):>18}{fmt(e_mean, e_std):>22}{fmt(l_mean, l_std):>18}")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "quality_pct_mean", "quality_pct_std",
                          "total_energy_j_mean", "total_energy_j_std",
                          "total_latency_s_mean", "total_latency_s_std", "n_runs"])
        for name, s in summary.items():
            writer.writerow([name, s["quality_pct_mean"], s["quality_pct_std"],
                              s["total_energy_j_mean"], s["total_energy_j_std"],
                              s["total_latency_s_mean"], s["total_latency_s_std"], s["n_runs"]])
    print(f"\nWrote {out_path}")

    if len(runs) < 3:
        print(f"\nOnly {len(runs)} run(s) so far — run with --repeats {max(0, 3 - len(runs))} "
              "more to get to 3-4 for meaningful error bars.")

    print("\nNote: 'static-cost-router' and 'energy-aware (abundant)' route identically "
          "by construction — the interesting comparison is static-cost-router vs. "
          "energy-aware (scarce), which shows what the router saves by reacting to "
          "energy scarcity instead of routing the same way regardless of energy state.")


if __name__ == "__main__":
    main()
