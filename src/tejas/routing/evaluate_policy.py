"""Joint config selection (inner loop, measured inner-OOF outcomes — not
predicted energy) + within-deployment routing outcomes: quality/energy vs.
static-cost-router, both oracles, bootstrap CIs, violation rates.

Per outer fold: the inner loop computes inner-OOF predictions once per
(energy_kind x quality_kind) pair on outer-*training* tasks, then sweeps
delta cheaply over those fixed predictions. For each (energy_kind,
quality_kind, delta) configuration, the resulting per-task tier choices are
scored against *measured* inner-OOF rows (not predictions) — configurations
whose one-sided lower-95%-bootstrap-CI quality-diff is >= -2pp are kept;
among those, the one with minimum *measured* inner-OOF energy wins (ties:
smaller delta, simpler energy model, simpler quality model). An
always-reference fallback config (0% savings, 0 quality diff) is always in
the candidate set so a feasible choice always exists. That fold-and-repeat
-specific configuration is then applied once, using models refit on the
*full* outer-training data, to the untouched outer-test tasks.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from tejas.benchmarking import results_loader
from tejas.cost_model.dataset import feature_columns, load_experiment_frame
from tejas.cost_model.deployment_features import CSV_TO_DEPLOYMENT, param_count_b
from tejas.cost_model.energy_model import ENERGY_MODEL_KINDS, EnergyModel
from tejas.cost_model.evaluate_models import N_INNER_SPLITS, N_OUTER_REPEATS, FEATURE_GROUPS, outer_splits
from tejas.cost_model.quality_model import QUALITY_MODEL_KINDS, QualityModel
from tejas.paths import RESULTS_DIR
from tejas.routing.oracle import _task_tier_stats, global_budget_oracle, strict_oracle
from tejas.routing.policy_optimizer import select_from_rows
from tejas.routing.static_policies import calibrate_scarce_route, mean_std, score_policy_for_run

DELTA_GRID = np.round(np.arange(0.0, 0.41, 0.02), 2)
QUALITY_BUDGET_PP = -2.0
INNER_BOOTSTRAP_N = 500
FINAL_BOOTSTRAP_N = 2000
RNG_SEED = 0

_ENERGY_RANK = {k: i for i, k in enumerate(ENERGY_MODEL_KINDS)}
_QUALITY_RANK = {k: i for i, k in enumerate(QUALITY_MODEL_KINDS)}


def _lookup_rows(df):
    return {(r.run, r.task_id, r.model_tier): (bool(r.correct), float(r.energy_j)) for r in df.itertuples()}


def _reference_tiers(df):
    return df.drop_duplicates("task_id").set_index("task_id")["reference_tier"].to_dict()


def _candidate_tiers(df, task_id):
    g = df[df["task_id"] == task_id]
    ref_tier = g["reference_tier"].iloc[0]
    ref_param = g.loc[g["model_tier"] == ref_tier, "param_count_b"].iloc[0]
    tiers = g["model_tier"].unique().tolist()
    return ref_tier, [t for t in tiers if g.loc[g["model_tier"] == t, "param_count_b"].iloc[0] <= ref_param]


def _bootstrap_mean(values, n=INNER_BOOTSTRAP_N, rng=None):
    rng = rng or np.random.default_rng(RNG_SEED)
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    return means


def _inner_oof_predictions(train_df, cat, num, random_state):
    """One shared 4-fold split for both ladders. Returns
    {(task_id, model_tier): {kind: pred}} for energy and quality."""
    inner = GroupKFold(n_splits=N_INNER_SPLITS)
    energy_oof = pd.DataFrame(index=train_df.index, columns=ENERGY_MODEL_KINDS, dtype=float)
    quality_oof = pd.DataFrame(index=train_df.index, columns=QUALITY_MODEL_KINDS, dtype=float)

    for itr_pos, ite_pos in inner.split(train_df, groups=train_df["task_id"]):
        itr, ite = train_df.iloc[itr_pos], train_df.iloc[ite_pos]
        for kind in ENERGY_MODEL_KINDS:
            m = EnergyModel(kind, cat, num).fit(itr, itr["energy_j"])
            energy_oof.iloc[ite_pos, energy_oof.columns.get_loc(kind)] = m.predict(ite)
        for kind in QUALITY_MODEL_KINDS:
            m = QualityModel(kind, cat, num).fit(itr, itr["correct"], groups=itr["task_id"])
            quality_oof.iloc[ite_pos, quality_oof.columns.get_loc(kind)] = m.predict_proba(ite)

    assert not energy_oof.isnull().any().any() and not quality_oof.isnull().any().any()

    pred_by_task_tier = {}
    key_df = train_df[["task_id", "model_tier"]].reset_index(drop=True)
    energy_oof = energy_oof.reset_index(drop=True)
    quality_oof = quality_oof.reset_index(drop=True)
    for (task_id, tier), g in key_df.groupby(["task_id", "model_tier"]):
        idx = g.index[0]  # predictions are identical across runs (same features) — take one
        pred_by_task_tier[(task_id, tier)] = {
            "energy": energy_oof.loc[idx].to_dict(),
            "quality": quality_oof.loc[idx].to_dict(),
        }
    return pred_by_task_tier


def _select_config(train_df, cat, num, measured_stats, task_ids, rng, selection_rule="min_energy"):
    """selection_rule: "min_energy" is Tejas's actual rule (argmin predicted
    energy among quality-safe tiers); "smallest_tier" instead picks the
    smallest quality-safe tier by parameter count, never consulting the
    energy model's predictions at all. The latter exists only for the
    ablation that asks whether the energy model contributes anything beyond
    what the quality gate plus "prefer cheaper" already gives for free —
    see policy_optimizer.select_from_rows's docstring for the same
    distinction at the per-task level."""
    pred = _inner_oof_predictions(train_df, cat, num, RNG_SEED)
    ref_tiers = _reference_tiers(train_df)
    candidates = {tid: _candidate_tiers(train_df, tid) for tid in task_ids}
    param_by_task_tier = {
        (tid, t): train_df.loc[(train_df["task_id"] == tid) & (train_df["model_tier"] == t),
                                "param_count_b"].iloc[0]
        for tid in task_ids for t in candidates[tid][1]
    }

    configs = [{"energy_kind": None, "quality_kind": None, "delta": None, "is_fallback": True}]
    for e_kind in ENERGY_MODEL_KINDS:
        for q_kind in QUALITY_MODEL_KINDS:
            for delta in DELTA_GRID:
                configs.append({"energy_kind": e_kind, "quality_kind": q_kind,
                                 "delta": float(delta), "is_fallback": False})

    feasible_configs = []
    for cfg in configs:
        if cfg["is_fallback"]:
            chosen = {tid: ref_tiers[tid] for tid in task_ids}
        else:
            chosen = {}
            for tid in task_ids:
                ref_tier, cand_tiers = candidates[tid]
                q_ref = pred[(tid, ref_tier)]["quality"][cfg["quality_kind"]]
                feasible_t = [t for t in cand_tiers
                              if pred[(tid, t)]["quality"][cfg["quality_kind"]] >= q_ref - cfg["delta"]]
                if ref_tier not in feasible_t:
                    feasible_t.append(ref_tier)
                if selection_rule == "smallest_tier":
                    chosen[tid] = min(feasible_t, key=lambda t: param_by_task_tier[(tid, t)])
                elif selection_rule == "min_energy":
                    # Tie-break toward the cheaper tier explicitly (see
                    # policy_optimizer.select_from_rows for why this must
                    # not be left to implicit iteration order).
                    chosen[tid] = min(feasible_t, key=lambda t: (pred[(tid, t)]["energy"][cfg["energy_kind"]],
                                                                   param_by_task_tier[(tid, t)]))
                else:
                    raise ValueError(f"unknown selection_rule: {selection_rule!r}")

        qdiffs = np.array([measured_stats[(tid, chosen[tid])]["quality"]
                            - measured_stats[(tid, ref_tiers[tid])]["quality"] for tid in task_ids])
        total_energy = sum(measured_stats[(tid, chosen[tid])]["energy"] for tid in task_ids)

        if cfg["is_fallback"]:
            lower_ci = 0.0
        else:
            boot_means = _bootstrap_mean(qdiffs, n=INNER_BOOTSTRAP_N, rng=rng)
            lower_ci = float(np.percentile(boot_means, 2.5))

        if lower_ci >= QUALITY_BUDGET_PP / 100.0:
            feasible_configs.append({**cfg, "chosen": chosen, "total_energy": total_energy,
                                      "mean_qdiff": float(qdiffs.mean()), "lower_ci": lower_ci})

    assert feasible_configs, "always-reference fallback should always be feasible — bug"

    def sort_key(c):
        e_rank = _ENERGY_RANK.get(c["energy_kind"], -1)
        q_rank = _QUALITY_RANK.get(c["quality_kind"], -1)
        delta = c["delta"] if c["delta"] is not None else -1.0
        return (c["total_energy"], delta, e_rank, q_rank)

    return min(feasible_configs, key=sort_key)


def run_policy_evaluation(csv_path, selection_rule="min_energy"):
    df, task_tier, runs = load_experiment_frame(csv_path)
    cat, num = feature_columns(FEATURE_GROUPS)
    lookup = _lookup_rows(df)
    measured_stats = _task_tier_stats(df)
    ref_tiers_all = _reference_tiers(df)
    rng = np.random.default_rng(RNG_SEED)

    per_repeat_decisions = {r: {} for r in range(N_OUTER_REPEATS)}
    winning_configs_log = []

    for repeat, fold_idx, train_df, test_df in outer_splits(df):
        train_tasks = train_df["task_id"].unique().tolist()
        test_tasks = test_df["task_id"].unique().tolist()

        winner = _select_config(train_df, cat, num, measured_stats, train_tasks, rng, selection_rule=selection_rule)
        winning_configs_log.append({"repeat": repeat, "fold": fold_idx,
                                     "energy_kind": winner["energy_kind"],
                                     "quality_kind": winner["quality_kind"],
                                     "delta": winner["delta"], "is_fallback": winner["is_fallback"]})

        if winner["is_fallback"]:
            chosen_test = {tid: ref_tiers_all[tid] for tid in test_tasks}
        else:
            em = EnergyModel(winner["energy_kind"], cat, num).fit(train_df, train_df["energy_j"])
            qm = QualityModel(winner["quality_kind"], cat, num).fit(
                train_df, train_df["correct"], groups=train_df["task_id"])
            chosen_test = {}
            for tid in test_tasks:
                g = test_df[test_df["task_id"] == tid]
                rows_by_tier = {t: g[g["model_tier"] == t].iloc[[0]] for t in g["model_tier"].unique()}
                ref_tier = g["reference_tier"].iloc[0]
                chosen_tier, diag = select_from_rows(rows_by_tier, ref_tier, winner["delta"], em, qm,
                                                      selection_rule=selection_rule)
                assert ref_tier in diag["feasible_tiers"], "predicted constraint violated — bug"
                chosen_test[tid] = chosen_tier

        per_repeat_decisions[repeat].update(chosen_test)

    for r in per_repeat_decisions:
        assert set(per_repeat_decisions[r].keys()) == set(df["task_id"].unique()), \
            f"repeat {r}: outer-test folds didn't cover all 40 tasks exactly once"

    return {
        "df": df, "lookup": lookup, "measured_stats": measured_stats, "ref_tiers": ref_tiers_all,
        "per_repeat_decisions": per_repeat_decisions, "winning_configs_log": pd.DataFrame(winning_configs_log),
        "runs": runs,
    }


def _score_per_run(lookup, runs, chosen_by_task):
    """Mirrors static_policies.py's score_policy_for_run, but keyed by a
    per-task chosen-tier dict instead of a task_tier -> model_tier route_fn."""
    per_run = []
    for run in runs:
        n = correct_n = 0
        energy_total = 0.0
        for task_id, tier in chosen_by_task.items():
            correct, energy = lookup[(run, task_id, tier)]
            n += 1
            correct_n += int(correct)
            energy_total += energy
        per_run.append({"quality_pct": 100 * correct_n / n, "total_energy_j": energy_total})
    q = [r["quality_pct"] for r in per_run]
    e = [r["total_energy_j"] for r in per_run]
    return (float(np.mean(q)), float(np.std(q, ddof=1)) if len(q) > 1 else 0.0,
            float(np.mean(e)), float(np.std(e, ddof=1)) if len(e) > 1 else 0.0)


def summarize(result, experiment_name):
    df, lookup, measured_stats, ref_tiers, runs = (
        result["df"], result["lookup"], result["measured_stats"], result["ref_tiers"], result["runs"])
    task_ids = df["task_id"].unique().tolist()

    print(f"\n=== {experiment_name}: learned policy vs. baselines ===")
    ref_chosen = {tid: ref_tiers[tid] for tid in task_ids}
    ref_q_mean, ref_q_std, ref_e_mean, ref_e_std = _score_per_run(lookup, runs, ref_chosen)
    print(f"  static-cost-router (reference): quality {ref_q_mean:.1f}+/-{ref_q_std:.1f}%, "
          f"energy {ref_e_mean:.1f}+/-{ref_e_std:.1f} J")

    strict_chosen, strict_energy_stat, strict_quality_stat = strict_oracle(df)
    budget_chosen, budget_energy_stat, budget_quality_stat = global_budget_oracle(df, QUALITY_BUDGET_PP)
    # strict_energy_stat/budget_energy_stat are built from per-(task,tier)
    # mean-over-runs energy (oracle._task_tier_stats), i.e. already on the
    # same "one run's worth, summed over 40 tasks" scale as ref_e_mean —
    # do NOT multiply ref_e_mean by len(runs) here, that would compare
    # against a 4x-inflated denominator.
    print(f"  strict oracle (measured, zero-loss per task):    {strict_energy_stat:.1f} J total "
          f"({100*(1-strict_energy_stat/ref_e_mean):.1f}% saved), "
          f"mean quality {strict_quality_stat:.3f}")
    print(f"  global-budget oracle (measured, aggregate {QUALITY_BUDGET_PP}pp): "
          f"{budget_energy_stat:.1f} J total "
          f"({100*(1-budget_energy_stat/ref_e_mean):.1f}% saved), "
          f"mean quality {budget_quality_stat:.3f}")

    print(f"  winning config frequency across {len(result['winning_configs_log'])} outer folds:")
    wc = result["winning_configs_log"]
    print(f"    fallback (always-reference): {wc['is_fallback'].mean():.0%}")
    non_fb = wc[~wc["is_fallback"]]
    if len(non_fb):
        print(f"    energy kind: {non_fb['energy_kind'].value_counts(normalize=True).to_dict()}")
        print(f"    quality kind: {non_fb['quality_kind'].value_counts(normalize=True).to_dict()}")
        print(f"    delta: mean={non_fb['delta'].mean():.3f}, median={non_fb['delta'].median():.3f}")

    pooled_records = []
    per_repeat_rows = []
    for repeat, chosen_by_task in result["per_repeat_decisions"].items():
        q_mean, q_std, e_mean, e_std = _score_per_run(lookup, runs, chosen_by_task)
        mean_qdiff = np.mean([measured_stats[(tid, chosen_by_task[tid])]["quality"]
                               - measured_stats[(tid, ref_tiers[tid])]["quality"] for tid in task_ids])
        satisfies = mean_qdiff >= QUALITY_BUDGET_PP / 100.0
        policy_total_energy = sum(measured_stats[(tid, chosen_by_task[tid])]["energy"] for tid in task_ids)
        ref_total_energy = sum(measured_stats[(tid, ref_tiers[tid])]["energy"] for tid in task_ids)
        savings_recovered = (
            (ref_total_energy - policy_total_energy) / (ref_total_energy - budget_energy_stat)
            if satisfies and (ref_total_energy - budget_energy_stat) > 0 else None
        )
        per_repeat_rows.append({
            "repeat": repeat, "quality_pct_mean": q_mean, "quality_pct_std": q_std,
            "energy_j_mean": e_mean, "energy_j_std": e_std,
            "quality_diff_pp_vs_static": mean_qdiff * 100,
            "pct_energy_saved_vs_static": 100 * (1 - policy_total_energy / ref_total_energy),
            "satisfies_aggregate_constraint": satisfies,
            "savings_recovered_vs_global_budget_oracle": savings_recovered,
        })
        for tid in task_ids:
            pooled_records.append({
                "repeat": repeat, "task_id": tid,
                "quality_diff": measured_stats[(tid, chosen_by_task[tid])]["quality"]
                                 - measured_stats[(tid, ref_tiers[tid])]["quality"],
                "energy_chosen": measured_stats[(tid, chosen_by_task[tid])]["energy"],
                "energy_ref": measured_stats[(tid, ref_tiers[tid])]["energy"],
            })

    per_repeat_df = pd.DataFrame(per_repeat_rows)
    print("\n  per-repeat results (distribution across 5 repeated outer partitions):")
    print(per_repeat_df.to_string(index=False))
    print(f"\n  across repeats: quality-diff mean={per_repeat_df['quality_diff_pp_vs_static'].mean():.2f}pp "
          f"std={per_repeat_df['quality_diff_pp_vs_static'].std():.2f}pp, "
          f"%energy-saved mean={per_repeat_df['pct_energy_saved_vs_static'].mean():.1f}% "
          f"std={per_repeat_df['pct_energy_saved_vs_static'].std():.1f}%")

    pooled = pd.DataFrame(pooled_records)
    rng = np.random.default_rng(RNG_SEED + 1)
    n = len(pooled)
    idx_matrix = rng.integers(0, n, size=(FINAL_BOOTSTRAP_N, n))
    qdiff_arr = pooled["quality_diff"].values
    echosen_arr = pooled["energy_chosen"].values
    eref_arr = pooled["energy_ref"].values
    boot_qdiff_mean = qdiff_arr[idx_matrix].mean(axis=1)
    boot_pct_saved = 100 * (1 - echosen_arr[idx_matrix].sum(axis=1) / eref_arr[idx_matrix].sum(axis=1))

    qdiff_ci = np.percentile(boot_qdiff_mean, [2.5, 97.5])
    pct_saved_ci = np.percentile(boot_pct_saved, [2.5, 97.5])
    frac_violating_bound = float(np.mean(boot_qdiff_mean < QUALITY_BUDGET_PP / 100.0))

    task_level_violation = pooled["quality_diff"] < (QUALITY_BUDGET_PP / 100.0)
    violation_rate = float(task_level_violation.mean())
    violation_magnitude = float(
        (QUALITY_BUDGET_PP / 100.0 - pooled.loc[task_level_violation, "quality_diff"]).mean()
    ) if violation_rate > 0 else 0.0

    print(f"\n  pooled bootstrap (n={n} repeat x task records, {FINAL_BOOTSTRAP_N} resamples):")
    print(f"    quality diff vs. static: 95% CI [{qdiff_ci[0]*100:.2f}, {qdiff_ci[1]*100:.2f}] pp")
    print(f"    % energy saved vs. static: 95% CI [{pct_saved_ci[0]:.1f}, {pct_saved_ci[1]:.1f}]")
    print(f"    fraction of bootstrap resamples violating -2pp bound: {frac_violating_bound:.1%}")
    print(f"    task-level measured non-inferiority violation rate: {violation_rate:.1%} "
          f"(mean shortfall among violators: {violation_magnitude*100:.2f}pp)")
    print(f"    predicted-constraint violation rate: 0/{sum(len(v) for v in result['per_repeat_decisions'].values())} "
          f"(guaranteed by construction — see policy_optimizer.select_from_rows assertion)")

    return per_repeat_df, pooled


def decisions_frame(result):
    """One row per (repeat, task_id): reference tier, chosen tier, workload
    class — the raw material for routing-behavior breakdowns and plots."""
    df = result["df"]
    task_type_tier = df.drop_duplicates("task_id").set_index("task_id")["task_type_tier"].to_dict()
    ref_tiers = result["ref_tiers"]
    rows = []
    for repeat, chosen_by_task in result["per_repeat_decisions"].items():
        for task_id, chosen_tier in chosen_by_task.items():
            rows.append({
                "repeat": repeat, "task_id": task_id,
                "task_type_tier": task_type_tier[task_id],
                "reference_tier": ref_tiers[task_id],
                "chosen_tier": chosen_tier,
                "downgraded": chosen_tier != ref_tiers[task_id],
            })
    return pd.DataFrame(rows)


def routing_behavior(decisions_df, experiment_name):
    print(f"\n  routing behavior ({experiment_name}, pooled across {decisions_df['repeat'].nunique()} repeats, "
          f"{len(decisions_df)} repeat x task decisions):")
    print(f"    overall: {decisions_df['downgraded'].mean():.1%} of decisions downgrade below the reference tier")
    print(f"    chosen-tier distribution: {decisions_df['chosen_tier'].value_counts(normalize=True).to_dict()}")
    print(f"    reference-tier distribution: {decisions_df['reference_tier'].value_counts(normalize=True).to_dict()}")
    print("    by workload class (task_type_tier):")
    by_class = decisions_df.groupby("task_type_tier").agg(
        n=("task_id", "count"),
        downgrade_rate=("downgraded", "mean"),
    ).reset_index()
    tier_mix = (decisions_df.groupby("task_type_tier")["chosen_tier"]
                .value_counts(normalize=True).unstack(fill_value=0).round(2).reset_index())
    by_class = by_class.merge(tier_mix, on="task_type_tier", how="left")
    print(by_class.to_string(index=False))
    return by_class


def _score_fixed_tier(measured_stats, task_ids, tier):
    quality = np.mean([measured_stats[(tid, tier)]["quality"] for tid in task_ids]) * 100
    energy = sum(measured_stats[(tid, tier)]["energy"] for tid in task_ids)
    return quality, energy


def baseline_comparison_table(csv_path, result, per_repeat_df):
    """The table the README/log ask for: every baseline policy side by
    side, so the gain from *learned, task-conditioned* decisions is visible
    against both naive fixed-tier routing and the existing hand-designed
    policies, not just against static-cost-router alone."""
    by_run_task_model, task_tier_map, runs = results_loader.load_raw_results(csv_path)
    calibrated_route = calibrate_scarce_route(by_run_task_model, task_tier_map, runs)
    policies = {
        "static-cost-router": lambda t: results_loader.STATIC_ROUTE[t],
        "scarce": lambda t: results_loader.ENERGY_AWARE_ROUTE["scarce"][t],
        "scarce, calibrated": lambda t: calibrated_route[t],
    }

    rows = []
    for name, route_fn in policies.items():
        per_run = [score_policy_for_run(by_run_task_model, task_tier_map, r, route_fn) for r in runs]
        q_mean, q_std = mean_std([r["quality_pct"] for r in per_run])
        e_mean, e_std = mean_std([r["total_energy_j"] for r in per_run])
        rows.append({"policy": name, "quality_pct": q_mean, "quality_pct_std": q_std,
                      "energy_j": e_mean, "energy_j_std": e_std})

    df = result["df"]
    measured_stats = result["measured_stats"]
    task_ids = df["task_id"].unique().tolist()
    for tier in ("small", "medium", "large"):
        q, e = _score_fixed_tier(measured_stats, task_ids, tier)
        rows.append({"policy": f"always-{tier}", "quality_pct": q, "quality_pct_std": None,
                      "energy_j": e, "energy_j_std": None})

    rows.append({
        "policy": "learned cost-model policy", "quality_pct": per_repeat_df["quality_pct_mean"].mean(),
        "quality_pct_std": per_repeat_df["quality_pct_mean"].std(),
        "energy_j": per_repeat_df["energy_j_mean"].mean(), "energy_j_std": per_repeat_df["energy_j_mean"].std(),
    })

    strict_chosen, strict_energy, strict_quality = strict_oracle(df)
    budget_chosen, budget_energy, budget_quality = global_budget_oracle(df, QUALITY_BUDGET_PP)
    rows.append({"policy": "strict oracle", "quality_pct": strict_quality * 100, "quality_pct_std": None,
                  "energy_j": strict_energy, "energy_j_std": None})
    rows.append({"policy": "global-budget oracle (-2pp)", "quality_pct": budget_quality * 100,
                  "quality_pct_std": None, "energy_j": budget_energy, "energy_j_std": None})

    table = pd.DataFrame(rows)
    static_row = table[table["policy"] == "static-cost-router"].iloc[0]
    table["quality_diff_pp_vs_static"] = table["quality_pct"] - static_row["quality_pct"]
    table["pct_energy_saved_vs_static"] = 100 * (1 - table["energy_j"] / static_row["energy_j"])
    print(f"\n  baseline comparison table:")
    print(table.to_string(index=False))
    return table


def main():
    for csv_name, deployment_key in CSV_TO_DEPLOYMENT.items():
        path = os.path.join(str(RESULTS_DIR), csv_name)
        result = run_policy_evaluation(path)
        per_repeat_df, pooled = summarize(result, deployment_key)

        decisions_df = decisions_frame(result)
        routing_behavior(decisions_df, deployment_key)
        decisions_out = os.path.join(str(RESULTS_DIR), f"policy_decisions_learned_{deployment_key}.csv")
        decisions_df.to_csv(decisions_out, index=False)
        print(f"  wrote {decisions_out}")

        baseline_table = baseline_comparison_table(path, result, per_repeat_df)
        baseline_out = os.path.join(str(RESULTS_DIR), f"baseline_comparison_{deployment_key}.csv")
        baseline_table.to_csv(baseline_out, index=False)
        print(f"  wrote {baseline_out}")

        out_path = os.path.join(str(RESULTS_DIR), f"policy_summary_learned_{deployment_key}.csv")
        per_repeat_df.to_csv(out_path, index=False)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
