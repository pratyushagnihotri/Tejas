"""Two oracles, computed directly from *measured* outcomes (no model
fitting, no CV — these are properties of the dataset's achievable
frontier, not predictions). Both restrict candidates per task to tiers no
more expensive than that task's reference (static) tier, matching
policy_optimizer.py's "never route up under scarcity" restriction.

strict_oracle: per task independently, minimum energy among tiers whose
measured quality is >= the reference tier's — the conservative,
zero-quality-loss bound.

global_budget_oracle: one tier per task, minimizing total energy subject
to the *aggregate* constraint the learned policy is actually held to
(mean quality delta across all evaluated tasks >= -2pp). Solved by exact
DP: each task's measured quality (mean over 4 runs) is a multiple of 0.25,
so quality deltas are too — the DP state is the running quality-delta sum
in units of 0.25, which stays small (bounded roughly by 4x task count)
rather than blowing up combinatorially.

budget_constrained_select / energy_budget_frontier: the dual of
global_budget_oracle — maximize total quality subject to a total *energy*
budget, for a whole grid of budgets at once. This is the multiple-choice
knapsack the paper's budget-vs-quality figure needs (Sections 4/5): "what
does the router do at 100/90/.../40% of the static router's energy?".
Energy is continuous (unlike quality's natural 0.25-multiple discretization),
so the DP here discretizes the energy axis into a fine grid instead;
`n_bins` controls the resulting quantization error, which is negligible
relative to typical per-task energy deltas (tens of joules) at the default.
"""

import math


def _task_tier_stats(df):
    """{(task_id, model_tier): {"energy": mean_energy_j, "quality": mean_correct}}"""
    stats = {}
    for (task_id, model_tier), g in df.groupby(["task_id", "model_tier"]):
        stats[(task_id, model_tier)] = {
            "energy": float(g["energy_j"].mean()),
            "quality": float(g["correct"].mean()),
        }
    return stats


def _candidates_for_task(df, task_id):
    """Model tiers with param_count_b <= the reference tier's, for this task."""
    task_rows = df[df["task_id"] == task_id]
    ref_tier = task_rows["reference_tier"].iloc[0]
    ref_param = task_rows.loc[task_rows["model_tier"] == ref_tier, "param_count_b"].iloc[0]
    tiers = task_rows["model_tier"].unique().tolist()
    out = []
    for t in tiers:
        p = task_rows.loc[task_rows["model_tier"] == t, "param_count_b"].iloc[0]
        if p <= ref_param:
            out.append(t)
    return ref_tier, out


def strict_oracle(df):
    """Returns {task_id: chosen_tier}, plus total_energy_j / mean_quality."""
    stats = _task_tier_stats(df)
    task_ids = df["task_id"].unique().tolist()
    chosen = {}
    for task_id in task_ids:
        ref_tier, candidates = _candidates_for_task(df, task_id)
        ref_quality = stats[(task_id, ref_tier)]["quality"]
        feasible = [t for t in candidates if stats[(task_id, t)]["quality"] >= ref_quality]
        chosen[task_id] = min(feasible, key=lambda t: stats[(task_id, t)]["energy"])
    total_energy = sum(stats[(tid, t)]["energy"] for tid, t in chosen.items())
    mean_quality = sum(stats[(tid, t)]["quality"] for tid, t in chosen.items()) / len(chosen)
    return chosen, total_energy, mean_quality


def global_budget_oracle(df, quality_budget_pp=-2.0):
    """Exact DP: minimize total energy s.t. mean quality delta vs. reference
    across all tasks >= quality_budget_pp/100."""
    stats = _task_tier_stats(df)
    task_ids = df["task_id"].unique().tolist()

    per_task_options = []  # [(task_id, [(tier, qdelta_units, energy), ...]), ...]
    for task_id in task_ids:
        ref_tier, candidates = _candidates_for_task(df, task_id)
        ref_quality = stats[(task_id, ref_tier)]["quality"]
        options = []
        for t in candidates:
            qdelta = stats[(task_id, t)]["quality"] - ref_quality
            qdelta_units = round(qdelta * 4)  # quality is a multiple of 0.25 (4 runs)
            options.append((t, qdelta_units, stats[(task_id, t)]["energy"]))
        per_task_options.append((task_id, options))

    n = len(task_ids)
    min_units = math.ceil(quality_budget_pp / 100.0 * n * 4)

    # DP: state = cumulative qdelta_units -> (min_energy, choice_for_this_task)
    # steps[i] records, for each reachable state *after* task i, which tier
    # was chosen and what the state was *before* task i (for backtracking).
    state = {0: 0.0}
    steps = []
    for task_id, options in per_task_options:
        new_state = {}
        step = {}
        for prev_units, prev_energy in state.items():
            for tier, qdelta_units, energy in options:
                u = prev_units + qdelta_units
                e = prev_energy + energy
                if u not in new_state or e < new_state[u]:
                    new_state[u] = e
                    step[u] = (prev_units, tier)
        state = new_state
        steps.append(step)

    feasible_final = {u: e for u, e in state.items() if u >= min_units}
    if not feasible_final:
        raise RuntimeError(
            f"no feasible allocation meets the quality budget ({quality_budget_pp}pp) — "
            "even routing every task to its reference tier should always be feasible "
            "(qdelta_units=0 for reference), this indicates a bug"
        )
    best_units = min(feasible_final, key=lambda u: feasible_final[u])
    total_energy = feasible_final[best_units]

    # backtrack
    chosen = {}
    u = best_units
    for i in range(n - 1, -1, -1):
        task_id = per_task_options[i][0]
        prev_u, tier = steps[i][u]
        chosen[task_id] = tier
        u = prev_u

    mean_qdelta = (best_units / 4.0) / n
    mean_quality = sum(stats[(tid, t)]["quality"] for tid, t in chosen.items()) / n
    assert mean_qdelta >= quality_budget_pp / 100.0 - 1e-9, (mean_qdelta, quality_budget_pp)
    return chosen, total_energy, mean_quality


def budget_constrained_select(per_task_options, budget_energy, n_bins=5000):
    """Generic multiple-choice-knapsack DP: per_task_options is
    [(task_id, [(tier, energy, quality), ...]), ...] (one option list per
    task, e.g. from measured stats for the oracle, or from a fitted model's
    predictions for a learned policy). Chooses one tier per task to maximize
    total quality subject to sum(energy) <= budget_energy. Returns
    {task_id: chosen_tier}. Callers restrict `energy`/`quality` to whatever
    they want maximized against — this function has no notion of what a
    "reference" tier is; that restriction (never route up) must already be
    baked into each task's option list before calling this.

    energy is discretized into fine bins spanning a range generously larger
    than budget_energy itself (rounding each option's energy up to a bin
    boundary would only make the DP conservative about the very budget it's
    trying to hit exactly, so we round to the *nearest* bin instead — the
    resulting quantization error is bounded by roughly budget_energy /
    (2*n_bins) per task, negligible next to real per-task energy deltas at
    the default n_bins). The extra headroom beyond budget_energy matters
    when `energy` values are *predicted*, not measured: a fold's cost model
    can be biased enough on its own held-out tasks that even the cheapest
    per-task combination it can see exceeds a tight budget under its own
    predictions, even though the true (measured) energy would have fit —
    tracking those over-budget states rather than discarding them lets the
    function fall back to "the cheapest combination this fold's model could
    find" instead of raising. That fallback is itself something worth
    reporting (it shows up downstream as achieved energy exceeding the
    nominal budget), not a bug to hide. This should not happen when `energy`
    is measured (the oracle case), since the reference assignment alone is
    always feasible at the smallest budget candidates are restricted to.
    """
    if budget_energy <= 0:
        raise ValueError(f"budget_energy must be positive, got {budget_energy}")
    max_bin_multiplier = 3
    max_energy = budget_energy * max_bin_multiplier
    n_total_bins = n_bins * max_bin_multiplier
    bin_width = max_energy / n_total_bins
    budget_bin = round(budget_energy / bin_width)

    def to_bin(e):
        return max(0, min(n_total_bins, round(e / bin_width)))

    state = {0: 0.0}
    steps = []
    for task_id, options in per_task_options:
        new_state = {}
        step = {}
        for prev_bin, prev_quality in state.items():
            for tier, energy, quality in options:
                b = prev_bin + to_bin(energy)
                if b > n_total_bins:
                    continue
                q = prev_quality + quality
                if b not in new_state or q > new_state[b]:
                    new_state[b] = q
                    step[b] = (prev_bin, tier)
        state = new_state
        steps.append(step)
        if not state:
            raise RuntimeError(
                "no allocation at all fits within 3x the budget — every task's "
                "cheapest candidate together must exceed even that, which "
                "shouldn't happen with realistic energy values; this indicates "
                "a bug in the caller's option lists, not a tight budget"
            )

    feasible = {b: q for b, q in state.items() if b <= budget_bin}
    if feasible:
        best_bin = max(feasible, key=lambda b: feasible[b])
    else:
        # Budget-infeasible even under the cheapest available combination
        # (only possible with predicted, not measured, energies) — fall
        # back to that cheapest combination rather than raising.
        best_bin = min(state)

    chosen = {}
    b = best_bin
    for i in range(len(per_task_options) - 1, -1, -1):
        task_id = per_task_options[i][0]
        prev_b, tier = steps[i][b]
        chosen[task_id] = tier
        b = prev_b
    return chosen


def energy_budget_frontier(df, budget_fracs, n_bins=5000):
    """The measured (not predicted) upper-bound frontier: for each fraction
    in budget_fracs (of the reference/static-router's total measured
    energy), the maximum mean quality achievable across all 40 tasks
    subject to that energy budget. Returns
    {frac: (chosen, achieved_energy, achieved_quality)} plus the reference
    total energy the fractions are relative to. frac=1.0 is always
    trivially feasible (route every task to its own reference tier)."""
    stats = _task_tier_stats(df)
    task_ids = df["task_id"].unique().tolist()
    ref_tiers = df.drop_duplicates("task_id").set_index("task_id")["reference_tier"].to_dict()
    ref_total_energy = sum(stats[(tid, ref_tiers[tid])]["energy"] for tid in task_ids)

    per_task_options = []
    for task_id in task_ids:
        ref_tier, candidates = _candidates_for_task(df, task_id)
        options = [(t, stats[(task_id, t)]["energy"], stats[(task_id, t)]["quality"]) for t in candidates]
        per_task_options.append((task_id, options))

    results = {}
    for frac in budget_fracs:
        budget = frac * ref_total_energy
        chosen = budget_constrained_select(per_task_options, budget, n_bins=n_bins)
        achieved_energy = sum(stats[(tid, chosen[tid])]["energy"] for tid in task_ids)
        achieved_quality = sum(stats[(tid, chosen[tid])]["quality"] for tid in task_ids) / len(task_ids)
        results[frac] = (chosen, achieved_energy, achieved_quality)
    return results, ref_total_energy


def _self_check():
    import os

    from tejas.cost_model.dataset import load_experiment_frame
    from tejas.paths import RESULTS_DIR

    path = os.path.join(str(RESULTS_DIR), "raw_results.csv")
    df, _, _ = load_experiment_frame(path)

    strict_chosen, strict_energy, strict_quality = strict_oracle(df)
    budget_chosen, budget_energy, budget_quality = global_budget_oracle(df, quality_budget_pp=-2.0)

    assert len(strict_chosen) == 40 and len(budget_chosen) == 40
    # global-budget oracle should never need *more* energy than the strict
    # oracle (it has a strictly larger feasible set per task)
    assert budget_energy <= strict_energy + 1e-6, (budget_energy, strict_energy)

    ref_energy = sum(
        df[(df["task_id"] == tid) & (df["model_tier"] == df.loc[df["task_id"] == tid, "reference_tier"].iloc[0])]
        ["energy_j"].mean()
        for tid in df["task_id"].unique()
    )
    print(f"reference (static) total energy: {ref_energy:.1f} J")
    print(f"strict oracle:        {strict_energy:.1f} J, mean quality {strict_quality:.3f} "
          f"({100*(ref_energy-strict_energy)/ref_energy:.1f}% saved)")
    print(f"global-budget oracle: {budget_energy:.1f} J, mean quality {budget_quality:.3f} "
          f"({100*(ref_energy-budget_energy)/ref_energy:.1f}% saved)")

    frontier, ref_total = energy_budget_frontier(df, [1.0, 0.8, 0.6, 0.4])
    assert abs(ref_total - ref_energy) < 1e-6, (ref_total, ref_energy)
    # The reference assignment (every task at its own reference tier) is
    # always a *feasible* choice at frac=1.0, so the DP's optimum there can
    # only match or beat it — never do worse. It need not *equal* the
    # reference assignment: if some task's cheaper tier ties or beats the
    # reference on measured quality, the DP correctly prefers it (this is
    # the same effect documented as the always-medium-beats-static finding
    # elsewhere), which is exactly why this is a >=/<= check, not equality.
    chosen_1, energy_1, quality_1 = frontier[1.0]
    ref_tiers = df.drop_duplicates("task_id").set_index("task_id")["reference_tier"].to_dict()
    ref_mean_quality = sum(_task_tier_stats(df)[(tid, ref_tiers[tid])]["quality"] for tid in ref_tiers) / len(ref_tiers)
    assert quality_1 >= ref_mean_quality - 1e-9, (quality_1, ref_mean_quality)
    assert energy_1 <= ref_total + 1e-6, (energy_1, ref_total)
    print("\nenergy_budget_frontier (measured, quality-maximizing per budget):")
    prev_quality = None
    for frac in (1.0, 0.8, 0.6, 0.4):
        _, e, q = frontier[frac]
        print(f"  budget={frac:.0%} of static ({frac*ref_total:.1f} J): "
              f"achieved energy {e:.1f} J, mean quality {q:.3f}")
        # tighter budgets can never do *better* than looser ones (monotone
        # feasible-set nesting) — a real bug (e.g. an off-by-one in the DP)
        # would most likely show up as a violation of this.
        if prev_quality is not None:
            assert q <= prev_quality + 1e-9, (frac, q, prev_quality)
        prev_quality = q


if __name__ == "__main__":
    _self_check()
