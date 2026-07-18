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


if __name__ == "__main__":
    _self_check()
