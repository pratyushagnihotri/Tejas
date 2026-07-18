"""m*(q) = argmin_{m: param(m) <= param(reference)} E_hat(q,m)
          s.t. Q_hat(q,m) >= Q_hat(q,reference) - delta

All-tier search restricted to "no more expensive than the abundant/static
reference" (matches the paper's scarce-state framing: never route *up*),
but considers every cheaper tier, not just one fixed downgrade target.

`reference_tier` is always in the feasible set by construction: it trivially
satisfies its own constraint (Q_hat(ref) >= Q_hat(ref) - delta for any
delta >= 0). This is a claim about *predicted* quality only — prediction
error can still make the measured outcome worse than the reference on an
unseen task. That's not a bug to prevent, it's exactly what
evaluate_policy.py's violation-rate metric exists to measure.

Margin-based abstention (optional, off by default — Final-extension tier,
not part of the primary configuration): fall back to reference_tier unless
the best non-reference candidate's predicted margin above the constraint
boundary is >= tau.
"""

from tejas.cost_model.deployment_features import param_count_b


def select_from_rows(rows_by_tier, reference_tier, delta, energy_model, quality_model, tau=None):
    """rows_by_tier: {model_tier: single-row DataFrame} for one task, one row
    per candidate tier (same task features, model_tier/param_count_b varying).
    Returns (chosen_tier, diagnostics dict)."""
    ref_param = param_count_b(rows_by_tier[reference_tier]["model_name"].iloc[0])
    q_ref = quality_model.predict_proba(rows_by_tier[reference_tier])[0]

    feasible_q = {}
    for tier, row in rows_by_tier.items():
        p = param_count_b(row["model_name"].iloc[0])
        if p > ref_param:
            continue
        q = quality_model.predict_proba(row)[0]
        if q >= q_ref - delta:
            feasible_q[tier] = q
    assert reference_tier in feasible_q, "reference_tier must always be feasible by construction"

    energies = {tier: float(energy_model.predict(rows_by_tier[tier])[0]) for tier in feasible_q}
    params = {tier: param_count_b(rows_by_tier[tier]["model_name"].iloc[0]) for tier in feasible_q}
    # Tie-break ties (e.g. a constant/global-mean energy model predicts the
    # same value for every tier) toward the cheaper tier explicitly, rather
    # than leaving it to dict/argument insertion order — an implicit
    # tie-break would silently reward a constant predictor whenever the
    # smallest feasible tier happens to be iterated first.
    best_tier = min(energies, key=lambda t: (energies[t], params[t]))

    abstained = False
    if tau is not None and best_tier != reference_tier:
        margin = feasible_q[best_tier] - (q_ref - delta)
        if margin < tau:
            best_tier = reference_tier
            abstained = True

    return best_tier, {
        "q_ref": q_ref,
        "feasible_tiers": list(feasible_q.keys()),
        "predicted_energy": energies,
        "abstained": abstained,
    }


def _self_check():
    import os

    from tejas.cost_model.dataset import feature_columns, load_experiment_frame
    from tejas.cost_model.energy_model import EnergyModel
    from tejas.cost_model.quality_model import QualityModel
    from tejas.paths import RESULTS_DIR

    path = os.path.join(str(RESULTS_DIR), "raw_results.csv")
    df, task_tier, _ = load_experiment_frame(path)
    cat, num = feature_columns(("base", "sql_semantic", "interactions"))

    em = EnergyModel("ridge", cat, num).fit(df, df["energy_j"])
    qm = QualityModel("logreg", cat, num).fit(df, df["correct"])

    one_run = df[df["run"] == df["run"].iloc[0]]
    n_tasks = 0
    for task_id, g in one_run.groupby("task_id"):
        rows_by_tier = {t: g[g["model_tier"] == t] for t in g["model_tier"].unique()}
        ref_tier = g["reference_tier"].iloc[0]
        for delta in (0.0, 0.1, 0.4):
            chosen, diag = select_from_rows(rows_by_tier, ref_tier, delta, em, qm)
            assert ref_tier in diag["feasible_tiers"], "reference_tier infeasible — bug"
        n_tasks += 1
    print(f"policy_optimizer self-check OK across {n_tasks} tasks x 3 delta values, "
          f"reference always feasible")

    g = one_run[one_run["task_id"] == one_run["task_id"].iloc[0]]
    rows_by_tier = {t: g[g["model_tier"] == t] for t in g["model_tier"].unique()}
    ref_tier = g["reference_tier"].iloc[0]
    chosen, diag = select_from_rows(rows_by_tier, ref_tier, 0.0, em, qm)
    print(f"example task: reference={ref_tier}, chosen(delta=0)={chosen}, "
          f"predicted_energy={diag['predicted_energy']}")

    # abstention: tau so large nothing clears it -> always falls back to reference
    chosen, diag = select_from_rows(rows_by_tier, ref_tier, 0.4, em, qm, tau=1.0)
    assert chosen == ref_tier and (diag["abstained"] or len(diag["feasible_tiers"]) == 1)
    print(f"abstention (tau=1.0) falls back to reference: chosen={chosen}")


if __name__ == "__main__":
    _self_check()
