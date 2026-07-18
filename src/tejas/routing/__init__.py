"""Pillar 4 -- Routing: policies that consume the cost model to make a
per-task decision. Hand-designed baselines (static-cost-router, scarce,
scarce-calibrated) live in `static_policies.py`; the learned policy's
argmin-under-a-quality-budget optimizer is in `policy_optimizer.py`,
evaluated end-to-end in `evaluate_policy.py`. `oracle.py` computes the
achievable-savings upper bounds any policy is judged against.
"""
