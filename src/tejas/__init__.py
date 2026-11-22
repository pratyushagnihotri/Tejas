"""Tejas: energy-aware LLM routing driven by a learned energy/quality cost
model. Tejas is the model-routing component of the quality-energy optimization
shift in "Towards Energy-Native Semantic Data Systems": for each request it
picks the lowest-energy model predicted to meet the quality requirement.

Four subpackages:
- tejas.measurement   -- real per-request energy measurement (Mac/DGX)
- tejas.benchmarking  -- the task suite and harness that runs models against
  tasks and records measured outcomes
- tejas.cost_model    -- E_hat(task, model)/Q_hat(task, model) prediction
  ladders, fit and validated per deployment
- tejas.routing       -- policies (hand-designed and learned) that consume
  the cost model to make a per-task routing decision
"""

__version__ = "0.1.0"
