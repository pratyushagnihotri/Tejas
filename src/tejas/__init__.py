"""Tejas: energy-aware LLM routing driven by a learned energy/quality cost
model, built as a proof of concept for computation-by-energy.

Four subpackages, matching the four building blocks described in the paper:
- tejas.measurement   -- real per-request energy measurement (Mac/DGX)
- tejas.benchmarking  -- the task suite and harness that runs models against
  tasks and records measured outcomes
- tejas.cost_model    -- E_hat(task, model)/Q_hat(task, model) prediction
  ladders, fit and validated per deployment
- tejas.routing       -- policies (hand-designed and learned) that consume
  the cost model to make a per-task routing decision
"""

__version__ = "0.1.0"
