# Tejas results

All measured and derived results of the Tejas evaluation, as CSV files. Files ending in
`_mac`, `_dgx_qwen`, or `_dgx_llama` hold one deployment each (see the main README for the hardware).
Figures are not stored here; regenerate them with the scripts in [`analysis/`](../analysis/).

| Files | Content | Written by |
| ----- | ------- | ---------- |
| `raw_results*.csv` | Measured energy, latency, and quality per run, task, and model | `tejas.benchmarking.run_experiment` |
| `model_eval_*.csv`, `oof_predictions_*.csv` | Energy and quality prediction accuracy; out-of-fold predictions | `tejas.cost_model.evaluate_models` |
| `transfer_*.csv` | Cross-deployment transfer between DGX-Qwen and DGX-Llama | `tejas.cost_model.cross_family_transfer` |
| `fewshot_calibration.csv` | Adapting the cost model to a new deployment with few measurements | `tejas.cost_model.fewshot_calibration` |
| `policy_summary*.csv` | Energy and quality per routing policy | `tejas.routing.static_policies`, `tejas.routing.evaluate_policy` |
| `baseline_comparison_*.csv`, `policy_decisions_learned_*.csv` | Learned policy vs. baselines and oracles; per-task routing decisions | `tejas.routing.evaluate_policy` |
| `budget_sweep_*.csv`, `budget_sweep_scarce_point_*.csv` | Achieved quality under energy budgets from 100% to 40% | `tejas.routing.evaluate_budget` |
| `ablation_energy_model_*.csv` | Full policy vs. smallest-sufficient-model policy | `tejas.routing.ablation_energy_model` |
