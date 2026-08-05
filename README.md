# Tejas — A Proof of Concept for Computation-by-Energy

<h1 align="center">
  <img src="tejas_logo.png" alt="Tejas Logo" width="600"/>
</h1>

This repository accompanies the vision paper **“Towards a Computation-by-Energy Execution Model for Energy-Native AI Data Systems.”**

Query optimizers traditionally assume that submitted workloads will execute and focus on selecting an efficient physical plan. We call this **computation-on-demand (CoD)**: the workload is fixed, and the optimizer decides how it runs. This execution model treats energy as a secondary objective. However, the growing energy demand of AI workloads makes it increasingly important for energy to influence execution decisions.

We propose a vision for a new execution model called **computation-by-energy (CoE)**. Instead of asking only how a fixed workload should run, CoE also decides whether it should run now, later, elsewhere, or in an adapted form. Workload requirements and current or predicted energy state jointly determine the execution decision and physical plan. The paper describes the interfaces, optimizer decisions, runtime controls, and outcome records required to support CoE.

We present **Tejas** as an initial instantiation of CoE for energy-conditioned model routing. Tejas determines whether explicit energy state can change the selected model before inference while preserving a workload-quality requirement. Each permitted model tier represents an alternative computation with different energy and quality consequences.

Across three inference deployments, Tejas reduces energy consumption by up to **48.6%**. Quality improves on one deployment and changes by less than half a percentage point on the other two. In contrast, applying the same fixed model downgrade to every request causes large quality losses on deployments with wider differences between model tiers.

Three deployments, real measured energy throughout:

| Experiment | Model ladder (small/medium/large) | Hardware | Energy measurement |
|---|---|---|---|
| Mac | Qwen2.5 3b / 7b / 14b | Apple M4 Pro (unified SoC) | `powermetrics` — whole-package power |
| DGX-Qwen | Qwen2.5 3b / 7b / 32b | 1× NVIDIA A100-40GB | `nvidia-smi` — GPU die only |
| DGX-Llama | Llama3 3b / 8b / 70b | 2× NVIDIA A100-40GB | `nvidia-smi` — GPU die only |

## The four pillars

| Pillar | Package | What it does |
|---|---|---|
| 1. Measurement | [`src/tejas/measurement/`](src/tejas/measurement/) | Real energy measurement: `powermetrics` (macOS) / `nvidia-smi` (DGX), trapezoidal integration to joules per request |
| 2. Benchmarking | [`src/tejas/benchmarking/`](src/tejas/benchmarking/) | 40-task harness (24 DB/SQL tasks + 16 [SemBench](https://github.com/SemBench/SemBench)-derived semantic-reasoning tasks), grading, and the experiment runner — the offline calibration workload and measurement store |
| 3. Cost modeling | [`src/tejas/cost_model/`](src/tejas/cost_model/) | Learned `E_hat(task, model, deployment)` / `Q_hat(task, model, deployment)` ladders (the decision models / model builder), validated with repeated nested grouped cross-validation |
| 4. Routing | [`src/tejas/routing/`](src/tejas/routing/) | Contract interface, energy-state manager, policy optimizer (quality-constrained and energy-budgeted), three measured oracles, and bootstrap-CI'd evaluation of both policies |

## Install

**Requirements**: Python ≥3.9, `git`. GPU deployments additionally need
[Ollama](https://ollama.com) and (for the DGX experiments) a Slurm
cluster with `nvidia-smi`; the Mac deployment needs `sudo` access for
`powermetrics`. None of this is required just to inspect the code,
reproduce the published numbers from the committed CSVs, or regenerate
figures — only to collect new energy measurements yourself.

```bash
git clone https://github.com/pratyushagnihotri/Tejas.git
cd Tejas
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs the `tejas` package in editable mode from `src/tejas/`
plus its dependencies (numpy, scipy, pandas, scikit-learn, matplotlib —
pinned versions in `pyproject.toml`). Editable install means changes to
the source under `src/tejas/` take effect immediately, no reinstall
needed. Console-script shortcuts are also installed alongside the
`python -m tejas.<module>` form used throughout this README:
`tejas-run-experiment`, `tejas-evaluate-models`, `tejas-evaluate-policy`,
`tejas-cross-family-transfer`, `tejas-fewshot-calibration`,
`tejas-static-policies`.

**Verify the install** — every module has a self-check, so this both
confirms the environment is set up correctly and reproduces a published
number from a clean checkout, with no measurements or GPU access needed:

```bash
.venv/bin/python3 -m tejas.cost_model.task_features   # feature-extraction self-check
.venv/bin/python3 -m tejas.routing.oracle              # reproduces the Mac oracle numbers from committed CSVs
```

**If you only want to re-collect energy measurements** (not required —
`results/` already has everything the paper reports):

- On macOS, `run_experiment.py` needs `sudo` to read `powermetrics`; use
  `--no-energy` to smoke-test the harness without root.
- On the DGX deployments, submit via Slurm — see
  [`slurm/run_experiment.slurm`](slurm/run_experiment.slurm) for the
  `sbatch` invocation and required `--gres`/`--export` flags per model
  family.
- Either way, [Ollama](https://ollama.com) must be running locally with
  the relevant models already pulled (`qwen2.5:3b-instruct`, `7b`, `14b`
  / `32b`, or `llama3.2:3b`, `llama3.1:8b`, `llama3.3:70b`, depending on
  which family you're running).

## Quickstart per pillar

All results already exist in `results/` (see below) — none of this needs
to be re-run to reproduce the paper's numbers from the published CSVs.
These are the commands that produced them.

**1. Run the benchmark harness** (needs [Ollama](https://ollama.com) running
locally with the relevant models pulled):

```bash
.venv/bin/python3 -m tejas.benchmarking.run_experiment --no-energy --limit 3   # smoke test, no sudo
sudo .venv/bin/python3 -m tejas.benchmarking.run_experiment --repeats 4         # full run, real energy (macOS)
```

DGX runs use `--family qwen-dgx` / `--family llama` under Slurm — see
[`slurm/run_experiment.slurm`](slurm/run_experiment.slurm).

**2. Simulate the fixed baselines** (`static-cost-router`, `Tejas-Fixed`,
calibrated threshold — table lookup over already-collected results, no
re-run needed):

```bash
.venv/bin/python3 -m tejas.routing.static_policies
```

**3. Fit and validate the cost model:**

```bash
.venv/bin/python3 -m tejas.cost_model.evaluate_models       # prediction-quality metrics, 5x5 nested CV
.venv/bin/python3 -m tejas.cost_model.verification_checks   # leakage + ablation spot-checks
.venv/bin/python3 -m tejas.cost_model.cross_family_transfer # DGX-Qwen <-> DGX-Llama transfer
.venv/bin/python3 -m tejas.cost_model.fewshot_calibration   # target-deployment adaptation curves
```

**4. Evaluate quality-constrained routing** (`Tejas-Learned`) against the
baselines and the strict / quality-budget oracles:

```bash
.venv/bin/python3 -m tejas.routing.evaluate_policy
```

**5. Evaluate energy-budgeted routing** across a sweep of budgets (100%
down to 40% of static), against `Tejas-Fixed` and the energy-budget
oracle — this is Figure 5 in the paper:

```bash
.venv/bin/python3 -m tejas.routing.evaluate_budget
```

**6. Ablation**: does the energy model contribute anything beyond the
quality gate, or does "pick the smallest predicted-safe tier" already do
just as well?

```bash
.venv/bin/python3 -m tejas.routing.ablation_energy_model
```

Every module above also has a `python -m tejas.<pillar>.<module>` self-check
(most were built with one from the start) — run any of them directly to
verify your environment reproduces the same numbers.

## Results and figures

The `results/` directory contains all raw and processed CSV files. These include measurements for each run, task, and model; model evaluation results; policy decisions; baseline comparisons; transfer and calibration results.

```bash
.venv/bin/python3 analysis/plot_results.py            # per-experiment figures
.venv/bin/python3 analysis/plot_cross_experiment.py   # Mac vs. DGX-Qwen vs. DGX-Llama
.venv/bin/python3 analysis/plot_ieee_routing.py        # IEEE-style paper figures, routing
.venv/bin/python3 analysis/plot_ieee_cost_model.py     # IEEE-style paper figures, cost model
.venv/bin/python3 analysis/plot_ieee_evaluation.py     # prediction accuracy, transfer, calibration figures
.venv/bin/python3 analysis/plot_ieee_budget.py         # Figure 5: quality achieved across energy budgets
```