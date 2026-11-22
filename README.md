# Tejas — Towards Energy-Native Semantic Data Systems

<h1 align="center">
  <img src="tejas_logo.png" alt="Tejas Logo" width="600"/>
</h1>

This repository contains the code and measurements for the vision paper
*Towards Energy-Native Semantic Data Systems*. It has two parts:

* [`probes/`](probes/): the cross-engine benchmark data and the feasibility probes (see [Feasibility probes](#feasibility-probes-for-energy-native-semantic-data-systems)).
* [`src/tejas/`](src/tejas/): **Tejas**, a prototype for quality-aware, energy-aware model routing
  (see [Tejas prototype](#tejas-prototype)).

## Overview

Semantic data systems extend query processing with operators that call large language models (LLMs) to
filter, join, classify, and transform data with natural-language predicates. Today, the query engine
and the LLM serving system are optimized separately. The DBMS knows the query structure: predicates,
projections, `LIMIT`, cardinalities, and which results could be reused. The serving system knows
batching efficiency, model residency, KV-cache state, and accelerator load. Energy depends on both, so
neither layer can minimize it alone.

The paper argues that the DBMS and the LLM serving infrastructure should be co-designed around one
objective: minimizing the energy needed to *produce, maintain, and reuse useful semantic knowledge*,
subject to the workload's requirements on quality, latency, and freshness. It proposes four
architectural shifts:

1. **Semantic reuse and materialization.** Model-derived results become managed state that the system
   can reuse, transform, refresh, or recompute, including reuse through predicate relationships that a
   prompt cache cannot see.
2. **Quality–energy optimization.** The optimizer chooses the model, semantic implementation, and
   approximation under an explicit quality contract, with energy as the objective.
3. **Joint DBMS–LLM execution.** Query-plan properties (`LIMIT`, projection, predicate order) and serving
   properties (batching, model residency, inference state) are decided together.
4. **Energy-native workload orchestration.** Deferrable semantic work, such as refreshing
   materializations, runs when energy is cheaper or cleaner, or not at all when existing results are
   still fresh enough.

The paper supports these premises with a cross-engine benchmark of six existing semantic query engines
and one controlled probe per shift. On SemBench Q1, correct answers from existing engines differ by up
to 513× in energy.

## Tejas prototype

**Tejas** implements one piece of the second shift: energy-aware model routing for LLM inference.

For each request, Tejas chooses among several model sizes. Each model has different energy consumption
and output quality. Tejas predicts both from pre-inference features and selects the lowest-energy model
that is predicted to meet the required quality, before inference begins.

Across three inference deployments, Tejas reduces energy consumption by up to **48.6%**. Quality improves
on one deployment and changes by less than half a percentage point on the other two. By comparison,
using the same smaller model for every request causes large quality losses when the differences between
model sizes are significant. Probe P1 applies the same routing idea to semantic predicates on SemBench.

## Tejas experimental deployments

We evaluated Tejas on three deployments. Energy consumption was measured directly during every experiment.

| Experiment | Available models        | Hardware                         | Energy measurement                            |
| ---------- | ----------------------- | -------------------------------- | --------------------------------------------- |
| Mac        | Qwen2.5 3B, 7B, and 14B | Apple M4 Pro with unified memory | `powermetrics`, measuring whole-package power |
| DGX-Qwen   | Qwen2.5 3B, 7B, and 32B | 1× NVIDIA A100 40 GB             | `nvidia-smi`, measuring GPU power             |
| DGX-Llama  | Llama 3B, 8B, and 70B   | 2× NVIDIA A100 40 GB             | `nvidia-smi`, measuring GPU power             |

The Mac measurements cover the complete system package. The DGX measurements cover GPU power only.

## Feasibility probes for Energy-Native Semantic Data Systems

[`probes/`](probes/) contains the code and data behind the paper's evidence: a cross-engine benchmark of
existing semantic query engines and one controlled probe per architectural shift (see [Overview](#overview)).

### Probes

The controlled probes use SemBench Movie and Qwen2.5 models served by vLLM on one NVIDIA A100-40GB.
They report GPU energy from the NVML cumulative energy counter, with an idle baseline.

| Probe | Shift | Question | Code | Slurm job |
| ----- | ----- | -------- | ---- | --------- |
| P1 | Quality–energy optimization | Does the energy-optimal model that meets the quality contract differ per tuple and query? (Tejas-style router, cascade, oracle) | [`p1_tejas_sembench/`](probes/p1_tejas_sembench/) | `slurm/p1_tiers.sbatch` |
| P2 | Joint DBMS–LLM execution | What does query-plan knowledge (LIMIT, projection, predicate order, wave size) save beyond vLLM batching and prefix caching? | [`p2_plan_aware/`](probes/p2_plan_aware/) | `slurm/p2_vllm.sbatch` |
| P3 | Semantic reuse | Can the DBMS reuse semantic results through predicate relationships where an LLM cache cannot? | [`p3_semantic_reuse/`](probes/p3_semantic_reuse/) | `slurm/p3_vllm.sbatch` |
| P4 | Workload orchestration | Can deferrable semantic maintenance exploit time-varying grid carbon intensity? (trace-driven) | [`p4_temporal/`](probes/p4_temporal/) | runs on a laptop |

Shared helpers for data loading, LLM calls, and energy measurement are in [`probes/common/`](probes/common/).
[`probes/README.md`](probes/README.md) describes the strategies of each probe and how to run them on the DGX.

### Cross-engine benchmark data

[`probes/data/`](probes/data/) contains the measured results of the cross-engine benchmark:

* six semantic query engines run unmodified (LOTUS, Palimpzest, ThalamusDB, CAESURA, Stretto, and Abacus as implemented in Stretto),
* four SemBench Movie queries (Q1, Q2, Q4, Q5) over 1,000 reviews,
* ten local 4-bit models (Qwen2.5, Llama 3, Gemma 3; 1B–32B) served by Ollama on one A100-40GB per run.

Energy is GPU energy including idle power plus the job's share of CPU package energy (RAPL). An answer
counts as correct if its SemBench `score` is at least 0.8.

| File | Content |
| ---- | ------- |
| `cross_engine_local.csv` | All 264 single-model runs (240 engine × model × query configurations) |
| `cross_engine_qwen7b_q1q5.csv` | Qwen2.5-7B, Q1 and Q5, all repetitions |
| `de_2025_hourly.csv`, `raw_*.json` | Hourly German grid carbon intensity and day-ahead price for 2025 (Energy-Charts), used by P4 |

[`probes/data/SOURCE.md`](probes/data/SOURCE.md) documents the provenance of each file.

## Repository structure

The main source code is located under [`src/tejas/`](src/tejas/).

| Directory                                            | Description                                                                                                                                                                                                         |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`src/tejas/measurement/`](src/tejas/measurement/)   | Collects power measurements using `powermetrics` on macOS and `nvidia-smi` on NVIDIA systems. It integrates the power samples over time to calculate the energy used by each request.                               |
| [`src/tejas/benchmarking/`](src/tejas/benchmarking/) | Contains the benchmark tasks, grading logic, and experiment runner. The benchmark includes 24 database and SQL tasks and 16 semantic-reasoning tasks derived from [SemBench](https://github.com/SemBench/SemBench). |
| [`src/tejas/cost_model/`](src/tejas/cost_model/)     | Trains and evaluates models that predict energy consumption and output quality for a given task, model, and deployment.                                                                                             |
| [`src/tejas/routing/`](src/tejas/routing/)           | Contains the routing policies, energy-state handling, optimization logic, baselines, oracle policies, and policy-evaluation code.                                                                                   |
| [`analysis/`](analysis/)                             | Contains scripts for generating the Tejas evaluation figures.                                                                                                                                                       |
| [`results/`](results/)                               | Contains the raw measurements and processed evaluation results.                                                                                                                                                     |
| [`slurm/`](slurm/)                                   | Contains the Slurm configuration used for the DGX experiments.                                                                                                                                                      |
| [`probes/`](probes/)                                 | Feasibility probes and cross-engine benchmark data for *Towards Energy-Native Semantic Data Systems*. See [Feasibility probes](#feasibility-probes-for-energy-native-semantic-data-systems). |

## Installation

### Requirements

The basic requirements are:

* Python 3.9 or newer
* Git

Additional software is needed only when collecting new measurements:

* [Ollama](https://ollama.com) for running the language models
* `sudo` access to `powermetrics` for macOS energy measurements
* a Slurm cluster with `nvidia-smi` for the DGX experiments

These additional requirements are not needed to inspect the source code, reproduce the reported results from the existing CSV files, or regenerate the figures.

Clone and install the repository:

```bash
git clone https://github.com/pratyushagnihotri/Tejas.git
cd Tejas

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
```

This installs the `tejas` package and its dependencies in editable mode. The main dependencies include NumPy, SciPy, pandas, scikit-learn, and Matplotlib. Exact versions are listed in `pyproject.toml`.

With an editable installation, changes made under `src/tejas/` are available immediately without reinstalling the package.

The installation also provides the following command-line programs:

* `tejas-run-experiment`
* `tejas-evaluate-models`
* `tejas-evaluate-policy`
* `tejas-cross-family-transfer`
* `tejas-fewshot-calibration`
* `tejas-static-policies`

The examples below use the equivalent `python -m tejas.<module>` form.

## Run with Docker

A `Dockerfile` and `docker-compose.yml` are included to run `Tejas` in a local environment. This path only needs [Docker](https://docs.docker.com/get-docker/) installed and running.

```bash
git clone https://github.com/pratyushagnihotri/Tejas.git
cd Tejas

docker build -t tejas .
docker run --rm tejas
```

The image installs `tejas` and its dependencies, and the default command reproduces the same two checks as the "Verify the installation" section below: task-feature extraction and the Mac oracle results, computed from the CSV files already committed in `results/`.

Any other command in this README can be run the same way, by replacing `.venv/bin/python3` with `docker run --rm tejas python`, for example:

```bash
docker run --rm tejas python -m tejas.routing.static_policies
docker run --rm tejas python -m tejas.routing.evaluate_policy
docker run --rm tejas python -m tejas.routing.evaluate_budget
```

For an interactive shell inside the container:

```bash
docker run --rm -it tejas bash
```

`docker-compose.yml` mounts `src/`, `results/`, and `analysis/` from the host into the container, so code changes and generated figures are visible on disk without rebuilding the image:

```bash
docker compose run --rm tejas bash
```

Like the local install, this covers everything that only needs the CSV files already in `results/`: self-checks, baseline and routing evaluation, and figure generation. Collecting *new* energy measurements still needs `powermetrics` or `nvidia-smi` running directly on the host — that part is out of scope for the container.

## Verify the installation

The following commands check that the environment is set up correctly:

```bash
.venv/bin/python3 -m tejas.cost_model.task_features
.venv/bin/python3 -m tejas.routing.oracle
```

The first command checks the task-feature extraction code. The second command reproduces the Mac oracle results using the CSV files included in the repository.

No GPU access or new energy measurements are required for these checks.

## Reproduce the experiments

All Tejas results are already available under `results/`. The commands in this section show how those results were produced.

### 1. Run a small benchmark test

The following command runs three tasks without collecting energy measurements:

```bash
.venv/bin/python3 -m tejas.benchmarking.run_experiment \
  --no-energy \
  --limit 3
```

This is useful for checking that the benchmark and Ollama connection work correctly.

### 2. Run the complete benchmark on macOS

```bash
sudo .venv/bin/python3 -m tejas.benchmarking.run_experiment \
  --repeats 4
```

This command uses `powermetrics` to collect real energy measurements and therefore requires `sudo`.

### 3. Run the benchmark on DGX

DGX experiments use the `qwen-dgx` or `llama` model family:

```bash
--family qwen-dgx
```

or:

```bash
--family llama
```

The experiments should be submitted through Slurm. See [`slurm/run_experiment.slurm`](slurm/run_experiment.slurm) for the complete `sbatch` command and the required `--gres` and `--export` settings.

## Model requirements

Ollama must be running locally when collecting new inference measurements. The required models must also be downloaded before starting an experiment.

Depending on the deployment, the models are:

```text
qwen2.5:3b-instruct
qwen2.5:7b-instruct
qwen2.5:14b-instruct
qwen2.5:32b-instruct
llama3.2:3b
llama3.1:8b
llama3.3:70b
```

The exact models needed depend on the selected model family and deployment.

## Evaluate the baselines

The following command evaluates the fixed routing baselines using the measurements already stored in `results/`:

```bash
.venv/bin/python3 -m tejas.routing.static_policies
```

The evaluated baselines include:

* `static-cost-router`
* `Tejas-Fixed`
* the calibrated-threshold policy

This step uses existing measurements and does not run the language models again.

## Train and evaluate the prediction models

Run the following commands to train and evaluate the energy and quality prediction models:

```bash
.venv/bin/python3 -m tejas.cost_model.evaluate_models
.venv/bin/python3 -m tejas.cost_model.verification_checks
.venv/bin/python3 -m tejas.cost_model.cross_family_transfer
.venv/bin/python3 -m tejas.cost_model.fewshot_calibration
```

These commands perform the following evaluations:

| Command                 | Purpose                                                                                            |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| `evaluate_models`       | Evaluates energy and quality prediction accuracy using repeated nested grouped cross-validation.   |
| `verification_checks`   | Checks for data leakage and runs selected ablation tests.                                          |
| `cross_family_transfer` | Evaluates transfer between the DGX-Qwen and DGX-Llama deployments.                                 |
| `fewshot_calibration`   | Evaluates how a model adapts to a new deployment using a small number of calibration measurements. |

## Evaluate quality-constrained routing

The following command evaluates the learned Tejas routing policy:

```bash
.venv/bin/python3 -m tejas.routing.evaluate_policy
```

It compares `Tejas-Learned` with the fixed baselines and the measured oracle policies.

The policy selects the lowest-energy model that is predicted to satisfy the required quality constraint.

## Evaluate energy-budgeted routing

The following command evaluates routing under different energy budgets:

```bash
.venv/bin/python3 -m tejas.routing.evaluate_budget
```

The evaluation considers budgets ranging from 100% to 40% of the energy consumed by the static large-model policy.

It compares:

* `Tejas-Learned`
* `Tejas-Fixed`
* the energy-budget oracle

The results from this experiment are used to generate the energy-budget figure (`plot_ieee_budget.py`).

## Run the energy-model ablation

The following command tests whether the learned energy model improves routing decisions:

```bash
.venv/bin/python3 -m tejas.routing.ablation_energy_model
```

This experiment compares the complete Tejas policy with a simpler policy that selects the smallest model predicted to satisfy the quality requirement.

The purpose is to determine whether explicit energy prediction provides an advantage beyond the quality constraint alone.

## Collect new energy measurements

Collecting new measurements is optional because the repository already includes all Tejas results.

### macOS

The experiment runner requires `sudo` to access `powermetrics`:

```bash
sudo .venv/bin/python3 -m tejas.benchmarking.run_experiment
```

To test the benchmark without measuring energy, use:

```bash
.venv/bin/python3 -m tejas.benchmarking.run_experiment --no-energy
```

### DGX and other NVIDIA systems

Use the provided Slurm script:

[`slurm/run_experiment.slurm`](slurm/run_experiment.slurm)

The script contains the GPU resource requirements and environment settings used for the DGX experiments.

## Results

The [`results/`](results/) directory contains all raw and processed CSV files generated by the project.

These files include:

* measurements for each run, task, and model,
* energy and quality prediction results,
* routing decisions,
* baseline comparisons,
* oracle results,
* cross-deployment transfer results,
* few-shot calibration results,
* energy-budget evaluations, and
* ablation-study results.

The files:

```text
budget_sweep_<deployment>.csv
budget_sweep_scarce_point_<deployment>.csv
```

contain the results used to generate the energy-budget figure.

The files:

```text
ablation_energy_model_<deployment>.csv
```

contain the results from the energy-model ablation study.

Every Tejas result can therefore be checked using the committed CSV files.

## Generate the Tejas figures

The Tejas figures are generated from the CSV files in `results/`. They are not stored as pre-generated binary files.

Run the following commands to regenerate them:

```bash
.venv/bin/python3 analysis/plot_results.py
.venv/bin/python3 analysis/plot_cross_experiment.py
.venv/bin/python3 analysis/plot_ieee_routing.py
.venv/bin/python3 analysis/plot_ieee_cost_model.py
.venv/bin/python3 analysis/plot_ieee_evaluation.py
.venv/bin/python3 analysis/plot_ieee_budget.py
```

The scripts generate:

| Script                     | Output                                                            |
| -------------------------- | ------------------------------------------------------------------|
| `plot_results.py`          | Figures for each individual deployment                            |
| `plot_cross_experiment.py` | Comparison of Mac, DGX-Qwen, and DGX-Llama                        |
| `plot_ieee_routing.py`     | Figures for routing-policy results                                |
| `plot_ieee_cost_model.py`  | Figures for energy and quality prediction                         |
| `plot_ieee_evaluation.py`  | Figures for prediction accuracy, model transfer, and calibration  |
| `plot_ieee_budget.py`      | Achieved quality under different energy budgets                   |

## Running individual checks

Most modules can also be executed directly:

```bash
python -m tejas.<package>.<module>
```

Many modules include a small self-check that verifies their main functionality and confirms that the stored results can be reproduced.
