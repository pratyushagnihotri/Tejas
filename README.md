# Tejas — A Proof of Concept for Computation-by-Energy

<h1 align="center">
  <img src="tejas_logo.png" alt="Tejas Logo" width="600"/>
</h1>

This repository contains the source code, measurements, evaluation scripts, and results for the vision paper.

## Overview

Traditional query optimizers assume that every submitted workload must be executed. Their main task is to determine an efficient physical execution plan. We refer to this approach as **computation-on-demand (CoD)**: the workload is fixed, and the optimizer decides how to execute it.

In this model, energy is usually treated as an additional cost metric rather than something that can change the execution decision itself. However, the increasing energy demand of AI workloads makes it important to consider whether a computation should be executed immediately, delayed, moved to another system, or adapted to use fewer resources.

The paper introduces **computation-by-energy (CoE)**. In CoE, workload requirements and the current or expected energy state are considered together. The system can decide:

* whether the workload should run,
* when it should run,
* where it should run,
* which implementation or model should be used, and
* whether the workload should be adapted to reduce energy consumption.

The paper describes the interfaces, optimizer decisions, runtime controls, and execution records needed to support this model.

## Tejas prototype

**Tejas** is an initial proof of concept for computation-by-energy. It focuses on energy-aware model routing for AI inference.

For each request, Tejas chooses among several model sizes. Each model has different energy consumption and output quality. Tejas uses the workload requirements and the available energy information to select a model before inference begins.

The goal is to reduce energy consumption while satisfying a required quality level.

Across three inference deployments, Tejas reduces energy consumption by up to **48.6%**. Quality improves on one deployment and changes by less than half a percentage point on the other two. By comparison, using the same smaller model for every request causes large quality losses when the differences between model sizes are significant.

## Experimental deployments

We evaluated Tejas on three deployments. Energy consumption was measured directly during every experiment.

| Experiment | Available models        | Hardware                         | Energy measurement                            |
| ---------- | ----------------------- | -------------------------------- | --------------------------------------------- |
| Mac        | Qwen2.5 3B, 7B, and 14B | Apple M4 Pro with unified memory | `powermetrics`, measuring whole-package power |
| DGX-Qwen   | Qwen2.5 3B, 7B, and 32B | 1× NVIDIA A100 40 GB             | `nvidia-smi`, measuring GPU power             |
| DGX-Llama  | Llama 3B, 8B, and 70B   | 2× NVIDIA A100 40 GB             | `nvidia-smi`, measuring GPU power             |

The Mac measurements cover the complete system package. The DGX measurements cover GPU power only.

## Repository structure

The main source code is located under [`src/tejas/`](src/tejas/).

| Directory                                            | Description                                                                                                                                                                                                         |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`src/tejas/measurement/`](src/tejas/measurement/)   | Collects power measurements using `powermetrics` on macOS and `nvidia-smi` on NVIDIA systems. It integrates the power samples over time to calculate the energy used by each request.                               |
| [`src/tejas/benchmarking/`](src/tejas/benchmarking/) | Contains the benchmark tasks, grading logic, and experiment runner. The benchmark includes 24 database and SQL tasks and 16 semantic-reasoning tasks derived from [SemBench](https://github.com/SemBench/SemBench). |
| [`src/tejas/cost_model/`](src/tejas/cost_model/)     | Trains and evaluates models that predict energy consumption and output quality for a given task, model, and deployment.                                                                                             |
| [`src/tejas/routing/`](src/tejas/routing/)           | Contains the routing policies, energy-state handling, optimization logic, baselines, oracle policies, and policy-evaluation code.                                                                                   |
| [`analysis/`](analysis/)                             | Contains scripts for generating the figures used in the paper.                                                                                                                                                      |
| [`results/`](results/)                               | Contains the raw measurements and processed evaluation results.                                                                                                                                                     |
| [`slurm/`](slurm/)                                   | Contains the Slurm configuration used for the DGX experiments.                                                                                                                                                      |

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

All results reported in the paper are already available under `results/`. The commands in this section show how those results were produced.

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

The results from this experiment are used to generate Figure 5 in the paper.

## Run the energy-model ablation

The following command tests whether the learned energy model improves routing decisions:

```bash
.venv/bin/python3 -m tejas.routing.ablation_energy_model
```

This experiment compares the complete Tejas policy with a simpler policy that selects the smallest model predicted to satisfy the quality requirement.

The purpose is to determine whether explicit energy prediction provides an advantage beyond the quality constraint alone.

## Collect new energy measurements

Collecting new measurements is optional because the repository already includes the results reported in the paper.

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

contain the results used to generate Figure 5.

The files:

```text
ablation_energy_model_<deployment>.csv
```

contain the results from the energy-model ablation study.

Every numerical result reported in the paper can therefore be checked using the committed CSV files.

## Generate the figures

The paper figures are generated from the CSV files in `results/`. They are not stored as pre-generated binary files.

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
| `plot_ieee_routing.py`     | Paper figures for routing-policy results                          |
| `plot_ieee_cost_model.py`  | Paper figures for energy and quality prediction                   |
| `plot_ieee_evaluation.py`  | Figures for prediction accuracy, model transfer, and calibration  |
| `plot_ieee_budget.py`      | Figure 5, showing achieved quality under different energy budgets |

## Running individual checks

Most modules can also be executed directly:

```bash
python -m tejas.<package>.<module>
```

Many modules include a small self-check that verifies their main functionality and confirms that the stored results can be reproduced.
