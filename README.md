# Tejas — Energy-Native AI Data Systems

<h1 align="center">
  <img src="tejas_logo.png" alt="Tejas Logo" width="300"/>
</h1>


Reference implementation and results for the CIDR 2027 vision paper
**"Sustainable AI Data Systems: From Computation-on-Demand to
Computation-by-Energy."** Tejas measures real per-request LLM inference
energy on real hardware, learns a per-task energy/quality cost model from
that data, and uses it to route each request to the cheapest model tier
that doesn't cost meaningful quality — recovering nonzero, statistically
bounded energy savings even where a hand-picked threshold policy finds
none.

Three experiments, real measured energy throughout (never estimated):

| Experiment | Model ladder (small/medium/large) | Hardware | Energy measurement |
|---|---|---|---|
| Mac | Qwen2.5 3b / 7b / 14b | Apple M4 Pro (unified SoC) | `powermetrics` — whole-package power |
| DGX-Qwen | Qwen2.5 3b / 7b / 32b | 1× NVIDIA A100-40GB | `nvidia-smi` — GPU die only |
| DGX-Llama | Llama3 3b / 8b / 70b | 2× NVIDIA A100-40GB | `nvidia-smi` — GPU die only |

**Headline result**: the learned cost-model policy recovers 48.6% energy
savings on Mac (+6.6pp quality vs. static routing) and, unlike the
hand-picked `scarce, calibrated` threshold policy (which recovers exactly
0% on both DGX experiments), a bounded nonzero saving on DGX-Qwen (9.7%)
and DGX-Llama (1.4%) as well. Full results: [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) §10.

## The four pillars

The paper frames the system as measurement → benchmarking → cost modeling
→ routing, a strict, acyclic pipeline. The code mirrors that directly:

| Pillar | Package | What it does |
|---|---|---|
| 1. Measurement | [`src/tejas/measurement/`](src/tejas/measurement/) | Real energy measurement: `powermetrics` (macOS) / `nvidia-smi` (DGX), trapezoidal integration to joules per request |
| 2. Benchmarking | [`src/tejas/benchmarking/`](src/tejas/benchmarking/) | 40-task harness (24 DB/SQL tasks + 16 [SemBench](https://github.com/SemBench/SemBench)-derived semantic-reasoning tasks), grading, and the experiment runner |
| 3. Cost modeling | [`src/tejas/cost_model/`](src/tejas/cost_model/) | Learned `E_hat(task, model, deployment)` / `Q_hat(task, model, deployment)` ladders, validated with repeated nested grouped cross-validation |
| 4. Routing | [`src/tejas/routing/`](src/tejas/routing/) | Hand-designed baseline policies, two measured oracles, and the cost-model-driven router with bootstrap-CI'd evaluation |

`analysis/` (figures) and `slurm/` (DGX job submission) sit outside the
installable package — they're consumers of `tejas`, not part of the system
itself. `docs/` has the full research narrative.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Installs `tejas` in editable mode from `src/tejas/` plus its dependencies
(numpy, scipy, pandas, scikit-learn, matplotlib — see `pyproject.toml`).
Console-script shortcuts (`tejas-run-experiment`, `tejas-evaluate-models`,
`tejas-evaluate-policy`, `tejas-cross-family-transfer`,
`tejas-fewshot-calibration`, `tejas-static-policies`) are installed
alongside `python -m tejas...` invocation.

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

**2. Simulate hand-designed routing policies** (table lookup over already-collected results, no re-run needed):

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

**4. Evaluate the learned router against baselines and both oracles:**

```bash
.venv/bin/python3 -m tejas.routing.evaluate_policy
```

Every module above also has a `python -m tejas.<pillar>.<module>` self-check
(most were built with one from the start) — run any of them directly to
verify your environment reproduces the same numbers.

## Results and figures

`results/` holds every raw and derived CSV this project has produced —
raw per-(run, task, model) measurements, model-evaluation metrics, policy
decisions, baseline comparisons, transfer and calibration results — so
every number in the paper is independently checkable. Figures are
regenerated from these CSVs rather than committed as binaries:

```bash
.venv/bin/python3 analysis/plot_results.py            # per-experiment figures
.venv/bin/python3 analysis/plot_cross_experiment.py   # Mac vs. DGX-Qwen vs. DGX-Llama
.venv/bin/python3 analysis/plot_ieee_routing.py        # IEEE-style paper figures, routing
.venv/bin/python3 analysis/plot_ieee_cost_model.py     # IEEE-style paper figures, cost model
.venv/bin/python3 analysis/plot_ieee_evaluation.py     # prediction accuracy, transfer, calibration figures
```

See [`docs/FIGURES_GUIDE.md`](docs/FIGURES_GUIDE.md) for what each figure shows and which claim it supports.

## Citation

CIDR 2027 submission (Pratyush Agnihotri, DFKI and TU Darmstadt). Citation
details to follow once accepted.
