# Feasibility probes for "Towards Energy-Native Semantic Data Systems"

One probe per architectural shift. All probes use the SemBench Movie data (sf_2000),
an OpenAI-compatible server (vLLM on the DGX), and the GreenQuery measurement method
(NVML GPU energy counter + RAPL package counter, idle baseline, windows >= 30 s).

| Probe | Shift | Question | Code | Job |
|---|---|---|---|---|
| P0 | 2 | Is energy different from tokens/cost? | GreenQuery (existing results) | - |
| P1 | 2 | Does the energy-optimal safe model differ per tuple/deployment? | `p1_tejas_sembench/` | `slurm/p1_tiers.sbatch` |
| P2 | 3 | What does plan knowledge save beyond vLLM (batching + prefix caching)? | `p2_plan_aware/` | `slurm/p2_vllm.sbatch` |
| P3 | 1 | Can the DBMS reuse semantic results where an LLM cache cannot? | `p3_semantic_reuse/` | `slurm/p3_vllm.sbatch` |
| P4 | 4 | Can deferrable semantic maintenance exploit time-varying energy? | `p4_temporal/` | runs on a laptop |

## P2 strategies
- **A** decoupled: one request per tuple, sequential, LIMIT applied after the scan.
- **B** serving-aware: same requests issued concurrently (vLLM batching + automatic prefix caching).
  `B_nested` = B with join pairs in nested-loop order (to check whether the join-order lever is already free).
- **C_*** each plan-level lever alone on top of B, and **C_all**:
  `proj` (only the needed attribute), `limit` (pipelined LIMIT in waves of 16), `proxy` (cheap CPU
  ordering for LIMIT), `join_order` (outer-grouped pairs for KV reuse), `pred_order`
  (conjunctive semantic predicates ordered by cost/(1-selectivity), selectivity sampled and charged).
- Quality is reported per strategy. `proj` can change quality (less context); check it before claiming it.
- Note: SemBench reviews are short (~150 chars), so prefix-reuse levers may show small gains on this data.

## P3 strategies
R0 recompute -> R1 exact tuple cache -> R2 + declared lattice (rave ⊑ positive, pan ⊑ negative,
negative ≡ ¬positive) -> R3 + LLM equivalence check for undeclared predicates -> R4 + small model for
residual narrower predicates. Ground truth for rave/pan comes from graded critic scores (1,425 reviews).
R0 also reports model consistency (does the model's own recomputation respect the lattice?).

## P1 policies (offline, grouped 5-fold CV by movie)
static per tier, uniform downgrade, Tejas-style learned router (pre-inference features), confidence
cascade (first-token P(yes)), oracle. Requirement: accuracy >= largest - 2 pp.

## Running on the DGX
```bash
# once, on the login node
uv venv --python 3.12 ~/pratyush/probes-venv
uv pip install --python ~/pratyush/probes-venv/bin/python vllm openai httpx pynvml numpy
rsync -a ~/AI_Agent/sustainableAI/probes <dgx>:~/pratyush/sustainableAI/   # from the laptop
# jobs (from ~/pratyush/sustainableAI)
sbatch probes/slurm/p2_vllm.sbatch                 # run first: decides the co-design thesis
sbatch probes/slurm/p3_vllm.sbatch
sbatch probes/slurm/p1_tiers.sbatch
# analysis
python -m probes.p2_plan_aware.analyze_p2 probes/results/p2/<run>
python -m probes.p1_tejas_sembench.analyze_p1 probes/results/p1/<run> small,medium,large best   # or: large (Tejas reference)
python -m probes.p4_temporal.simulate_p4 --trace <german_hourly_trace.csv> --job-energy-wh <from P3> --out probes/results/p4
```

## Local smoke tests (Ollama, no energy counters)
```bash
PROBE_BASE_URL=http://localhost:11434/v1 PROBE_MODEL=llama3.2:1b PROBE_CONCURRENCY=8 \
  probes/.venv/bin/python -m probes.p2_plan_aware.run_p2 --smoke --out /tmp/p2_smoke
```
