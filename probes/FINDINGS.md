# Findings for "Towards Energy-Native Semantic Data Systems" (ICDE 2027 DEFT)

Thesis (Carsten): co-design the DBMS and LLM serving around one objective — minimise the energy
needed to produce, maintain and reuse useful semantic knowledge.

Setup: SemBench Movie (sf_2000, 2,000 reviews), vLLM 0.34 on one A100-40GB (DGX), Qwen2.5 family.
Energy = NVML GPU counter, windows >= 30 s, prefix cache reset before every run, idle ~59 W.
**Boundary: GPU energy only** (RAPL needs root on the node). State this in the paper.

---

## Shift 2 — why energy is not cost/performance (P0 = GreenQuery, existing)
- Up to **513x** energy spread among configurations that all return correct results.
- Engine choice: median **13x**; model size 3B->32B: median **2.8x**. Configuration > model.
- Stretto / Abacus use **22–26x** the energy of the best configuration.
- Energy vs. number of LLM calls: Spearman rho = **0.50**; a token-based estimator lands at ~**1/10** of measured energy.
- Q1: ThalamusDB 0.33 Wh vs. LOTUS 98.3 Wh.
- New supporting points from the probes:
  - Idle draw is ~16% of every measured window (P2) -> energy is non-additive, not a per-token price.
  - Quantised 7B-AWQ uses **~27% more** energy than 7B bf16 at equal accuracy (P1) -> size/tokens do not predict energy.

## Shift 2 — quality–energy, model choice (P1 = Tejas reframed on SemBench)
Q1 predicate on all 2,000 reviews, accuracy vs. fresh/rotten label, GPU energy per pass:

| Model | "clearly positive" (SemBench) | "positive" | Energy |
|---|---|---|---|
| 3B | 0.692 | 0.709 | 1.9 kJ |
| 7B | 0.839 | 0.863 | 3.8 kJ |
| 7B-AWQ | 0.863 | 0.871 | 4.8 kJ |
| 14B | 0.839 | **0.878** | 7.7 kJ |
| 32B-AWQ | **0.665** | 0.829 | 22.1 kJ |

- **Accuracy is not monotone in model size**, and one word ("clearly") flips which model is right:
  the 32B reads "clearly" strictly (rejects lukewarm 3/5 reviews with P(yes) ~ 0).
- -> "Largest model - 2 pp" (Tejas' reference) is unreliable: trivial with one wording, misleading with the other.
  Quality must be defined against an explicit **contract**, not against the biggest model.
- **Per-tuple routing adds little** once the right model is picked (plain wording, reference = 14B):
  static 7B 0.169x of 32B energy (-1.5 pp); learned router 0.167x (-1.8 pp); oracle bound 0.135x.
- **Confidence cascade loses** in every setting (0.41x–1.6x): small models are confidently wrong.
- Message: the energy lever is choosing the model **per predicate and deployment** (6x energy, +17 pp),
  i.e. a planning decision the DBMS must make — not a per-request router in front of the server.

## Shift 3 — joint DBMS–LLM execution (P2 = plan-aware execution, 7B, 5 reps, SD < 1.5%)
A = decoupled (one request per tuple), B = vLLM batching + prefix caching (fair serving baseline),
C = B + plan knowledge.

| Query | A | B | C_all | A->B | B->C_all | A->C_all |
|---|---|---|---|---|---|---|
| q1_limit | 22.8 kJ | 7.96 kJ | 0.046 kJ | 2.9x | **171x** | 489x |
| q5_join_limit | 69.2 kJ | 23.1 kJ | 0.65 kJ | 3.0x | **36x** | 107x |
| q6_twopred_full | 36.3 kJ | 12.9 kJ | 4.9 kJ | 2.8x | **2.6x** | 7.4x |
| q1_full | 22.7 kJ | 7.97 kJ | 3.97 kJ | 2.9x | **2.0x** | 5.7x |
| q5_join_full | 17.6 kJ | 6.22 kJ | 3.49 kJ | 2.8x | **1.8x** | 5.0x |

- Serving alone (A->B) is a flat ~2.8x; plan knowledge (B->C) multiplies on top. Neither layer gets the product alone.
- Per lever (vs. B):
  - **Pipelined LIMIT**: 16 calls instead of 2,000 (0.010x B) at equal precision. Co-design point: the DBMS
    must issue small waves (16) against the server's preferred batch (64) — a joint batch/early-stop decision.
  - **Projection**: ~0.50x on filters, quality unchanged (F1 0.883 -> 0.885); on the LIMIT join precision
    drops 0.98 -> 0.88 -> projection must be quality-aware (links to shift 2).
  - **Predicate ordering** (selectivity sampled, cost charged): 0.74x.
  - **Join order for KV reuse: no gain** (within 0.1% of B on the full join, ~5% on the LIMIT join
    = same as plain nested-loop order). vLLM prefix caching already captures it. Do not claim.
  - **Cheap CPU proxy for LIMIT: no gain**, worse precision (0.83) on the join. Do not claim.

## Shift 1 — semantic reuse and caching (P3, 7B + 1.5B, 3 reps, ±1%)
7-query workload over 1,418 reviews with graded scores (positive, rave, negative, pan, "liked",
positive for one movie, positive refresh).

| Strategy | GPU energy | vs. recompute | Large-model calls | Reuse |
|---|---|---|---|---|
| R0 recompute | 17.4 kJ | 1.00x | 8,605 | 0% |
| R1 exact tuple cache (what an LLM cache can do) | 14.5 kJ | 0.84x | 7,090 | 18% |
| R2 + declared predicate lattice | 8.7 kJ | 0.50x | 4,254 | 51% |
| R3 + LLM equivalence check (undeclared predicates) | 5.9 kJ | 0.34x | 2,836 | 67% |
| R4 + small model for residual narrower predicates | 3.8 kJ | **0.22x** | 1,418 (+1,418 small) | 67% |

- Reusing **answers** through predicate logic (subsumption, complement) beats reusing computation
  (exact/prefix caches): 51–67% vs. 18% reuse.
- **Reuse is only valid where the model is self-consistent** — measured from R0:
  rave => positive 95%, pan => negative 89%, negative == not-positive 88%.
  Deriving "negative" from "not positive" costs F1 0.79 -> 0.745; aliasing "liked" costs 0.847 -> 0.821.
  -> The DBMS must know the model's consistency to decide when a rewrite is admissible
  (contract-relative correctness; a concrete reason for co-design, not just caching).
- Caveat: rave/pan ground truth is weak (F1 0.54 / 0.24 even with recompute) — no quality claims on those.

## Shift 4 — energy-native orchestration (P4, simulation)
German 2025 hourly grid trace (Energy-Charts / SMARD, CC BY 4.0): carbon 106–741 g/kWh, 576 negative-price
hours. Deferrable semantic-maintenance jobs, EDF placement into cheapest slots before the deadline.

| Slack | Carbon vs. run-immediately |
|---|---|
| 2 h | 0.94x |
| 4 h | 0.89x |
| 8 h | 0.80x |
| 12 h | 0.73x |
| 24 h | 0.63x |

- No deadline misses. Optimising for price instead: 0.41x cost at 24 h slack.
- Compounds with reuse: job energy from P3 (4.83 Wh recompute -> 1.06 Wh with reuse) + 24 h slack
  -> 4.10 kg -> 0.57 kg CO2 = **0.14x**.
- Caveat: simulation on a real trace with measured job energy, not a hardware run.

---

## Claims to make
1. Energy is a different objective from tokens, calls or model size (P0, idle share, AWQ result).
2. The largest gains need knowledge only the DBMS has (LIMIT, projection, predicate order, predicate lattice),
   and they multiply with serving optimisations rather than replace them (P2, P3).
3. Correctness of energy-saving rewrites is contract- and model-relative (P1 wording, P3 consistency).
4. Time is a free dimension for deferrable semantic maintenance (P4), and it compounds with reuse.

## Claims to avoid
- "KV-aware join ordering saves energy" — not on this data/stack.
- "Cheap proxies make LIMIT cheaper" — they did not.
- "Per-tuple routing / cascades save substantial energy" — they do not here; model choice per predicate does.
- "Bigger models are worse" — only true for one wording; say "model choice depends on the predicate/contract".
- Whole-system energy numbers — GPU only.

## Still running / open
- P1 on a second predicate (SemBench Q5 same-sentiment pairs, 3B/7B/14B/32B-AWQ) — job 50147.
- P2 on a second model (14B, 3 reps) — job 50148.
- P2 figure not generated yet (no matplotlib on the DGX).
- 32B tested only quantised (bf16 does not fit one 40 GB A100); 7B-AWQ shows quantisation does not hurt accuracy.

## Where the numbers are
`probes/results/p2/50136-*/summary.csv`, `probes/results/p3/50137-*/`,
`probes/results/p1/ladder-{plain,clearly}/policies_ref-*.json`, `probes/results/p4/de2025_p3_*wh/`.

## P2 wave-size sweep (job 50149, Qwen2.5-7B, A100; results/p2_wave/50149-*)
Pipelined LIMIT filter; wave w = requests released before checking the LIMIT. Metric: marginal GPU energy
(window energy - idle power x time), mean of 3 reps. Selectivity: "positive" ~50-60% qualify, "acting" ~17-23%.

| pred / LIMIT (bg=0) | best w | E(best) | w=1 vs best | w=256 vs best |
|---|---|---|---|---|
| positive / 5  | 16 (8 equal) | 37 J  | 1.9x | 11.9x |
| positive / 50 | 16 | 206 J | 2.8x | 2.2x |
| acting / 5    | 16 | 63 J  | 2.5x | 7.3x |
| acting / 50   | 64 (16 within noise: 717 vs 662 J, sd ~70) | 662 J | 3.2x | 1.5x |

- Claim: neither extreme is optimal; w=1 (no wasted calls) costs 1.9-3.2x, w=256 (max batching) 1.5-12x.
- Optimum shifts 16 -> 64 only for rare predicate + LIMIT 50, and that shift is within noise: do NOT claim a strong shift.
- bg=32 (shared server): marginal energy is negative/unusable (foreground displaces background in shared batches,
  so window-minus-background attribution fails). Report nothing about energy under load. Latency under load: w=1
  is 8-110 s vs 0.4-3 s for w>=64.
- Outlier: acting/50/w=256 latency 203 s (server stall, mostly idle power); its marginal energy is unaffected.

## P1 second predicate: Q5-style join predicate (job 50147; results/p1/50147-q5pairs)
2,000 random unordered pairs of the 128 Ant-Man reviews (SemBench Q5 movie); predicate "same sentiment";
base rate 49.95% (chance = 0.50). Contract: acc >= best static tier - 2 pp; best = 14B (0.739).

| tier | acc | E/E_32B |
|---|---|---|
| 3B | 0.512 (chance) | 0.084 |
| 7B | 0.570 | 0.168 |
| 14B | **0.739** | 0.340 |
| 32B-AWQ | 0.666 (fails) | 1.000 |
| learned router | 0.726 | 0.324 |
| oracle | 0.823 | 0.208 |
| cascade | 0.739 | 0.593 (1.7x static 14B) |

- Claim: the contract-relative model choice flips between predicates of the same workload: 7B for Q1, 14B for
  Q5-pairs (7B is 17 pp below there). 32B fails the contract on both. Router ~= best static again.
- Caveat: pairs are our construction from the Q5 movie, not the SemBench Q5 query itself.

## P2 on a second model: Qwen2.5-14B (job 50148; results/p2/50148-Qwen_Qwen2.5-14B-Instruct)
- Batching (A/B): 2.5-2.7x (7B: 2.8-3.0x). Plan-aware (B/C_all): 1.9-194x on queries without quality loss.
  Q5 LIMIT join: 442x but precision 0.86 -> 0.70 (projection) -- do NOT cite 442x.
- Mechanisms replicate: early stop 0.01x, projection (Q1 full) 0.50x, predicate order 0.74x, KV join order 1.00x.
- Projection on Q5 LIMIT join hurts on both models (7B P .98->.88; 14B P .86->.73).
- CPU-proxy ordering is model-dependent in energy (7B 1.37x, 14B 0.38x) but loses precision on both
  (7B .98->.83, 14B .86->.70). Claim: "never safe for quality", not "always costs energy".

## P2 wave-size sweep under shared load (job 50152; results/p2_wave_load/50152-*)
Server shared with 32 closed-loop background requests. Work-based attribution: background charged for the
requests it completes at its stand-alone J/request (paired 20 s calibration window per config); remainder =
foreground marginal energy. 3 reps, mean ± sd.

| pred / LIMIT | best w (load) | E best load | E best idle | fastest w (load) | E fastest / E best |
|---|---|---|---|---|---|
| positive / 5  | 4-16 (9 J tie) | 9 J   | 37 J  | 16  | 1.0x |
| positive / 50 | 16 (w4 59±18 within noise) | 54 J | 206 J | 256 (1.40 s) | 2.9x (156 J) |
| acting / 5    | 4-16 (w1 -11±27 = noise) | 10-15 J | 63 J | 64 (0.51 s) | 2.7x (41 J vs w16 15 J) |
| acting / 50   | 64 (w16 211±15 equal) | 202 J | 662 J | 64 | 1.0x |

- Optimum stays moderate (4-64) under load; it does NOT shift to large waves. The early two-sample hint
  (w256 < w1) was noise (w1 85±128 J). Do not claim a load-driven shift.
- Same query adds 3-4x less energy on the busy server (37->9, 206->54, 662->202 J): marginal energy depends on
  serving state.
- Idle: fastest wave = least-energy wave in all 4 settings. Under load: they diverge in 2 of 4; the fastest
  wave costs 2.7-2.9x the least-energy one. -> direct evidence that energy and latency rankings diverge under
  shared serving (supports Related Work latency paragraph).
- Full batches (w256) vs best under load: 2.9-18x (acting/50/w256 excluded: repeat server stall, 203 s, sd 1588 J).
