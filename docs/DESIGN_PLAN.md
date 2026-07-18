# Energy cost model for AI data systems (v3, corrected)

## Context

`docs/RESEARCH_LOG.md` §6 Finding 3 / §9: the `scarce, calibrated` policy fixes
DGX's quality collapse (Finding 2) but with a single binary, hand-picked,
tier-average threshold — on DGX it refuses every downgrade, 0% energy
saved. v1 of this plan fixed that with a per-task epsilon-sweep on the
static-vs-scarce pair, but review found it still routing-centered: energy
never actually drove the decision, search excluded 2-tier downgrades,
epsilon was tuned on the same data it was scored on. v2 reframed around a
proper energy/quality cost model (`E_hat`, `Q_hat`) with nested CV,
bootstrap CIs, oracle regret, and cross-deployment transfer — but a second
review pass found six further issues, two of which (oracle misalignment,
cross-dataset task overlap) would have invalidated the strongest reported
numbers. This version fixes all six. No new experiment runs in any
version — everything comes from the three existing CSVs.

## What changed from v2 (the six fixes)

1. **Cross-dataset transfer conflates two different questions.** All three
   CSVs share the same 40 task IDs, so "train on DGX-Qwen, test on
   DGX-Llama" tests *deployment* transfer while workload is held constant —
   it does not test transfer to *unseen* workloads. This only matters for
   the cross-family/cross-platform legs, which are now explicitly tiered
   below as **Next**, not built this session — but the protocol is
   specified correctly now so it's not redone later: report both **Protocol
   A** (matched-workload: same 40 tasks, different deployment) and
   **Protocol B** (unseen-workload: task IDs split before crossing
   deployments, repeated splits since there are only 40). Also drop the
   claim that this "isolates model family" — family, parameter range, and
   GPU count (1 vs 2) all differ simultaneously between Qwen-DGX and
   Llama-DGX, so it's a joint deployment-transfer test, not a clean
   single-factor one.

2. **The `delta=0` sanity check was wrong.** `Q_hat(q,m) >= Q_hat(q,ref) -
   delta` is a constraint on *predicted* quality; prediction error can
   still select a genuinely worse tier on an unseen test task. Fixed sanity
   check: "every non-reference selection satisfies the *predicted*
   non-inferiority condition by construction; *measured* violations of the
   −2pp bound are possible and are reported as an explicit metric (task-level
   violation rate, aggregate violation magnitude, fraction of bootstrap
   samples violating the bound)" — the violation rate is part of what makes
   this evaluation informative, not a bug to suppress.

3. **The oracle was misaligned with the policy's own objective.** A
   per-task zero-quality-loss oracle answers a different, more conservative
   question than a policy operating under a global −2pp *mean* budget. Now
   reporting **two oracles**: (a) strict per-task oracle (min energy s.t.
   that task's quality ≥ reference, no exceptions — conservative,
   interpretable), and (b) global quality-budget oracle: choose one tier
   per task minimizing total measured energy subject to the same aggregate
   constraint the policy is held to (`mean(Q(q,m_q) - Q(q,m_ref)) >=
   -0.02`) — solved exactly as a small 0/1 multiple-choice knapsack/DP over
   40 tasks × 3 tiers (tiny, exact, no approximation needed). (b) is the
   correct upper bound for the headline non-inferiority result; (a) is the
   conservative reference point. `savings_recovered = (E_ref - E_policy) /
   (E_ref - E_oracle)` is only meaningful, and only reported, when the
   policy itself satisfies the same aggregate constraint — if it doesn't,
   report it as constraint-violating, not as a savings-recovered number.

4. **Delta (and model-type) selection must use measured inner-OOF policy
   outcomes, not predicted energy.** Predictions pick *which tier* a given
   task routes to; but *which configuration* (energy-model type ×
   quality-model type × delta × abstention on/off) is best must be judged
   by running the full policy on inner out-of-fold predictions and scoring
   it against the *actual measured* rows for those tasks (same lookup
   `tejas.routing.static_policies` already does) — then picking, among configurations whose
   one-sided 95% lower CI on quality-diff is ≥ −2pp, the one with minimum
   *measured* inner-OOF energy (ties broken by smaller delta, simpler
   model, lower abstention rate). This also means energy-model and
   quality-model types are selected **jointly** for decision quality, not
   independently by their own standalone metrics — a slightly-worse-MAE
   model can still produce a better tier ranking.

5. **Measurement boundary must be checked, not assumed, before any
   cross-platform energy claim.** Checked directly against
   `energy_meter.py` / `gpu_energy_meter.py`: Mac's `powermetrics` reports
   `Combined Power (CPU + GPU + ANE)` — whole-SoC package power (Apple
   Silicon has no separate host/accelerator boundary). DGX's
   `gpu_energy_meter.py` reports `nvidia-smi --query-gpu=power.draw`,
   scoped to just the job's allocated GPU(s) via `CUDA_VISIBLE_DEVICES` —
   GPU silicon only, host CPU/RAM explicitly excluded. **These are
   different physical measurement boundaries, not just different scales.**
   `deployment_features.py` now carries a `measurement_scope` field
   (`"package_cpu_gpu_ane"` for Mac, `"gpu_die_only"` for DGX), and the
   model-fitting code refuses (assertion, not a silent no-op) to pool
   absolute-J targets across rows with different `measurement_scope`.
   Cross-platform experiments are relabeled **relative tier-energy
   transfer** (ratio target `E(q,m)/E(q,m_small)`, same deployment's own
   denominator, model predicts the ratio directly so the test task's
   denominator is never needed at inference time) — not "cross-platform
   absolute energy cost model," which would overclaim.

6. **Single 5-fold outer split is unstable with 40 tasks.** Now repeated
   nested grouped CV: outer 5-fold × 5 repeats (different shuffles), inner
   4-fold `GroupKFold` on each outer-train split, all folds group-safe on
   `task_id`. Folds are constructed with `StratifiedGroupKFold` (sklearn
   ≥1.1, confirmed installed at 1.9.0) stratified by `task_type_tier`, so
   no fold starves a workload class (16/40 tasks are `semantic_hard` — a
   plain random grouping could easily leave one inner fold with very few).
   Report the distribution across the 5 repeats (not just one number),
   plus model-selection and delta-selection frequency across repeats.

## Scope for this session: adopting the reviewer's own tiering

The second review explicitly tiered its own recommendations and said the
first tier alone "would already provide the essential evidence for the
CIDR paper." Adopting that directly:

- **Required core (this session, built and verified):** feature
  extraction + `measurement_scope` audit, energy/quality model ladders with
  fold-local sklearn `Pipeline`s, repeated nested grouped CV, all-tier
  optimizer, both oracles, within-deployment results for all 3 experiments,
  bootstrap CIs, prediction/policy ablations, calibration diagnostics.
- **Next (specified correctly now, built after Required core if time
  remains, otherwise handed off precisely via §8):** matched- and
  unseen-workload cross-family transfer (Protocol A/B), relative
  cross-platform transfer.
- **Final extension (spec'd, deferred):** few-shot deployment calibration
  (with the linear-calibration baseline below, not just full refit),
  margin-based abstention fallback.

## Module layout

```
task_features.py        offline feature extraction (no ML, no re-run)
deployment_features.py  grounded table: platform/family/param-count/gpu-count/measurement_scope
energy_cost_model.py    E_hat ladder + fold-local Pipeline + log-energy smearing back-transform
quality_model.py        Q_hat ladder + fold-local Pipeline + fold-safe sigmoid calibration
oracle.py               strict per-task oracle; global quality-budget oracle (exact DP)
policy_optimizer.py     m*(q) = argmin_{m: param<=param(ref)} E_hat(q,m) s.t. Q_hat(q,m) >= Q_hat(q,ref) - delta
evaluate_models.py      repeated nested grouped CV; prediction-quality metrics only (no routing)
evaluate_policy.py      joint config selection (measured inner-OOF), within-deployment routing outcomes, both oracles, bootstrap CIs
```

### `task_features.py`

`{task_id: {...}}` from `tasks.TASKS` + `sembench_tasks.TASKS` (same 40
tasks in all 3 experiments):
- `task_type`, `task_tier`, joint `task_type_tier` (e.g. `sql_hard`,
  `semantic_hard`) — read as workload-class effects, not difficulty or
  semantic effects in isolation (the actual confound).
- `prompt_chars`, approximate `prompt_tokens` (regex word/punctuation
  split — explicitly *not* a per-model tokenizer).
- SQL: **lightweight lexical proxies** parsed from `gold_sql` (the task
  definition, pre-execution) — `n_joins`, `n_predicates`, `n_aggregations`,
  `n_groupby_cols`, `n_projected_cols`, `has_subquery`, `has_window_fn`.
  Named as proxies, not "structural analysis," because regex parsing can
  miscount commas-in-functions / nested AND / aliased columns. Validated
  with a small hand-checked expected-count table for a handful of the 24
  SQL tasks (printed when the module runs standalone, same pattern
  `sembench_tasks.py` already uses for its own self-check).
- Semantic: `n_reviews`, `answer_format`.

### `deployment_features.py`

```python
DEPLOYMENTS = {
  "mac":       {"platform": "mac", "device_class": "apple_silicon", "model_family": "qwen",
                "gpu_count": 0, "measurement_scope": "package_cpu_gpu_ane"},
  "dgx_qwen":  {"platform": "dgx", "device_class": "nvidia_a100", "model_family": "qwen",
                "gpu_count": 1, "measurement_scope": "gpu_die_only"},
  "dgx_llama": {"platform": "dgx", "device_class": "nvidia_a100", "model_family": "llama",
                "gpu_count": 2, "measurement_scope": "gpu_die_only"},
}
```
`param_count_b(model_name)` via regex, `log_param_count = log1p(param_count_b)`
(3–70B is heavily skewed — raw values would let the largest tier dominate
a linear model). `params_per_gpu = param_count_b / gpu_count` defined only
for DGX rows (`gpu_count=0` on Mac would divide by zero — Mac is treated as
a distinct execution architecture, not "0 GPUs of capacity").

### `energy_cost_model.py` / `quality_model.py`

Fold-local `sklearn.pipeline.Pipeline` wrapping a `ColumnTransformer`
(`OneHotEncoder(handle_unknown="ignore", sparse_output=False)` for
categoricals, `StandardScaler` for numerics) + estimator — every encoder/
scaler/calibrator is fit inside the training fold only, never on the full
dataset first (matters most once a whole platform/family is held out and
unseen categories appear).

- Energy ladder: `GlobalMeanEnergy`, `TierLookupEnergy` (mean log-energy
  per `(deployment, model_tier)`), `Ridge`, `ElasticNet`,
  `HistGradientBoostingRegressor` — target `log(energy_j)`. Back-transform
  uses a **training-fold-only Duan smearing correction**
  (`c = mean(exp(residuals))`, `E_hat = c * exp(logE_hat)`) instead of bare
  `exp()`, which would systematically underpredict the arithmetic mean
  (geometric-mean bias). Matters for joule MAE and absolute-savings
  numbers even though it can cancel out for pure tier-ranking.
- Quality ladder: `TierBaselineQuality`, `LogisticRegression`,
  `HistGradientBoostingClassifier` — the latter with fold-safe sigmoid
  (Platt) calibration fit only on inner-training data (isotonic is unsafe
  at this sample size).
- Explicit interactions `log_param_count × is_semantic`,
  `log_param_count × platform` are added for the **linear models only**
  (Ridge/ElasticNet/LogReg can't otherwise represent the sign flip Finding
  4 shows for semantic tasks); tree models learn interactions natively, so
  they get the interaction *features* only as part of an explicit ablation
  (base workload/model features → +SQL/semantic features → +interactions →
  +deployment features), not by default — this ablation is itself a
  reportable result about which feature families matter.
- Normalized/ratio energy target reserved for the (deferred, Next-tier)
  cross-platform legs — never substituted into within-deployment absolute-J
  models.

### `oracle.py`

- Strict: per task, `argmin measured_energy(m)` s.t.
  `measured_quality(m) >= measured_quality(m_ref)`.
- Global budget: exact DP over 40 tasks × (≤3 feasible tiers each),
  minimizing total measured energy s.t. mean quality delta across all 40
  tasks ≥ −2pp — small enough (≤3^40 branching but DP state is just
  running sum of quality-delta at a fixed discretization, or since only 4
  runs ⇒ quality per task takes values in {0,.25,.5,.75,1}, the deltas are
  finitely many, making an exact DP/ILP-free enumeration over per-task
  choices with a running-budget table entirely tractable).

### `policy_optimizer.py`

```python
def select(task_row, candidates, reference_tier, delta, energy_model, quality_model):
    q_ref = quality_model.predict_proba(task_row, reference_tier)
    feasible = [m for m in candidates if param_count_b(m) <= param_count_b(reference_tier)
                and quality_model.predict_proba(task_row, m) >= q_ref - delta]
    return min(feasible, key=lambda m: energy_model.predict(task_row, m))
```
`reference_tier` always feasible by construction (predicted, not measured
— see fix #2). Margin-based abstention (`margin = Q_hat(q,m) - (Q_hat(q,ref)
- delta) >= tau`, else fall back to `reference_tier`) is implemented but
kept as an **optional ablation flag**, selected/off by default — Final
extension tier, not part of the primary configuration.

### `evaluate_models.py` — prediction-quality metrics only

Repeated nested grouped CV (outer 5×5, inner 4, `StratifiedGroupKFold` by
`task_type_tier`). Reports, per experiment, out-of-fold:
- energy: MAE (J), median absolute % error, RMSE(log-energy), R²,
  average per-task Spearman correlation across the 3 tiers, fraction of
  tasks where the (quality-blind) minimum-energy tier is correctly
  identified. ("Feasible" minimum-energy tier — i.e. quality-gated — moves
  to `evaluate_policy.py`, since feasibility depends on the quality model.)
- quality: accuracy (secondary), Brier score, log loss, 5-bin calibration
  reliability, calibration slope/intercept, and `ΔQ` error
  (`|Q_hat(q,m)-Q_hat(q,ref) - (Q(q,m)-Q(q,ref))|` — closer to what the
  policy actually consumes than raw probability accuracy).
- which ladder member wins per outer repeat (not asserting one universal
  winner).

### `evaluate_policy.py` — joint config selection + routing outcomes

Per outer fold: inner loop computes inner-OOF predictions once per
(energy-model type × quality-model type) pair on outer-*training* tasks,
then sweeps `delta` (and abstention on/off) cheaply over those fixed
predictions — for each configuration, runs the actual policy and scores it
against **measured** inner-OOF rows (fix #4), keeps configurations with
one-sided lower-95%-CI quality-diff ≥ −2pp, picks minimum measured energy
among those (ties: smaller delta → simpler model → lower abstention rate).
That fold-and-repeat-specific configuration is applied once to the
untouched outer-test tasks. Report the distribution across the 5 repeats,
not a single point estimate.

Metrics (task-level paired bootstrap CI, resample `task_id` with
replacement, ~2000 resamples, 95%, pooled from concatenated outer-test
folds):
- quality Δ vs. `static-cost-router`, % energy saved vs. `static-cost-router`
- predicted-constraint violation rate (should be 0 by construction) vs.
  **measured** non-inferiority violation rate/magnitude (fix #2 — reported,
  not assumed away)
- both oracles' savings, `savings_recovered` (only interpreted for
  configurations that satisfy the aggregate constraint — fix #3)

Writes `results/policy_summary_learned_<experiment>.csv` (matches
`tejas.routing.static_policies`'s naming convention).

**Next tier (specified, not built unless Required core finishes with time
to spare):** Protocol A (matched-workload) and Protocol B (unseen-workload,
task IDs split before crossing deployments, repeated splits) cross-family
transfer between DGX-Qwen/DGX-Llama; relative tier-energy transfer across
Mac↔DGX. **Final extension:** few-shot calibration — compared against a
**linear calibration baseline**, not just full refit: `logE_target = a +
b*logE_hat_source`, fit only intercept `a` at k=1 target task, both `a`,`b`
at k=5/10 — this directly answers "is a new deployment a cheap scale-shift
or does it need new task–model interactions," a sharper systems question
than "more data helps."

## Docs

- `requirements.txt`: add `scikit-learn`.
- `README.md`: new "Energy cost model" section (how to run
  `evaluate_models.py` / `evaluate_policy.py`).
- `docs/RESEARCH_LOG.md`: new `## 10. Energy cost model — results` covering
  Required-core methodology and results (ladder comparison, calibration
  diagnostics, within-deployment policy performance with bootstrap CIs,
  both oracles' regret/savings-recovered), the `measurement_scope` finding
  (Mac vs. DGX aren't measuring the same physical boundary — a genuine
  methodological finding, not just a caveat), and Next/Final-extension
  items correctly specified as explicit future work (§8) rather than
  rushed or silently dropped.

## Verification (replaces v2's incorrect sanity list)

- No `task_id` crosses an inner or outer fold boundary, in any repeat.
- Fold-local preprocessing: encoders/scalers/calibrators fit only on
  training rows within each fold; unknown categories handled explicitly
  (relevant once Next-tier transfer work holds out a whole platform/family).
- `reference_tier` always in the feasible set; every non-reference
  selection satisfies the *predicted* constraint by construction.
- Measured non-inferiority violations are computed and reported, not
  assumed impossible.
- Energy predictions finite and strictly positive after the smearing
  back-transform.
- Both oracles individually satisfy their own constraints; `savings_recovered`
  reported only when the evaluated policy also satisfies the aggregate
  constraint, else flagged as constraint-violating.
- Shuffling task features (keeping deployment/model features intact)
  degrades task-aware prediction and routing metrics but leaves
  deployment/model marginals intact (leakage check).
- The feature ablation (base → +SQL/semantic → +interactions →
  +deployment) produces the expected monotonic-ish improvement, reported
  as a result, not just a check.
- Repeated outer splits don't produce qualitatively contradictory
  headline conclusions (if they do, that itself gets reported, not hidden).

The concrete bar: does the within-deployment learned policy recover a
nonzero, CI-bounded-non-inferior share of the **global-budget** oracle's
energy savings on DGX-Qwen/DGX-Llama, where `calibrated` today recovers
exactly 0%? No UI — verification is running each script and inspecting
these properties plus that headline number.
