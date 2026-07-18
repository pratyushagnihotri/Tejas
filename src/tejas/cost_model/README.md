# Energy Cost Model for Energy-Aware LLM Routing

This package answers the question `docs/RESEARCH_LOG.md` §9 left open: given a
task and a candidate model tier, how much energy will it cost and how much
quality will it lose or gain, and can a router use that to make a
per-task, continuous trade-off decision instead of one hand-picked,
all-or-nothing threshold? It's written to be adapted directly into the
paper's methodology/evaluation section — the section headers below roughly
correspond to what a "System Design" and "Evaluation Methodology" section
would need.

## 1. Problem this replaces

Prior to this package, the router had two policies, both hand-designed:

- `scarce`: downgrade every task in a tier by exactly one model size under
  simulated energy scarcity. Free on Mac (small model gap), but backfired
  badly on DGX (−12.6pp to −33.7pp quality) once the gap between the
  cheapest and most capable model widened (`docs/RESEARCH_LOG.md` Finding 2).
- `scarce, calibrated`: only take that same one-size downgrade if the
  *tier-average* measured quality gap is ≤10pp, else don't downgrade at
  all for that tier. This fixed the DGX quality collapse but was binary
  per tier — on DGX the average gap was always >10pp, so it downgraded
  *nothing*, recovering 0% of the available energy savings (Finding 3).

Both are single, fixed rules applied uniformly to every task in a
difficulty tier. Neither can express "this specific hard SQL task is safe
to downgrade even though hard tasks *on average* aren't."

## 2. What this package does instead

Two learned functions, fit per deployment (Mac / DGX-Qwen / DGX-Llama —
kept separate, see §5 on why they aren't pooled):

- `E_hat(task, model_tier) -> predicted joules`
- `Q_hat(task, model_tier) -> predicted P(correct)`

and a router that, for each task, searches every model tier no more
expensive than that task's reference (energy-abundant/static) tier, and
picks the cheapest one whose predicted quality is within `delta` of the
reference tier's predicted quality:

```
m*(task) = argmin_{m : param(m) <= param(reference)} E_hat(task, m)
           s.t. Q_hat(task, m) >= Q_hat(task, reference) - delta
```

`delta` is not hand-picked. It's chosen automatically per fold by testing
a grid of values against *measured* held-out outcomes and picking the
largest energy saving that doesn't push the *lower bound* of a bootstrap
confidence interval on quality below −2 percentage points (see §6). This
is the mechanism that turns "10pp fixed threshold, all tasks in a tier or
none" into "however much downgrading a specific deployment's data actually
supports, decided per task."

## 3. Module map

| File | Role |
|---|---|
| `task_features.py` | Per-task features (workload shape), computed offline from `tejas.benchmarking.tasks`/`sembench_tasks` — no re-run needed |
| `deployment_features.py` | Per-deployment metadata (platform, model family, parameter count, GPU count, measurement scope) — hand-authored, grounded in `tejas.measurement.energy_meter`/`gpu_energy_meter` |
| `dataset.py` | Joins the above with `tejas.benchmarking.results_loader.load_raw_results()`'s measured outcomes into one training frame; defines feature groups for ablation |
| `energy_model.py` | 5-member energy prediction ladder (global mean, per-tier lookup, Ridge, ElasticNet, HistGradientBoosting) |
| `quality_model.py` | 3-member quality prediction ladder (per-tier baseline, logistic regression, HistGradientBoosting) with fold-safe calibration |
| `tejas.routing.oracle` | Two oracle definitions computed directly from measured data (no fitting) — the achievable-savings upper bounds the policy is compared against |
| `tejas.routing.policy_optimizer` | The `select_from_rows` function implementing the argmin above |
| `evaluate_models.py` | Repeated nested CV reporting *prediction* quality (does `E_hat`/`Q_hat` predict well?) |
| `tejas.routing.evaluate_policy` | Repeated nested CV reporting *routing* outcomes (does the resulting policy save energy without losing quality?) — this is where `delta` and which ladder member to use get selected |
| `verification_checks.py` | Single-split spot-checks: feature-shuffle leakage test, feature-group ablation |

## 4. Design decisions and rationale

**Why a model ladder instead of one model.** With only 40 tasks (~480
rows including repeats), a single sophisticated model risks overfitting
undetected. The ladder includes trivial baselines (global mean, per-tier
lookup) specifically so that a sophisticated model's apparent win has to
clear a real bar — "beats a lookup table" — not just "produces a number."
In practice, no ladder member wins universally: energy prediction favors
ElasticNet on Mac but HistGradientBoosting on both DGX experiments;
quality mostly favors logistic regression. This is reported as-is rather
than picking one model type "for the paper" — see `docs/RESEARCH_LOG.md` §10.

**Why the energy target is `log(energy_j)`, with a smearing
back-transform.** Energy spans roughly a 3B-to-70B model range and is
heavily right-skewed. Fitting the log and exponentiating back
underpredicts the arithmetic mean (a well-known bias — Duan 1983); the
fix is a training-fold-only correction factor `c = mean(exp(residuals))`,
so `E_hat = c * exp(log_prediction)`. Getting this wrong wouldn't show up
as a crash, just a silent, systematic underestimate of absolute joules —
exactly the kind of bug this project's bug list (`docs/RESEARCH_LOG.md` §5)
is full of, so it's treated with the same seriousness here.

**Why quality is compared as a *difference*, not a raw probability.**
Calibration slopes across all three experiments are ~0.73–0.75, meaning
the quality models are mildly underconfident. `Q_hat(m) - Q_hat(reference)`
partially cancels a systematic bias that a raw `Q_hat(m) >= threshold`
rule wouldn't.

**Why the search space is "any tier ≤ reference," not just one fixed
downgrade.** The prior `scarce` policies only ever considered downgrading
exactly one tier. A hard task's reference tier is "large" — this package
also lets it drop straight to "small" if the quality model says that's
safe, which the old policies structurally couldn't express.

**Why energy-tier ties are broken toward the cheaper option explicitly.**
When the energy model can't discriminate between tiers (the constant
`global_mean` baseline predicts the same value for all three), Python's
`min()` breaks ties by iteration order — which happened to silently make
the naive constant predictor look like it was "choosing" the cheapest
tier, when it was actually just exploiting insertion order. Both
`policy_optimizer.select_from_rows` and `evaluate_policy.py`'s inner
selection now explicitly sort by `(predicted_energy, param_count)` so
this is an intentional, documented rule rather than an accident. (Caught
during this build, not left in — see commit history / `docs/RESEARCH_LOG.md`
§10 for the honest accounting.)

**Why `delta` and the ladder member are selected on *measured* inner
out-of-fold outcomes, not on predicted energy.** An earlier draft of this
design picked `delta` by comparing predicted energy directly — which
never actually validated that the resulting decisions were any good, just
that the energy model believed they were cheap. The corrected version
runs the *actual policy* (with predictions making individual tier
decisions) on data the models were not trained on, and scores it against
what really happened. Predictions decide *which tier*; measured outcomes
decide *which configuration (energy model type × quality model type ×
delta)* was worth using at all.

**Why two oracles.** A per-task, zero-quality-loss oracle (`strict`) and
a global-budget oracle that allows the same aggregate −2pp average loss
the learned policy is held to (`global_budget`, solved exactly via a
small dynamic program) answer different questions. `strict` is the
conservative "how much can we save without ever losing on any single
task" number; `global_budget` is the correct upper bound to compare the
learned policy against, since it's optimizing under the same constraint.
Comparing the policy only to `strict` would understate how much of the
*actually achievable* budget it's capturing.

**Why `StratifiedGroupKFold`, not plain `GroupKFold`, for the outer
split.** 16 of the 40 tasks are `semantic_hard`; a handful of other
`task_type_tier` classes have as few as 3 tasks (`concept_easy`,
`concept_hard`). Plain random grouping could leave an outer fold with
almost none of a workload class, making that fold's metrics meaningless
for that class. Stratifying by `task_type_tier` keeps every fold roughly
representative. (sklearn does warn that it can't stratify perfectly for
the 3-member classes — expected and harmless given the fixed 40-task set,
not something to fix by adding more folds.)

**Why `measurement_scope` blocks pooling by default.** Checked directly
against `tejas.measurement.energy_meter` (Mac: `powermetrics`, whole-SoC
package power — CPU+GPU+ANE combined, since Apple Silicon has no separate
boundary) and `tejas.measurement.gpu_energy_meter` (DGX:
`nvidia-smi --query-gpu=power.draw`, scoped to just the job's allocated
GPU(s), host CPU/RAM explicitly excluded). These are different physical
measurement boundaries, not just different scales — `energy_model.py`
raises rather than silently fitting if asked to pool absolute-joule
targets across rows with different `measurement_scope`.

**Why `scikit-learn`/`pandas` were added as dependencies.** The project
previously had only `matplotlib`/`numpy`. Hand-rolling logistic/linear
regression and cross-validation splitting in raw numpy would add a new
class of bug to a project whose bug list is already full of subtle
correctness issues; using tested, standard implementations was judged the
safer choice for a research artifact whose numbers need to be trusted.

## 5. Underlying assumptions (state these in the paper's limitations)

- **Task set is fixed at 40 tasks × 3 tiers.** All cross-validation and
  bootstrap numbers reflect a small-N regime; per-fold model-selection
  frequencies (`docs/RESEARCH_LOG.md` §10) should be read as "which model
  tended to win," not "which model is definitively best."
- **Deployments are modeled separately, not pooled**, except where
  explicitly noted (cross-deployment transfer). Absolute energy in joules
  is not assumed comparable across `measurement_scope` values.
- **Quality per task is discretized.** With 4 repeated runs, a task's
  measured quality at a given tier is always a multiple of 0.25 (0, .25,
  .5, .75, 1) — this is exploited directly in `tejas.routing.oracle`'s
  dynamic program (exact, not approximate, because the state space is
  small) and shows up as the odd-looking precise numbers (e.g.
  "−0.625pp") in results tables.
- **The router never routes "up.'"** Candidates are restricted to tiers no
  more expensive than the task's reference (static/abundant) tier —
  matches the paper's framing of *scarcity* triggering downgrades, not a
  general model-selection problem.
- **Prompt length is an approximation.** `prompt_tokens` is a regex
  word/punctuation split, not a real per-model tokenizer count. Treat it
  as a proxy for prompt complexity, not a token-budget-accurate number.
- **SQL structural features are lexical proxies, not a parse.**
  `n_joins`, `n_predicates`, etc. are regex counts over the gold SQL
  query text, hand-verified against a handful of tasks
  (`task_features.py`'s `_EXPECTED` table) but not guaranteed correct for
  SQL shapes outside this task set (e.g. commas inside nested function
  calls would miscount projected columns).
- **Task features are treated as deterministic given (task, tier,
  deployment)** — they don't vary by run. The 4 repeats are used to
  estimate a task's quality *distribution* (how often a given tier gets
  it right), not because the features themselves are stochastic.
- **`param_count_b` is the primary cross-model "size" signal**, not
  `model_tier`'s small/medium/large label — deliberate, so a model fit on
  one family's tier boundaries isn't meaningless when applied to another
  family's differently-sized tiers (relevant for the cross-deployment
  transfer work).

## 6. Validation methodology, precisely

- **Outer loop**: 5-fold `StratifiedGroupKFold` (by `task_type_tier`,
  grouped by `task_id`) × 5 repeats with different shuffles = 25 outer
  folds. Every task is held out exactly once per repeat.
- **Inner loop** (within each outer-training split): 4-fold plain
  `GroupKFold`, used for (a) `evaluate_models.py`: picking which ladder
  member to report outer metrics for, by inner out-of-fold MAE (energy)
  or log loss (quality); (b) `tejas.routing.evaluate_policy`: generating
  inner out-of-fold predictions once per ladder-member pair, then
  sweeping `delta` cheaply over those fixed predictions, scoring every
  (energy-model, quality-model, delta) combination against *measured*
  inner-held-out rows, and keeping only combinations whose one-sided
  lower-95%-bootstrap-CI quality difference is ≥ −2pp.
- **An always-reference fallback** (never downgrade, 0% savings, 0
  quality diff) is always a candidate configuration, guaranteeing a
  feasible choice exists even if every learned configuration fails the
  constraint on a given fold.
- **Final reporting**: task-level paired bootstrap (resample `task_id`
  with replacement, 2000 resamples) pooled across all 5 repeats × 40
  tasks, giving 95% CIs on quality difference and % energy saved that
  reflect uncertainty over *which tasks exist*, not just run-to-run
  execution noise (a stronger claim than the 4-run stdev used elsewhere
  in `docs/RESEARCH_LOG.md`).

## 7. Known limitations / future work

Roughly in priority order — see `docs/DESIGN_PLAN.md` for the
full original design spec, including the "Next" and "Final extension"
tiers this session deliberately deferred:

1. **Cross-family transfer** (train on DGX-Qwen, evaluate on DGX-Llama and
   vice versa) — tests whether workload features generalize once model
   identity changes. Specified with two protocols (matched-workload:
   same 40 tasks, different deployment; unseen-workload: task IDs split
   before crossing deployments) so the "did it memorize these specific
   40 tasks" question is answered honestly rather than conflated with
   "does it generalize." Implemented in `cross_family_transfer.py`.
2. **Cross-platform transfer** (Mac ↔ DGX) — blocked on
   `measurement_scope` for absolute joules by design; needs a
   relative/ratio energy target (`E(task,m)/E(task,m_small)`) instead,
   explicitly not the same claim as an absolute cross-platform energy
   model.
3. **Few-shot deployment calibration** — how much data from a *new*
   deployment is needed before the router is useful there? Implemented in
   `fewshot_calibration.py`, with a linear-calibration baseline
   (`logE_target = a + b*logE_hat_source`, fitting only the intercept at
   k=1 sample) to distinguish "new deployment is just a scale shift" from
   "needs genuinely new task–model interaction data."
4. **Margin-based abstention** — implemented in
   `tejas.routing.policy_optimizer` but kept off by default; worth
   evaluating as its own ablation once the core numbers are stable.
5. **Real tokenizers, a real SQL parser (e.g. `sqlglot`), prompt
   embeddings** — all flagged as approximations above; upgrading them is
   lower priority than the transfer/calibration work since none of them
   are suspected of currently being *wrong*, just coarser than they
   could be.
6. **Small-N caveats throughout** — 40 tasks is enough to demonstrate the
   mechanism and clear the "nonzero, non-inferior" bar, but per-fold
   model-selection frequencies and the exact CI widths should be
   expected to move if the task set grows; this is a feasibility result,
   not a claim of a fully mature, deployment-ready cost model.
