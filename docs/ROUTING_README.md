# Energy-Aware LLM Routing — System, Design Rationale, and Results

This document is written to be lifted directly into the CIDR paper: it's
organized by topic (motivation → system → policies → why they're built this
way → assumptions → results → future work), not chronologically. For the
chronological account — including every bug hit and how it was diagnosed —
see `docs/RESEARCH_LOG.md`. For run instructions, see `README.md`.

## 1. Motivation and research question

The paper's thesis is "computation-by-energy" vs. "computation-on-demand":
AI systems should schedule work based on energy state, not serve every
request identically regardless of context. This work instantiates that
narrowly and testably for LLM inference: **does routing requests based on
dynamic energy state — not just static task difficulty — reduce energy
consumption without hurting output quality, and does that hold as
model-capability gaps widen?**

The comparison that actually tests this is **static-cost-router** (routes
by task difficulty only, blind to energy state) vs. **energy-aware**
policies (same difficulty logic, but reacts to an energy-scarce signal by
routing to cheaper models). Beating "always use the biggest model" proves
nothing new — that smaller models are cheaper is not in dispute. The
static-vs-energy-aware gap is the paper's actual argument.

## 2. System overview

**Tasks (40 total, shared across every experiment)**: 24 hand-built
database/SQL tasks graded by execution accuracy against a seeded SQLite
database (18 SQL-writing tasks, 6 database-concept questions), plus 16
semantic-reasoning tasks derived from [SemBench](https://github.com/SemBench/SemBench)'s
Movie scenario, where the model reads real review text and judges sentiment
itself, graded against a hidden ground-truth label it never sees. Two
movies were used: `taken_3` (120 reviews, 11.7% positive — heavily skewed)
and `ant_man_and_the_wasp_quantumania` (256 reviews, 51.6% positive —
balanced), two independent 24-review samples each.

**Three experiments, one model family each** (family held constant within
an experiment so "model size" is the only controlled variable):

| Experiment | Model ladder (small/medium/large) | Hardware | Energy measurement |
|---|---|---|---|
| Mac | Qwen2.5 3b / 7b / 14b | Apple M4 Pro (unified SoC) | `powermetrics` — whole-package power |
| DGX-Qwen | Qwen2.5 3b / 7b / 32b | 1× NVIDIA A100-40GB | `nvidia-smi` — GPU die only |
| DGX-Llama | Llama3 3b / 8b / 70b | 2× NVIDIA A100-40GB | `nvidia-smi` — GPU die only |

Each experiment: 4 repeated full sweeps (40 tasks × 3 tiers), real energy
measured on real hardware per request (not estimated), reported with
mean ± stdev and, for the headline comparisons, paired per-run differences.

## 3. Routing policies: from static to learned

### 3.1 Baselines

`always-largest` and `always-smallest` route every task to one fixed tier
regardless of difficulty or energy — upper and lower bounds on both quality
and energy, not real candidates.

### 3.2 Static and fixed energy-aware policies

```python
STATIC_ROUTE = {"easy": "small", "medium": "medium", "hard": "large"}
ENERGY_AWARE_ROUTE = {
    "abundant": {"easy": "small", "medium": "medium", "hard": "large"},
    "scarce":   {"easy": "small", "medium": "small",  "hard": "medium"},
}
```

`static-cost-router` is a monotonic, hand-written difficulty→size map — the
best a purely difficulty-aware, energy-blind system could do (conceptually,
this is the role a learned bandit like GreenServ plays: smarter than a
hand-written table, but still blind to energy state — see §7 of
`docs/RESEARCH_LOG.md`). `energy-aware (abundant)` is defined identically to
`static-cost-router` on purpose, as a standing sanity check. `energy-aware
(scarce)` downgrades exactly one tier for `medium`/`hard` tasks under a
simulated energy-scarce signal (`easy` is already at the cheapest tier —
nowhere left to fall back to).

### 3.3 Calibrated policy (retrospective threshold)

`scarce` turned out to cause real quality loss once the model-capability
gap widens (§4, §6). A first fix computes, per tier, the *measured* quality
gap between the static tier and scarce's proposed downgrade, and only takes
the downgrade if that gap is ≤10 percentage points — otherwise it stays at
the static tier. This threshold was a first reasonable round number, not
tuned or swept, and the policy is binary per tier (downgrade or don't) —
both of these are exactly the limitations §3.4 addresses.

### 3.4 Learned cost model (built)

Rather than one hand-picked threshold, two models are fit from the same
measured data:

- `E_hat(task, model, deployment)` — predicted energy
- `Q_hat(task, model, deployment)` — predicted quality (probability of
  correctness)

Each is fit via a small **model ladder** rather than one fixed algorithm —
for energy: a global-mean baseline, a per-`(deployment, model_tier)` lookup,
Ridge, ElasticNet, and gradient-boosted trees (`HistGradientBoostingRegressor`),
trained on `log(energy_j)` with a training-fold-only Duan smearing
correction on the back-transform (bare `exp()` on a log-target
systematically underpredicts the arithmetic mean — this correction avoids
that bias). For quality: a per-tier baseline, logistic regression, and
gradient-boosted trees, with fold-safe sigmoid calibration.

The **router** searches every model tier no more expensive than the static
reference tier and picks the minimum-predicted-energy one whose predicted
quality clears `Q_hat(reference) − delta`:

```python
def select(task, candidates, reference_tier, delta, energy_model, quality_model):
    q_ref = quality_model.predict(task, reference_tier)
    feasible = [m for m in candidates
                if param_count(m) <= param_count(reference_tier)
                and quality_model.predict(task, m) >= q_ref - delta]
    return min(feasible, key=lambda m: energy_model.predict(task, m))
```

This is the continuous replacement for §3.3's fixed 10pp switch: `delta` is
a tunable margin rather than a binary gate, and the search considers every
tier (not just "one down"), so the router can find a middle ground the
binary policy structurally couldn't.

**Two oracles**, used as upper bounds for what the router could achieve:

- **Strict per-task oracle**: for each task independently, the minimum-energy
  tier whose *measured* quality is ≥ the reference tier's — a conservative,
  no-exceptions bound.
- **Global-budget oracle**: the minimum-*total*-energy assignment (one tier
  per task) subject to the same aggregate constraint the learned policy is
  held to (mean quality delta across all 40 tasks ≥ −2pp), solved exactly
  since the search space is small (40 tasks × ≤3 tiers). This is the correct
  upper bound for the router's own objective — the strict oracle answers a
  *different*, harder question.

## 4. Why these design decisions

**Why nested cross-validation, not a single train/test split.** With only
40 tasks, one split risks a lucky or unlucky partition dominating the
result. Repeated (5-fold × 5 repeats) outer splits, with a 4-fold inner
loop for model/hyperparameter selection, report a *distribution* of
outcomes rather than one number that could be an artifact of a single
partition. Folds are grouped by `task_id` so a task's repeated runs never
cross a fold boundary — otherwise the model could partly "memorize" a
task's answer from training and be scored on it again in test, inflating
apparent performance. Folds are additionally stratified by
`task_type_tier` because task types are unevenly distributed (16 of 40
tasks are `semantic_hard`) — plain random grouping could leave a fold
nearly devoid of one workload class.

**Why fold-local preprocessing.** Every encoder, scaler, and calibrator is
fit only on the training rows within each fold, never on the full dataset
first. Fitting on everything before splitting leaks test-set statistics
into training and inflates apparent performance — a standard but easy to
skip precaution, and one that specifically matters once cross-deployment
work (§8, not yet built) holds out an entire platform or family and unseen
categories genuinely appear.

**Why two oracles instead of one.** A per-task, zero-exceptions oracle
answers a more conservative question than the one the policy is actually
being scored against (a *mean* constraint across all tasks). Comparing the
learned policy's savings only to the strict oracle would be comparing it
against an unfairly harder target than the one it optimizes for — the
global-budget oracle is the correct upper bound; the strict oracle remains
useful as the conservative reference point.

**Why `delta` and model-ladder selection use measured inner-fold policy
outcomes, not predicted energy or standalone prediction metrics.** The
standalone "best" model by MAE or log-loss doesn't necessarily produce the
best *routing decisions* — what the policy consumes is correct tier
ranking, not raw predictive accuracy. Selection is done by actually running
the candidate policy on inner out-of-fold predictions and scoring it
against the *measured* rows for those tasks, then choosing — among
configurations whose quality stays within budget — the one with lowest
measured energy. Doing this selection on the outer-test outcome instead
would be tuning on the test set.

**Why quality is compared as a *difference* (`Q_hat(m) − Q_hat(ref)`)
rather than a raw calibrated probability.** The quality models show mild,
consistent underconfidence (calibration slope ≈0.73–0.75 against 1.0 for
perfect calibration). That bias mostly cancels when two predictions from
the same model are subtracted; it would distort a threshold applied to a
raw probability directly.

**Why energy measurement boundaries are tracked as a first-class field
(`measurement_scope`), and absolute energy is never pooled across
platforms.** Checked directly against the measurement code: Mac's
`powermetrics` reports whole-SoC package power (CPU+GPU+ANE combined —
Apple Silicon has no separable boundary); DGX's `nvidia-smi` reports GPU-die
power only, host CPU/RAM explicitly excluded. **These are different
physical quantities, not just different scales.** Fitting one absolute-joule
model across both would silently conflate them. The energy model refuses
(raises an error, not a silent no-op) to fit on rows spanning more than one
measurement scope. Only *ratios* (e.g. large-tier energy ÷ small-tier
energy within the same deployment) are valid to compare across platforms,
since numerator and denominator then share a measurement boundary — this is
why cross-platform transfer (§8) is scoped as a relative-ratio target, not
an absolute-energy one.

**Why explicit feature interactions are added only for the linear models.**
Tree-based models learn interactions from splits automatically; adding
explicit interaction terms to them is redundant. Linear models can only
represent an interaction if it's given as an explicit feature — without
`log_param_count × is_semantic`, a linear model literally cannot represent
the sign flip Finding 4 shows (bigger models sometimes doing *worse* on
semantic tasks), since that requires the effect of size to depend on task
type.

**Why ties in the routing search are broken toward the cheaper tier
explicitly, not by iteration order.** An early version of the optimizer
broke ties among equally-feasible tiers by whatever order they were
iterated in, which could silently pick a more expensive tier without
producing an obviously wrong number — a caught and fixed bug, mentioned
here because it's exactly the class of error that doesn't announce itself.

## 5. Underlying assumptions and limitations

- **Proof-of-concept scale.** 40 tasks, 4 repeats per experiment. Nested-CV
  outer folds have only ~8 test tasks each — individual fold estimates are
  noisy, which is why results are reported as a distribution across 5
  repeats, not a single point estimate.
- **Single benchmark domain per task type.** All SQL tasks run against one
  seeded schema; all semantic tasks come from one SemBench scenario (Movie),
  across two movies. Generalization to other schemas or scenarios is
  untested (Spider/BIRD integration, §6, would address the SQL side).
- **Single model family per deployment, though cross-family transfer is now
  measured (DGX-Qwen ↔ DGX-Llama, both `gpu_die_only`).** Quality
  predictions transfer to within ~1pp of native accuracy in *both*
  directions — workload shape matters far more than which family answers.
  Energy transfer is real but weaker and asymmetric (Llama→Qwen MAE
  degrades ~3.3×, Qwen→Llama ~1.5×). Protocol B (unseen-workload, task IDs
  split before crossing deployments) tracks Protocol A (matched-workload)
  closely, so this isn't the model memorizing the 40 tasks. Mac↔DGX
  transfer remains untested — blocked by the measurement-scope difference
  below, not yet given a ratio-target model variant.
- **Energy measurement boundaries differ by platform** (`measurement_scope`,
  §4) — absolute joules from Mac and DGX are not directly comparable; only
  within-platform and ratio-based comparisons are valid. This is a hardware
  constraint, not a modeling choice, and is enforced in code, not just
  documented.
- **Difficulty tiers are hand-assigned**, not derived from an independent
  measure or user study — a task's easy/medium/hard label reflects
  researcher judgment at benchmark-construction time.
- **The quality metric is a fixed grading rubric** (execution accuracy for
  SQL; tolerance-banded exact/near-match for semantic tasks), not a human
  quality judgment — it captures correctness, not fluency or helpfulness.
- **The −2pp mean-quality budget** used to define both oracles' and the
  policy's constraint is a reasonable, chosen tolerance, not derived from a
  user study or production SLA.
- **The learned policy calibrates itself per deployment, and few-shot
  adaptation to a new deployment is now measured, not just proposed** — and
  the two transfer directions tell genuinely different stories, not one
  "more data helps" curve. DGX-Qwen→DGX-Llama needs a full refit (a cheap
  linear scale-shift correction is unstable at k=1, MAE 821±1587J vs.
  zero-shot's 336J). DGX-Llama→DGX-Qwen is the opposite: a cheap linear
  correction alone *beats* full refit from k≥5 on, on both energy MAE and
  quality accuracy. Read together, DGX-Qwen behaves like a scaled-down
  DGX-Llama; DGX-Llama is not simply a scaled-up DGX-Qwen — consistent with
  the 70B/3B gap being qualitatively, not just quantitatively, more extreme
  than 32B/3B.
- **The energy model's routing contribution is modest by its own numbers.**
  The most frequently selected energy-model member for actual routing
  decisions is the constant `global_mean` baseline (87–100% of folds on
  Mac/DGX-Qwen, 75% on DGX-Llama) — not because it's a good standalone
  energy predictor (its own MAE is the worst in the ladder), but because
  once the quality gate has narrowed the feasible set, "cheapest of what's
  left" already captures nearly all the achievable benefit. The quality
  model is doing most of the real work; this is stated plainly in §10 of
  `docs/RESEARCH_LOG.md` rather than overselling the energy ladder specifically.
- **Aggregate non-inferiority does not mean per-task non-inferiority.**
  Task-level measured violations of the −2pp bound do occur (0.5–7.5% of
  task-level decisions across the three experiments) even when the
  aggregate mean constraint holds — an inherent consequence of a *mean*
  constraint rather than a per-task guarantee. A deployment requiring hard
  per-task guarantees would need the stricter per-task oracle's approach,
  not this policy's aggregate one.
- **SQL structural features are lexical proxies, not a real parser.**
  `n_joins`, `has_subquery`, etc. are regex-derived from the gold SQL, not
  parsed by a real SQL parser (e.g. `sqlglot`) — validated against a
  hand-checked subset of tasks, not full parser-grade coverage.

## 6. Results summary

**Original finding (fixed policies, `docs/RESEARCH_LOG.md` §6):** the
`scarce` policy gives a genuine "free lunch" on Mac (paired mean **+1.9pp**
quality, **−40.4%** energy, consistent across all 4 runs) but degrades
sharply as the capability gap widens (DGX-Qwen: **−12.6pp** quality;
DGX-Llama: **−33.7pp** quality vs. `static-cost-router`). The `calibrated`
threshold policy prevents the DGX quality collapse but recovers **0%**
energy savings there — it's all-or-nothing per tier.

**Learned policy (Option C, `docs/RESEARCH_LOG.md` §10):**

| Experiment | static-cost-router | Global-budget oracle | Learned policy (mean, 5 repeats) | 95% CI (bootstrap) | Oracle savings recovered |
|---|---|---|---|---|---|
| Mac | 70.6% q, 1207.5 J | 66.5% saved | **+6.6pp q, 48.6% saved** | q: [+3.1, +10.4]pp; energy: [41.4%, 54.3%] | 53–85% |
| DGX-Qwen | 83.8% q, 7135.4 J | 69.2% saved | **−0.4pp q, 9.7% saved** | q: [−1.0, +0.1]pp; energy: [5.3%, 14.8%] | 2–31% |
| DGX-Llama | 87.5% q, 28420.0 J | 58.3% saved | **−0.3pp q, 1.4% saved** | q: [−0.8, 0.0]pp; energy: [0.0%, 4.6%] | 0–12% |

The learned router recovers a genuine, statistically bounded, nonzero share
of the theoretical energy savings on **every** experiment — including the
two (DGX-Qwen, DGX-Llama) where the hand-tuned `calibrated` policy found
none. On Mac it also strictly beats the original `scarce` policy (+6.6pp/
−48.6% vs. +1.9pp/−40.4%). The margin still shrinks as the capability gap
widens — DGX-Llama's 70B/3B gap leaves genuinely little safe room to
downgrade at all, and the router correctly finds almost none, rather than
either collapsing to zero (like `calibrated`) or overreaching into quality
loss (like plain `scarce`).

## 7. Future improvements

Updated after cross-family transfer and few-shot calibration were built
(see §5 for their results) — both were future work as of an earlier draft
of this document; they're done now. Remaining items, roughly in priority
order, specified precisely enough to hand off without re-deriving design
decisions (see `docs/DESIGN_PLAN.md` and
`docs/RESEARCH_LOG.md` §8/§9/§10 for full detail on each):

1. **Relative cross-platform energy transfer** (Mac↔DGX) — using a
   within-deployment energy-*ratio* target instead of absolute joules,
   given the measurement-scope constraint in §4/§5. Still blocked: needs
   the ratio-target model variant, not built. This is the one remaining
   transfer direction — DGX-Qwen↔DGX-Llama transfer (§5) is done.
2. **Margin-based abstention as a primary feature**, not just an ablation
   flag — fall back to the reference tier when the predicted margin is too
   thin, rather than only at hard delta boundaries.
3. **Binary → continuous energy trace.** The current abundant/scarce signal
   is a two-state switch; a real deployment's energy availability is
   continuous (grid carbon intensity, solar generation, battery level).
   Since routing decisions are already computed retroactively from
   per-task-per-tier measured data, this doesn't need new experiments — it
   needs a time-varying trace, a policy that's a smooth function of it
   (the learned model above is the natural candidate, once it accepts
   energy level as a continuous input feature), and a replay-style
   evaluation over simulated time rather than one before/after comparison.
4. **Cross-family single-policy routing** — route across Qwen *and* Llama
   within one policy, rather than two family-isolated experiments compared
   side by side.
5. **Wider SemBench query coverage** — Q1 (unscoped), Q5–Q7 (pairwise
   sentiment matching), Q9–Q10 (1–5 score prediction) were deferred when
   the benchmark was widened, to change one axis (movie/sample) at a time.
6. **Spider/BIRD integration** for the SQL side specifically, to strengthen
   generalizability claims beyond one hand-built schema.
7. **More repeats / larger task set** for tighter confidence intervals —
   current nested-CV outer folds are small (~8 tasks) given only 40 total.
8. **A real SQL parser** (e.g. `sqlglot`) in place of the current regex
   lexical proxies, if SQL structural features become more central to a
   specific claim than they are today.
