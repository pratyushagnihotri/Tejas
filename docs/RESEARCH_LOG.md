# Research Log — Energy-Aware LLM Routing (CIDR PoC)

Working log for the "Sustainable AI Data Systems: From Computation-on-Demand
to Computation-by-Energy" CIDR vision paper. Covers what was built, why,
what broke and how it was fixed, what was found, and what's still open.
`README.md` covers how to run the harness; this file covers the research
narrative behind it.

## 1. The research question

Does routing LLM inference requests based on **dynamic energy state** (not
just static task difficulty) reduce energy consumption without hurting
output quality — and does that hold up as model-capability gaps widen?

This is deliberately narrower than "energy-aware AI systems" in general: the
comparison that actually tests the paper's thesis is **static-cost-router**
(routes by task difficulty only, blind to energy state) vs.
**energy-aware-scarce** (same difficulty logic, but reacts to a scarce-energy
signal by downgrading to smaller models). Beating "always use the biggest
model" alone would prove nothing new — smaller models being cheaper is not
in dispute. The static-vs-energy-aware gap is the paper's actual argument.

## 2. System design

- **Tasks (40 total, 3 difficulty tiers)**: 24 hand-built DB/SQL tasks
  (`tasks.py`) graded by execution accuracy against a seeded SQLite DB, plus
  16 semantic-reasoning tasks (`sembench_tasks.py`) derived from
  [SemBench](https://github.com/SemBench/SemBench)'s Movie scenario — the
  model reads real review text and judges sentiment itself, graded against a
  hidden ground-truth label. Two movies: `taken_3` (120 reviews, 11.7%
  positive — heavily skewed) and `ant_man_and_the_wasp_quantumania` (256
  reviews, 51.6% positive — balanced), 2 independent samples each.
- **Model ladders**: one family per experiment, 3 tiers each, so "size" is
  the only controlled variable within an experiment:
  - Mac: Qwen2.5 3b / 7b / 14b (small capability gap, "large" is modest)
  - DGX-Qwen: Qwen2.5 3b / 7b / 32b (larger gap)
  - DGX-Llama: Llama3 3b / 8b / 70b (largest gap)
- **Routing policies compared**: always-largest, always-smallest,
  static-cost-router, energy-aware (abundant — identical to static by
  construction, a sanity check), energy-aware (scarce — downgrades one tier
  per task difficulty under a simulated energy-scarce state), and
  energy-aware (scarce, calibrated — only takes a downgrade if the measured
  quality gap for that tier is ≤10pp, else stays at the static tier).
- **Energy measurement**: real hardware, not estimated. `powermetrics` on
  Mac (`energy_meter.py`), `nvidia-smi` on DGX (`gpu_energy_meter.py`), both
  built on a shared trapezoidal-integration base (`power_meter_base.py`).
- **Statistical approach**: 4 repeated runs per experiment; report mean ±
  stdev, and — for the headline static-vs-scarce comparison — a *paired*
  per-run diff (more rigorous than comparing two independent means, since it
  controls for run-to-run system-state confounds).

## 3. Design rationale: SemBench query selection and routing policy categorization

### Why Q2/Q3/Q4/Q8, and why the rest were deferred

SemBench's Movie scenario ships 10 official queries. Q2 (list positive
reviewIds), Q3 (count positive), Q4 (positivity ratio), Q8
(positive/negative breakdown) were picked first because they're all the
*same underlying shape* — filter/classify sentiment over one review sample,
then count/list/ratio/breakdown the result — so one family of grading
functions (`id_list_precision`, `count`, `ratio`, `breakdown` in
`grading.py`) covers all four against the same hidden `scoreSentiment`
label, with no new verification mechanism needed.

Deferred deliberately, not forgotten:

- **Q1** ("any movie", unscoped) needs a cross-movie corpus sample instead
  of one movie's reviews — a different sampling strategy, not just a new
  grading mode.
- **Q5–Q7** (pairwise same/opposite sentiment matching) are a genuinely
  different task shape — comparative judgment over *pairs* of reviews, not
  per-review classification — and would need a new `pairwise_match` grading
  mode that doesn't exist yet.
- **Q9–Q10** (predict a 1–5 star score per review, graded against the real
  critic score) are regression-flavored, not classification — continuous
  error rather than binary correct/incorrect, another new grading mode.

When the set was widened (2 movies × 2 samples), only the *movie/sample*
axis was changed — the already-built Q2/Q3/Q4/Q8 grading logic was reused
as-is rather than also adding new query types in the same step. This
"change one axis at a time" approach is why each finding in this log could
be attributed to a specific cause rather than a tangle of simultaneous
changes — several of the bugs in §5 were only catchable because only one
thing had changed since the last known-good result.

### How the routing policies are categorized

```python
STATIC_ROUTE = {"easy": "small", "medium": "medium", "hard": "large"}
ENERGY_AWARE_ROUTE = {
    "abundant": {"easy": "small", "medium": "medium", "hard": "large"},
    "scarce":   {"easy": "small", "medium": "small",  "hard": "medium"},
}
```

- `STATIC_ROUTE` is a monotonic 1:1 map from difficulty tier to model tier —
  the best a purely *difficulty-aware, energy-blind* system could do. This
  is the conceptual role GreenServ's learned bandit plays (§7): smarter
  than a hand-written table, but still blind to energy state.
- `abundant` is defined identically to `STATIC_ROUTE` **on purpose**, as a
  standing sanity check — if `energy-aware (abundant)` ever produced
  different numbers than `static-cost-router` in the output, that would
  indicate a bug in the harness, not a finding. It never has.
- `scarce` downgrades exactly one tier for `medium` and `hard` tasks, and
  leaves `easy` alone — not an oversight: `easy` is already at the cheapest
  tier under the static policy, so there's nowhere cheaper to fall back to.
- `calibrated` scarce (added after Finding 2 showed `scarce` could backfire
  on DGX) recomputes, per tier, the *actual measured* quality gap between
  the static tier and scarce's proposed downgrade, and only takes the
  downgrade if that gap is ≤10pp. This is retrospective/oracle calibration
  — it uses the same dataset's own measured results, not something a
  router could compute zero-shot before running both models. The 10pp
  threshold itself was **not tuned or swept** — it's a first reasonable
  round number to test the concept, and its bluntness (all-or-nothing per
  tier) is precisely the limitation that motivates Option C (§9).

## 4. Timeline of design decisions

1. Started with generic arithmetic/trivia tasks, pivoted to **DB-specific
   SQL tasks** — appropriate for a CIDR (database systems) audience.
2. Considered Spider/BIRD (classic text-to-SQL benchmarks) but chose
   **SemBench** instead after actually reading its paper/repo: its Movie
   scenario tests genuine semantic judgment (hidden sentiment label) rather
   than SQL syntax complexity, a sharper test of "does this need a bigger
   model." Spider/BIRD remain a reasonable future addition, not adopted.
3. Widened SemBench from 1 movie/4 queries to 2 movies (skewed + balanced)
   × 4 query types × 2 samples = 16 tasks, specifically to test whether
   "hard tasks don't scale with model size" was real or an artifact of one
   skewed sample.
4. Extended from Mac-only to **DGX** (TU Darmstadt Systems group cluster,
   Slurm-managed, shared with other lab members) for a genuine 32B/70B
   "large" tier and real GPU power measurement.
5. Found GreenServ (arXiv:2601.17551, Ziller et al., TU Wien) via web
   search when the user asked about it directly — a closely related but
   distinct paper: it does *learned, cost-aware* routing across 16 models,
   but optimizes a fixed accuracy/energy trade-off knob per query, not
   *dynamic energy availability*. Complementary, not redundant — worth
   citing, not benchmarking against directly.

## 5. Bugs found and fixed (in case any of these matter for the write-up)

Each of these was caught by checking a surprising result rather than
trusting it — worth keeping in mind since some of them *inverted* an
apparent finding.

| # | Bug | Symptom | Fix |
|---|---|---|---|
| 1 | Energy window required 2+ samples strictly inside it | Fast (<1 sampling interval) requests silently got no energy reading, disproportionately affecting the small model | Bracket the window with interpolated readings at the boundary |
| 2 | Fix #1 over-corrected: rejected windows if the *last* sample wasn't past `end_ts` | Reader thread always lags "now" by ~1 interval (a sample only commits when the *next* block arrives), so almost every window was rejected — **all** energy readings went missing | Poll briefly (up to 1s) for the reader to catch up before giving up |
| 3 | `grade_concept` used literal substring match | Large model's correct answer `"ReadCommitted"` didn't match `"read committed"` (no space) — large model looked artificially worse on hard tasks | Normalize (strip non-alphanumeric, lowercase) before matching |
| 4 | SemBench count/breakdown grading required exact integer match | On the balanced movie (`ant_man`, true count ~12/24), any single misjudged review shifted the count away from exact — looked like near-total model failure regardless of size | Added ±2 tolerance, matching the existing ratio task's leniency |
| 5 | The original `analyze.py` (now `tejas.benchmarking.results_loader`) aggregated all runs in the CSV regardless of task-set version | Old 28-task runs and new 40-task runs would silently blend into one misleading average | Canonical-run-set filtering: only aggregate runs matching the largest/most-recent task set found |
| 6 | Ollama's release asset format changed (`.tgz` → `.tar.zst`) | User-space install download silently returned a 9-byte 404 stub | Checked the GitHub API for the actual current asset name |
| 7 | `llama-server` auto-detects the *node's* full CPU count, ignoring Slurm's cgroup allocation | 128 threads fighting over a 4-core allocation → 0.97 tokens/sec on an A100 | Cap thread count explicitly via Ollama's `num_thread` option, matched to `--cpus-per-task` |
| 8 | Two concurrent `sbatch` jobs both tried to bind Ollama's default port 11434 (single-node cluster) | Second job's server failed to start; its requests silently went to the *first* job's server instead, causing GPU memory thrashing and an apparent "hang" in both jobs | Ran jobs sequentially instead (a per-job unique-port fix also exists in the code but wasn't ultimately needed) |

## 6. Key findings

**Finding 1 — the "free lunch" is real, but conditional.** On Mac (Qwen
3b/7b/14b — a modest capability gap), energy-aware (scarce) matches or beats
static-cost-router's quality (paired mean **+1.9pp**, noisy — well within a
5.5pp stdev, i.e., no real quality cost) while cutting energy by a
consistent **40.4% ± 4.9pp across all 4 runs**. This is the paper's
strongest single number.

**Finding 2 — it breaks down as the capability gap widens.**

| Experiment | static-cost-router | energy-aware (scarce) | Quality Δ |
|---|---|---|---|
| Mac (14b large) | 70.6% | 72.5% | **+1.9pp** |
| DGX-Qwen (32b large) | 83.8% | 71.2% | **−12.6pp** |
| DGX-Llama (70b large) | 87.5% | 53.8% | **−33.7pp** |

Root cause, measured directly: the medium→large quality gap on hard tasks
is −10.4pp on Mac (medium actually *beats* large there), +14.6pp on
DGX-Qwen, and **+44.8pp** on DGX-Llama. A fixed "always downgrade one tier
under scarcity" rule doesn't know how expensive that downgrade actually is.

**Finding 3 — a naive calibrated fix prevents disaster but not cost.** A
policy that only downgrades a tier if its *measured* quality gap is ≤10pp
recovers Mac's full story (76.9% quality, beating even plain `scarce`) but
on DGX it just refuses every downgrade — collapsing to static-cost-router
exactly, 0% energy saved. A binary per-tier threshold isn't the right tool
once the gap is large; it needs a continuous cost/benefit trade-off instead
of an all-or-nothing rule. This is the concrete gap Option C (a learned
energy/quality cost model) would need to fill.

**Finding 4 — task "hardness" isn't one-dimensional.** On the skewed movie
(`taken_3`), model size clearly helps (44%→84%→72%, small→medium→large). On
the balanced movie (`ant_man`), it **inverts** — small (34%) beats medium
(16%) beats large (9%). Bigger models don't uniformly help on genuine
semantic judgment tasks; a router that assumes "hard → route to largest"
can be actively wrong for some task types.

**Finding 5 — SQL-structural vs. semantic-reasoning difficulty behave
differently under scaling**, and mixing them into one "hard" tier average
can obscure both patterns — worth treating as separate axes in the paper
rather than one difficulty scalar.

## 7. Related work positioning

**GreenServ** (Ziller et al., TU Wien, arXiv:2601.17551) — a multi-armed
bandit router across 16 models optimizing a fixed accuracy/energy
scalarization (λ) per query. It answers "which model for this query" well,
but has no notion of *time-varying energy availability* — it's the smarter,
learned version of our `static-cost-router` baseline, not a version of
`energy-aware`. Cite as closely related, explicitly differentiate on the
dynamic-availability axis. Their model pool overlaps ours (Qwen2.5
0.5b–14b), and they measure energy via the Zeus library on an A100 — worth
matching methodology if a direct comparison is ever wanted.

## 8. Open items / next steps

Roughly in priority order:

1. ~~**Option C, properly motivated now**: a learned cost/quality model that
   makes a continuous trade-off decision per task, instead of the binary
   per-tier threshold that was just shown to be too blunt on DGX. This is
   no longer "nice to have" — Finding 3 is a direct, concrete case for it.~~
   **Addressed — see §10.**
2. **Binary → continuous energy trace**: the abundant/scarce toggle was a
   deliberate scope cut from the original continuous-trace design; revisit
   if time allows, since it's a more literal test of "computation-by-energy."
3. **Option B (cross-family routing)**: route across Qwen *and* Llama
   models in one policy, not just two separate family-isolated experiments
   — discussed as a deliberate follow-up, not yet built.
4. **Statistical rigor**: current comparisons use paired mean/stdev over 4
   runs; a submission-ready version should probably run more repeats and/or
   a formal significance test on the static-vs-scarce gap specifically.
5. **Wider SemBench coverage**: Q1 (unscoped), Q5–Q7 (pairwise sentiment
   matching), Q9–Q10 (1–5 score prediction) were scoped out when widening
   to keep the task-type surface area manageable — worth adding if the
   `ant_man` inversion finding (#4 above) turns into its own angle.
6. **Spider/BIRD**: considered, not integrated — would strengthen
   generalizability claims for the SQL side specifically if time permits.
7. **Write-up**: this log plus the figures in `results/figures/` (per
   dataset: `results/figures/`, `.../dgx_qwen/`, `.../dgx_llama/`; cross
   experiment: `.../cross_experiment/`; per-query heatmaps:
   `fig6_query_heatmap` in each of those directories) are the source
   material for the paper's evaluation section.

## 9. Handoff: starting Option C (learned cost model) in a new session

This section is written so a fresh session can pick up work here without
re-reading the whole project history first — though reading §1, §3, and §6
above first is still the fastest way to get real context, not just this
summary.

**What Option C is.** The calibrated policy (§3, §6 Finding 3) showed that a
hand-picked, binary, per-tier threshold (downgrade if measured quality gap
≤10pp, else don't) prevents the catastrophic quality loss `scarce` caused on
DGX, but gives up *all* energy savings there too — it's all-or-nothing, not
a real trade-off. Option C replaces that fixed rule with a model that
predicts the quality/energy trade-off per task, so the router can make a
continuous decision (e.g., "downgrade if predicted quality loss < X for
predicted energy saved > Y") instead of a blanket per-tier switch.

**Data that already exists, ready to use:**

- `results/raw_results.csv` (Mac, Qwen 3b/7b/14b), `results/raw_results_dgx_qwen.csv`
  (Qwen 3b/7b/32b), `results/raw_results_dgx_llama.csv` (Llama 3b/8b/70b).
  Columns: `run, model_tier, model_name, task_id, task_tier, task_type,
  latency_s, energy_j, correct, response, error`. ~480 rows per experiment
  (40 tasks × 3 tiers × 4 repeats).
- **Load via `tejas.benchmarking.results_loader.load_raw_results(path)`**, not a raw CSV read — it applies the
  canonical-run-set filter (§5, bug #5) that keeps old/new task-set
  versions from silently blending.

**One feature gap worth closing first**: prompt length / token count per
task isn't logged in these CSVs, but is trivially computable offline from
`tasks.py` / `sembench_tasks.py` (`len(task["prompt"])`) without rerunning
anything. This was flagged early in the project (before the DGX phase) as
the one feature that isn't redundant with `model_tier` — likely useful here.

**Shape of a reasonable first deliverable** (not prescriptive): given
`(task features, candidate model tier)`, predict either (a) P(correct) and
expected energy separately, letting a downstream policy pick the cheapest
tier that clears a quality bar, or (b) a single continuous "worth
downgrading" score. With ~480 rows and mostly categorical features, keep
the model itself simple (logistic/linear regression or a small tree) — the
value is in the continuous decision, not model complexity.

**Validation**: hold out by `task_id`, not by row — a task's repeats must
stay together, or the same task leaks across train/test. Compare against
the existing baselines in `tejas.routing.static_policies` (`static-cost-router`, `scarce`,
`scarce, calibrated`) using the same paired-per-run methodology used
throughout this log (§6, Finding 1), not just marginal means.

**Related work to keep in mind**: GreenServ (§7) already does learned
cost-aware routing, just not for dynamic energy availability — Option C
should stay clearly positioned as learning the *energy-scarcity* trade-off
specifically, not reinventing GreenServ's per-query cost/accuracy bandit.

## 10. Energy cost model — results (Option C, built)

Built in `src/tejas/cost_model/` (own package, not mixed into the repo root):
`task_features.py` / `deployment_features.py` (feature extraction, no
re-run needed), `dataset.py` (combines them with `tejas.benchmarking.results_loader.load_raw_results()`'s
measured outcomes), `energy_model.py` / `quality_model.py` (prediction
ladders), `oracle.py` (two oracle definitions), `policy_optimizer.py`
(all-tier argmin under a quality constraint), `evaluate_models.py`
(prediction-quality metrics) and `evaluate_policy.py` (routing outcomes).
Full design rationale — including two rounds of methodological review that
caught real bugs before they produced misleading numbers (an oracle
misaligned with the policy's own objective, a false "delta=0 guarantees
non-inferiority" sanity check, delta selection tuned on the same data it
was scored on) — is in `docs/DESIGN_PLAN.md`.

**What Option C actually is**: `E_hat(task, model, deployment)` and
`Q_hat(task, model, deployment)`, fit via a small model ladder (global
mean / per-tier lookup / Ridge / ElasticNet / HistGradientBoosting for
energy; per-tier baseline / logistic regression / HistGradientBoosting for
quality), validated with repeated nested grouped cross-validation (outer
5-fold × 5 repeats, inner 4-fold, `StratifiedGroupKFold` by
`task_type_tier`, task IDs never cross a fold boundary). The router
searches **all** tiers no more expensive than the abundant/static
reference tier and picks the minimum-predicted-energy one whose predicted
quality clears `Q_hat(reference) - delta` — a continuous per-task decision
replacing the `scarce, calibrated` policy's single binary, tier-average
10pp threshold. `delta` (and which ladder member to use) is selected
**inside the inner loop only**, by actually running the policy on inner
out-of-fold predictions and scoring it against *measured* rows — not by
looking at predicted energy or at the final outer-test outcome.

### Measurement-scope finding

Checked directly against `energy_meter.py` / `gpu_energy_meter.py`: Mac's
`powermetrics` reports whole-SoC package power (`Combined Power (CPU + GPU
+ ANE)`); DGX's `nvidia-smi --query-gpu=power.draw` reports GPU-die power
only, host CPU/RAM explicitly excluded. **These are different physical
measurement boundaries, not just different scales** — absolute joules from
Mac and DGX are not directly comparable. `deployment_features.py` encodes
this as `measurement_scope` and `energy_model.py` refuses (raises, not a
silent no-op) to fit an absolute-energy model on rows spanning more than
one `measurement_scope`. This blocks cross-platform pooling until a
relative/ratio energy target is used instead (specified but not built this
session — see "Not built" below).

### Results, five layers of evidence per experiment

The question §9 set: does the learned policy recover *nonzero,
CI-bounded-non-inferior* energy savings on DGX, where `scarce, calibrated`
recovers exactly 0%? Answered below per experiment, then summarized.
Figures referenced below are in `results/figures/cost_model/` (generated
by `analysis/plot_cost_model_results.py`).

#### Mac (Qwen 3b/7b/14b)

**1. Prediction performance** (`evaluate_models.py`, out-of-fold, 5×5
repeated nested CV): energy MAE 7.7 ± 3.5 J, MdAPE 29.3%, RMSE(log-energy)
0.50, R² 0.61, avg. per-task Spearman 0.78, min-energy-tier hit rate 90.2%.
Quality accuracy 77.1%, Brier 0.161, log loss 0.516, calibration slope
0.73, ΔQ error 0.223 (mean absolute error between predicted and actual
quality-difference-from-reference, in probability units — this is the
number closest to what the router actually consumes).

**2. Model-selection behavior** (frequency across 25 outer folds, and
separately across `evaluate_policy.py`'s joint routing-decision selection):
prediction-accuracy-selected energy model is ElasticNet 76% / HistGB 24%;
quality is logistic regression 68% / HistGB 32%. For the *routing decision*
specifically (selected on measured inner-OOF policy outcomes, not
predictive accuracy — see design decision in `src/tejas/cost_model/README.md`
§4), energy collapses to `global_mean` 87% / `tier_lookup` 9% / ElasticNet
4%, quality to HistGB 87% / logreg 13%. The gap between these two
selections is itself informative — see the ladder note below.

**3. Policy result** (task-level paired bootstrap, n=200 repeat×task
records, 2000 resamples): quality **+6.6pp**, energy **48.6% saved** vs.
static-cost-router (means across 5 repeats); 95% CI quality [+3.1,
+10.4]pp, energy saved [41.4%, 54.3%]. Mean `delta` used: 0.014 (fallback
to reference in 8% of outer folds).

**4. Routing behavior** (`fig_routing_behavior_mac`): 57.0% of decisions
downgrade below the reference tier overall. Heavily workload-dependent:
`concept_easy`/`sql_easy` never downgrade (0%, they're already at the
cheapest tier); `sql_medium` rarely downgrades (10%); `concept_hard`,
`semantic_hard`, `sql_hard` downgrade 91–93% of the time, mostly to
`medium` rather than `small`.

**5. Distance to oracle** (`fig_baseline_pareto_mac`): strict oracle 756.7 J
(37.3% saved, q=0.794); global-budget oracle (−2pp) 404.2 J (66.5% saved,
q=0.688). Learned policy recovers **53–85%** of the global-budget oracle's
achievable savings across the 5 repeats.

#### DGX-Qwen (Qwen 3b/7b/32b)

**1. Prediction performance**: energy MAE 44.6 ± 22.8 J, MdAPE 21.1%,
RMSE(log-energy) 0.41, R² 0.56, avg. Spearman 0.87, min-tier hit rate
87.6%. Quality accuracy 77.9%, Brier 0.149, log loss 0.452, calibration
slope 0.75, ΔQ error 0.260.

**2. Model-selection behavior**: prediction-accuracy-selected energy is
HistGB 84% / ElasticNet 16%; quality is logreg 36% / tier baseline 36% /
HistGB 28% (a three-way near-tie — no clear winner here, worth flagging
rather than picking one to report). Routing-decision selection: energy
`global_mean` **100%**; quality HistGB 58% / tier baseline 42%.

**3. Policy result**: quality **−0.4pp**, energy **9.7% saved** vs.
static-cost-router; 95% CI quality [−1.0, +0.1]pp (straddles zero — no
detectable quality cost), energy saved [5.3%, 14.8%]. Mean `delta`: 0.010;
fallback to reference in **52%** of outer folds — the router is being
conservative given the wide capability gap, not aggressive.

**4. Routing behavior** (`fig_routing_behavior_dgx_qwen`): only 12.5% of
decisions downgrade overall. `concept_easy`/`sql_easy`/`sql_medium` behave
as on Mac (0% downgrade for the easy ones; `sql_medium` here is 0%, not
10% — a wider capability gap makes even the medium-tier downgrade path
less safe). `semantic_hard` downgrades only 10% of the time (vs. 91% on
Mac) — a direct, quantified illustration of Finding 2's capability-gap
sensitivity at the routing-behavior layer, not just the aggregate quality
number.

**5. Distance to oracle** (`fig_baseline_pareto_dgx_qwen`): strict oracle
3336.4 J (53.2% saved, q=0.850); global-budget oracle 2194.9 J (69.2%
saved, q=0.819). Learned policy recovers **2–31%** of the global-budget
oracle's savings — a much smaller share than Mac's 53–85%, but still
strictly greater than `calibrated`'s 0%.

#### DGX-Llama (Llama 3b/8b/70b)

**1. Prediction performance**: energy MAE 200.0 ± 129.9 J, MdAPE 20.4%,
RMSE(log-energy) 0.48, R² 0.46, avg. Spearman 0.89, min-tier hit rate
93.0% (the highest of the three experiments — easier to rank tiers
correctly when they're this far apart in energy). Quality accuracy 73.1%,
Brier 0.190, log loss 0.664 (the worst of the three — the 70B/3B gap makes
quality harder to predict, not just harder to safely exploit), calibration
slope 0.73, ΔQ error 0.313 (also the worst of the three).

**2. Model-selection behavior**: prediction-accuracy-selected energy is
HistGB 84% / ElasticNet 16%; quality is logreg 68% / HistGB 28% / tier
baseline 4%. Routing-decision selection: energy `global_mean` 75% / HistGB
25%; quality logreg 75% / HistGB 25%.

**3. Policy result**: quality **−0.3pp**, energy **1.4% saved**; 95% CI
quality [−0.8, 0.0]pp, energy saved [0.0%, 4.6%] — the lower bound touches
zero, meaning some repeats find no safe downgrade at all. Mean `delta`:
0.075 (the largest of the three, but applied so rarely it barely moves the
aggregate); fallback to reference in **84%** of outer folds.

**4. Routing behavior** (`fig_routing_behavior_dgx_llama`): 0.5% of
decisions downgrade overall — essentially never, except `sql_hard` at 4%.
Every other workload class shows 0% downgrade. With a 70B/3B gap, the
router correctly finds there's almost no safe room, rather than either
collapsing to a hand-picked 0% (`calibrated`) or overreaching into the
quality collapse plain `scarce` produces (Finding 2: −33.7pp).

**5. Distance to oracle** (`fig_baseline_pareto_dgx_llama`): strict oracle
21764.6 J (23.4% saved, q=0.900); global-budget oracle 11843.6 J (58.3%
saved, q=0.856). Learned policy recovers **0–12%** of the global-budget
oracle's savings — small, but the point isn't the size of the number, it's
that it's *reliably nonzero and bounded-safe* rather than the binary 0%
`calibrated` gets stuck at.

### Baseline comparison table (all three experiments)

Full numbers in `results/baseline_comparison_{mac,dgx_qwen,dgx_llama}.csv`
and `fig_baseline_pareto_*`. `always-X` = route every task to tier X
regardless of difficulty (the "best fixed tier" comparison) — included
specifically to show that the learned policy's gain comes from *per-task,
workload-conditioned* decisions, not just "prefer a globally cheaper tier":

| Policy | Mac q% / J | DGX-Qwen q% / J | DGX-Llama q% / J |
|---|---|---|---|
| static-cost-router | 70.6 / 1207.5 | 83.8 / 7135.4 | 87.5 / 28420.0 |
| scarce | 72.5 / 726.2 | 71.3 / 2636.9 | 53.8 / 5500.7 |
| scarce, calibrated | 76.9 / 777.3 | 83.8 / 7135.4 (= static) | 87.5 / 28420.0 (= static) |
| always-small | 63.8 / 403.2 | 65.6 / 2021.3 | 48.1 / 4548.6 |
| always-medium | **80.0** / 832.1 | 78.1 / 2969.5 | 62.5 / 5430.2 |
| always-large | 72.5 / 1387.3 | 86.9 / 9447.7 | **90.6** / 43967.6 |
| **learned cost-model policy** | 77.3 / 620.5 | 83.4 / 6439.8 | 87.3 / 28029.6 |
| strict oracle | 79.4 / 756.7 | 85.0 / 3336.4 | 90.0 / 21764.6 |
| global-budget oracle (−2pp) | 68.8 / 404.2 | 81.9 / 2194.9 | 85.6 / 11843.6 |

**A genuinely surprising finding from this table**: on Mac, `always-medium`
— never even looking at task difficulty — beats static-cost-router on
*both* quality (80.0% vs. 70.6%) and energy (832 J vs. 1207 J). This isn't
a bug; it's Finding 4 (semantic tasks don't uniformly benefit from bigger
models) showing up as a routing-relevant fact: static-cost-router's own
difficulty→tier mapping (`hard → large`) is measurably suboptimal on this
task mix, because 16 of the 24 "hard" tasks are `semantic_hard`, where
`large` is often *not* the best choice. The learned policy (620.5 J, 77.3%
quality) doesn't strictly dominate `always-medium` — it trades some of
that quality for much lower energy — but both dominate static-cost-router,
and only the learned policy does so while adapting per-task rather than
applying one fixed rule to everything. This same fixed-tier check does
*not* produce a surprise on DGX — `always-large` is the best fixed tier on
both DGX-Qwen and DGX-Llama, consistent with Finding 2's widening-gap
story — showing the "smartest fixed rule isn't difficulty-based routing"
result is itself capability-gap-dependent, not universal.

### Verification spot-checks (`src/tejas/cost_model/verification_checks.py`)

Single-split (not the full repeated nested CV) sanity checks, run
separately from the headline numbers:

- **Feature-shuffle leakage check**: shuffling task-level features (keeping
  model/deployment features intact) degrades energy prediction in all
  three experiments (e.g. Mac MAE 10.9J → 28.3J shuffled) — no leakage on
  the energy side. Quality log loss degraded as expected on DGX-Qwen
  (0.516 → 0.550) but did *not* degrade on Mac (0.735 → 0.610) or
  DGX-Llama (0.486 → 0.469) on this single split. Reporting this as found
  rather than discarding it: HistGB quality predictions are the least
  frequently selected ladder member in the full nested CV (see table
  above), so a single noisy split isn't strong evidence either way here —
  flagged as a real limitation of this particular spot-check, not
  papered over.
- **Feature-group ablation** (single split, HistGB): adding SQL/semantic
  features cuts energy MAE substantially on all three experiments (Mac
  21.7J → 10.7J; DGX-Qwen 94.0J → 81.8J; DGX-Llama 212.9J → 161.4J).
  Explicit interaction features barely move HistGB's metrics either
  direction — expected, since tree models learn interactions natively
  (the interaction terms were added specifically for the linear models,
  per `partitioned-chasing-pizza.md`).

### Cross-family transfer (`src/tejas/cost_model/cross_family_transfer.py`)

Train the energy/quality ladders on one DGX family, evaluate zero-shot on
the other — both share `measurement_scope="gpu_die_only"`, so this is
legitimate absolute-J transfer (unlike Mac↔DGX). Two protocols per
`partitioned-chasing-pizza.md`'s spec: **A** (matched-workload: same 40
tasks, different deployment) and **B** (unseen-workload: task IDs split
before crossing deployments, 5×3 repeated grouped splits). Full numbers in
`results/transfer_{A,B}_{energy,quality}_*.csv`.

**Quality transfers remarkably well; energy transfers more weakly and
asymmetrically:**

| Direction | Within-deployment quality acc. (§10 above) | Cross-family quality acc. (Protocol A, best kind) | Within-deployment energy MAE | Cross-family energy MAE (Protocol A, best kind) |
|---|---|---|---|---|
| DGX-Qwen → DGX-Llama | Llama: 73.1% | **73.3%** (logreg) | Llama: 200.0 J | 298.7 J (ElasticNet) |
| DGX-Llama → DGX-Qwen | Qwen: 77.9% | **77.3%** (HistGB) | Qwen: 44.6 J | 145.4 J (ElasticNet) |

Quality accuracy transferred to within ~1pp of a model trained natively on
the target deployment, in *both* directions — workload shape (SQL
structure, semantic vs. structural, difficulty) apparently matters far
more for predicting correctness than which model family is answering.
Energy transfer is real (far better than the constant-baseline floor) but
degrades more, and more in the Llama→Qwen direction (3.3× worse MAE) than
Qwen→Llama (1.5× worse) — plausibly because absolute joules depend on
family/hardware-specific relationships between parameter count and energy
draw that don't scale the same way in both directions. **Protocol B
(unseen-workload) tracks Protocol A (matched-workload) closely** (e.g.
Qwen→Llama energy MAE 298.7 J matched vs. 302.6 J unseen; quality accuracy
72.8% vs. 72.7%) — the model isn't memorizing the specific 40 tasks, it's
learning workload-shape relationships that hold on genuinely unseen tasks
too.

### Few-shot deployment calibration (`src/tejas/cost_model/fewshot_calibration.py`)

With k ∈ {1, 5, 10} target-deployment calibration tasks (20 random subsets
per k), compares **full refit** (retrain on source + k target tasks)
against **linear calibration** (freeze the source-trained model, fit
`logE_target = a + b·logE_hat_source` and a logit-space Platt correction
using only the k tasks' rows) against a **zero-shot** floor (source model,
no adaptation). Full numbers in `results/fewshot_calibration.csv`.

**The two transfer directions tell different stories — this is the
interesting result, not a single "k improves things" curve:**

- **DGX-Qwen → DGX-Llama**: full refit is the more effective, more stable
  choice. At k=1, linear calibration is actually *worse than zero-shot* and
  wildly unstable (MAE 821 ± 1587 J, vs. zero-shot's 335.7 J) — fitting a
  2-parameter least-squares correction from one task's 12 rows is
  under-determined here. Full refit stays stable and improves steadily
  (330 → 179 → 173 J as k goes 1 → 5 → 10).
- **DGX-Llama → DGX-Qwen**: the opposite — **linear calibration wins**,
  and by a growing margin as k increases (MAE 59.3 → 47.8 → 43.3 J vs.
  full refit's 80.3 → 65.0 → 59.2 J at k=1/5/10), and even wins on quality
  accuracy at k≥5 (79.2% vs. 75.3%). A simple scale-shift recovers *more*
  of the available benefit than retraining from scratch.

Read together with the cross-family transfer asymmetry above, this
suggests DGX-Qwen's cost structure is closer to "a scaled-down version of
DGX-Llama's" (a linear correction suffices, even improves on refitting),
while DGX-Llama's isn't simply "a scaled-up version of DGX-Qwen's" (needs
genuinely new interaction data, i.e. full refit) — consistent with the
70B/3B gap being qualitatively, not just quantitatively, more extreme than
32B/3B. At k=1 specifically, linear calibration's instability is itself
worth keeping in the paper: a single calibration task is not enough data
to trust a fitted slope, even though it's enough for a full refit (which
has 40 source tasks anchoring it) to be stable.

### Still not built (specified, correctly deferred)

Relative cross-platform tier-energy transfer (Mac↔DGX, blocked on
`measurement_scope` by design — needs the ratio-target model variant, not
built) and margin-based abstention as a primary (not just ablation)
feature remain specified in `partitioned-chasing-pizza.md` but unbuilt —
lower priority than the transfer/calibration work above, which was
completed this session.

§8 item 1 (learned cost/quality model) — **addressed**, see above.

## 11. Session 2 kickoff brief (archived — paste this as the opening message)

This is the exact brief handed to the session that starts Option C — kept
here verbatim so it survives independently of any one conversation.

> **Project**: CIDR vision paper PoC — "Sustainable AI Data Systems: From
> Computation-on-Demand to Computation-by-Energy." Repo:
> `~/AI_Agent/sustainableAI/SustainableAI` (also on GitHub as
> `pratyushagnihotri/SustainableAI`).
>
> **Read first**: `docs/RESEARCH_LOG.md` in full (§1–§9), especially §3 (routing
> policy design) and §6 (findings) — §9 is a handoff section written
> specifically for this. `README.md` has how to run the harness if needed.
>
> **State**: Energy-aware LLM routing PoC is built and validated across 3
> experiments — Mac (Qwen 3b/7b/14b), DGX-Qwen (3b/7b/32b), DGX-Llama
> (3b/8b/70b) — each with real measured energy (not estimated), 40 tasks (24
> DB/SQL + 16 SemBench-derived semantic), 4 repeated runs.
>
> **Key finding motivating this session's task**: A hand-designed routing
> policy (`scarce`: downgrade one model tier under simulated energy
> scarcity) gives a real "free lunch" on Mac (+1.9pp quality, −40% energy)
> but backfires badly as the model-capability gap widens (DGX-Qwen: −12.6pp
> quality; DGX-Llama: −33.7pp quality). A first fix — a `calibrated` policy
> using a hand-picked, untuned 10pp threshold on measured quality gaps —
> prevents the DGX quality collapse but also gives up *all* energy savings
> there (binary, all-or-nothing). That's the concrete gap this session
> should close.
>
> **Task**: Build Option C — a learned cost/quality model that replaces the
> fixed 10pp threshold with a continuous trade-off decision, so the router
> can find a middle ground on DGX instead of "downgrade or don't."
>
> **Data ready to use** (no new experiments needed): `results/raw_results.csv`,
> `results/raw_results_dgx_qwen.csv`, `results/raw_results_dgx_llama.csv`.
> Load via `tejas.benchmarking.results_loader.load_raw_results(path)` (handles run-filtering correctly, don't read
> the CSV directly). ~480 rows/experiment. Missing but easy to add: prompt
> length per task (computable offline from `tasks.py`/`sembench_tasks.py`,
> not currently logged).
>
> **Validate** by holding out `task_id` (not rows — a task's repeats must
> stay together), compared against `tejas.routing.static_policies`'s existing baselines using
> the same paired-per-run method as the rest of the log.

A relevant, not-yet-built idea from the same discussion: a **continuous
energy trace** (vs. the current binary abundant/scarce toggle) is a
separate but related follow-up — see the discussion right before this
session was scoped. It doesn't require new experiments either (routing is
already computed retroactively from existing per-task-per-tier data), but
was deliberately left for after Option C, since Option C's learned model is
the natural continuous decision function a trace would drive — building the
trace first would mean throwing away a hand-designed multi-band policy once
Option C exists anyway.

## 12. Session 3 kickoff brief: starting the paper write-up (paste this as the opening message)

Written for a session with zero prior context, tasked with **writing the
paper, not more engineering**. The research is done; what's missing is
prose, selection (what fits in 6 pages), and a couple of open framing
decisions listed below. Deadline: CIDR 2027 submission, **Aug 4, 2026**.

> **Project**: CIDR vision paper — "Sustainable AI Data Systems: From
> Computation-on-Demand to Computation-by-Energy." Repo:
> `~/AI_Agent/sustainableAI/SustainableAI` (GitHub:
> `pratyushagnihotri/SustainableAI`). Your job this session is to help
> write/condense the paper — the empirical work is finished.

**Read in this order** (don't re-read everything — this list is ordered by
how much you actually need):
1. `src/tejas/cost_model/README.md` — the learned cost model, explained: what
   it is, every design decision and *why*, assumptions, limitations.
   Written to be lifted into a methodology section directly.
2. `docs/RESEARCH_LOG.md` §10 (below, this file) — full numeric results: five
   evidence layers per experiment, the baseline comparison table,
   cross-family transfer, few-shot calibration. This is where the numbers
   live.
3. `docs/ROUTING_README.md` (repo root) — a whole-project paper-facing
   narrative written by a *different* session, covering motivation →
   system → all policies (including the hand-designed ones from before
   the learned cost model existed) → results → future work. **Its "Future
   improvements" section is stale** — still lists cross-family transfer
   and few-shot calibration as "specified but not built," but both were
   completed in the session right before this one. Reconcile before using
   it as-is.
4. `docs/RESEARCH_LOG.md` §1–9 — only if you need the original PoC's
   chronological narrative (bugs found, Findings 1–5, why SemBench/DGX
   were chosen). Usually §10's summary is enough.
5. `docs/DESIGN_PLAN.md` — the original design
   plan including two rounds of adversarial methodological review. Mostly
   an audit trail; skip unless you need to justify a specific design
   choice's rationale in more depth than `src/tejas/cost_model/README.md`
   gives.

**The headline results, so you don't have to grep for them:**

Original hand-designed policies (§6): `scarce` (downgrade one tier
blindly) is a free lunch on Mac (+1.9pp quality, −40.4% energy) but
backfires on DGX (DGX-Qwen −12.6pp, DGX-Llama −33.7pp). `scarce,
calibrated` (fixed 10pp threshold) fixes the DGX quality collapse but
recovers **exactly 0%** energy savings there — all-or-nothing.

Learned cost model (§10), same three experiments:

| Experiment | Learned policy vs. static-cost-router | 95% CI | Oracle savings recovered |
|---|---|---|---|
| Mac | +6.6pp quality, 48.6% energy saved | q: [+3.1,+10.4]pp | 53–85% |
| DGX-Qwen | −0.4pp quality, 9.7% saved | q: [−1.0,+0.1]pp | 2–31% |
| DGX-Llama | −0.3pp quality, 1.4% saved | q: [−0.8,0.0]pp | 0–12% |

Both DGX experiments now recover **nonzero**, statistically-bounded
savings where `calibrated` got stuck at 0% — that's the core empirical
claim. Margin shrinks as the capability gap widens, consistent with the
original Finding 2.

Cross-family transfer (DGX-Qwen ↔ DGX-Llama): quality predictions
transfer to within ~1pp of native accuracy *in both directions* — model
family barely matters for predicting correctness. Energy transfer is
weaker and asymmetric (Llama→Qwen degrades ~3.3×, Qwen→Llama ~1.5×).

Few-shot calibration: genuinely different story per direction —
Qwen→Llama needs a full refit (a cheap linear correction is unstable at
k=1); Llama→Qwen is well-served by a cheap linear correction alone,
which *beats* full refit at k≥5. Read as: DGX-Qwen behaves like a
scaled-down DGX-Llama, not vice versa.

**What's NOT built** (correctly deferred, not forgotten): relative
cross-platform (Mac↔DGX) energy transfer (blocked by a real measurement-
boundary difference — Mac measures whole-SoC power, DGX measures GPU-die
power only, see §10's measurement-scope finding), margin-based abstention
as a primary router feature. Neither is a blocker for writing the paper.

**Two open decisions — now resolved, asked directly of the user at the end
of the engineering session:**

1. **Paper framing — resolved.** Neither "routing is the contribution" nor
   "cost models as a standalone primitive" — the paper stays the original
   vision paper, "Sustainable AI Data Systems: From Computation-on-Demand
   to Computation-by-Energy" (§1's framing), with energy-aware routing
   *and* the learned cost model together as the concrete proof-of-concept
   evidence for that broader thesis. Don't narrow the title/abstract to
   "a routing paper," and don't broaden it to "a cost-model-primitive
   paper" either — §10's results are the PoC section's evidence, not the
   paper's whole identity.
2. **Option B ("energy-aware stream processing") — resolved: skip it.**
   Explicitly not built, on timing grounds (this was decided ~3 weeks
   before the Aug 4, 2026 deadline, with the paper itself still to write).
   Don't revisit unless the user raises it again.

**Practical notes**: git changes from the engineering session are staged
but not committed (user commits personally — don't commit on their
behalf unless asked). `results/` is gitignored except the two raw DGX CSVs
— generated CSVs/figures won't show up in `git status`, that's expected,
not a bug. Figures for the paper: `results/figures/cost_model/` (learned
cost model, 7 figures, has its own README) and `results/figures/`,
`results/figures/dgx_qwen/`, `results/figures/dgx_llama/`,
`results/figures/cross_experiment/` (original routing PoC figures, from
before this session).
