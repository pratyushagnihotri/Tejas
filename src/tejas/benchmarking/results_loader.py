"""Reads the harness's raw per-(run, task, model_tier) CSV output correctly:
only aggregates runs that share the exact same task set, so a benchmark
revision (e.g. widening the task suite) never silently blends old and new
runs into one misleading average.

Also carries the two route tables (STATIC_ROUTE, ENERGY_AWARE_ROUTE) that
define what "difficulty tier -> model tier" means for this benchmark, since
that mapping is a property of the benchmark/task design (which tier each
task belongs to) as much as it is a routing policy choice — kept here
rather than in tejas.routing specifically so tejas.cost_model can depend on
it without creating a pillar-3-depends-on-pillar-4 cycle (the cost model
needs a task's reference tier as a feature; it should not need to import
the routing pillar to get it).
"""

import csv
from collections import defaultdict

STATIC_ROUTE = {"easy": "small", "medium": "medium", "hard": "large"}
ENERGY_AWARE_ROUTE = {
    "abundant": {"easy": "small", "medium": "medium", "hard": "large"},
    "scarce": {"easy": "small", "medium": "small", "hard": "medium"},
}


def load_raw_results(raw_path):
    """Only aggregates runs that share the exact same task set. Different
    runs can come from different benchmark revisions (e.g. before vs. after
    widening the SemBench set) — silently averaging across those would
    blend two different experiments into one misleading number."""
    with open(raw_path) as f:
        all_rows = list(csv.DictReader(f))

    run_task_ids = defaultdict(set)
    for row in all_rows:
        run = row.get("run") or "1"
        run_task_ids[run].add(row["task_id"])

    canonical_run = max(run_task_ids, key=lambda r: len(run_task_ids[r]))
    canonical_set = run_task_ids[canonical_run]
    valid_runs = sorted(
        [r for r in run_task_ids if run_task_ids[r] == canonical_set], key=int)
    excluded_runs = sorted(
        [r for r in run_task_ids if r not in valid_runs], key=int)

    if excluded_runs:
        print(f"Note: excluding run(s) {', '.join(excluded_runs)} from aggregation — "
              f"different task set ({len(run_task_ids[excluded_runs[0]])} tasks) than "
              f"the current one ({len(canonical_set)} tasks). Those runs predate a "
              f"task-set change and aren't comparable to it.\n")

    by_run_task_model = {}
    task_tier = {}
    for row in all_rows:
        run = row.get("run") or "1"
        if run not in valid_runs:
            continue
        by_run_task_model[(run, row["task_id"], row["model_tier"])] = row
        task_tier[row["task_id"]] = row["task_tier"]

    return by_run_task_model, task_tier, valid_runs
