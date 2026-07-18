"""Offline, no-rerun-needed feature extraction for the 40 tasks shared by
all three experiments (tasks.py's 24 DB/SQL tasks + sembench_tasks.py's 16
semantic tasks). Used by energy_model.py / quality_model.py to build
training features, and by evaluate_models.py's feature ablation.

SQL counts (n_joins, n_predicates, ...) are parsed from each task's
`gold_sql` with regexes, not a real SQL parser — call them lexical proxies,
not structural analysis. Regex parsing of commas/keywords can miscount
edge cases (function-call commas, aliased columns, deeply nested clauses)
that a real parser (e.g. sqlglot) wouldn't. The self-check below hand-
verifies a handful of tasks so at least the common shapes here are right.
"""

import re

from tejas.benchmarking.tasks import TASKS as SQL_CONCEPT_TASKS
from tejas.benchmarking.sembench_tasks import TASKS as SEMANTIC_TASKS

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_JOIN_RE = re.compile(r"\bJOIN\b", re.I)
_PREDICATE_RE = re.compile(r"\bWHERE\b|\bAND\b|\bOR\b|\bHAVING\b", re.I)
_AGG_RE = re.compile(r"\b(?:COUNT|SUM|AVG|MAX|MIN)\s*\(", re.I)
_SELECT_RE = re.compile(r"\bSELECT\b", re.I)
_WINDOW_RE = re.compile(r"\bOVER\s*\(", re.I)
_SELECT_FROM_RE = re.compile(r"\bSELECT\b(.*?)\bFROM\b", re.I | re.S)
_GROUPBY_RE = re.compile(r"\bGROUP BY\b(.*?)(?:\bHAVING\b|\bORDER BY\b|\bLIMIT\b|$)", re.I | re.S)
_N_REVIEWS_RE = re.compile(r"a sample of (\d+) reviews")


def _split_top_level(s, sep=","):
    """Comma-split respecting parenthesis depth, so COUNT(a, b)-style calls
    (not that any task here has one) wouldn't be miscounted as 2 columns."""
    parts, depth, current = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def _n_projected_cols(sql):
    m = _SELECT_FROM_RE.search(sql)
    return len(_split_top_level(m.group(1))) if m else 0


def _n_groupby_cols(sql):
    m = _GROUPBY_RE.search(sql)
    return len(_split_top_level(m.group(1))) if m else 0


def _sql_lexical_proxies(gold_sql):
    return {
        "n_joins": len(_JOIN_RE.findall(gold_sql)),
        "n_predicates": len(_PREDICATE_RE.findall(gold_sql)),
        "n_aggregations": len(_AGG_RE.findall(gold_sql)),
        "n_groupby_cols": _n_groupby_cols(gold_sql),
        "n_projected_cols": _n_projected_cols(gold_sql),
        "has_subquery": len(_SELECT_RE.findall(gold_sql)) > 1,
        "has_window_fn": bool(_WINDOW_RE.search(gold_sql)),
    }


_ZERO_SQL_PROXIES = {
    "n_joins": 0, "n_predicates": 0, "n_aggregations": 0,
    "n_groupby_cols": 0, "n_projected_cols": 0,
    "has_subquery": False, "has_window_fn": False,
}


def build_task_features():
    features = {}

    for t in SQL_CONCEPT_TASKS:
        prompt = t["prompt"]
        base = {
            "task_type": t["type"],
            "task_tier": t["tier"],
            "task_type_tier": f"{t['type']}_{t['tier']}",
            "prompt_chars": len(prompt),
            "prompt_tokens": len(_TOKEN_RE.findall(prompt)),
            "n_reviews": 0,
        }
        if t["type"] == "sql":
            base["answer_format"] = "sql"
            base.update(_sql_lexical_proxies(t["gold_sql"]))
        else:  # concept
            base["answer_format"] = "short_text"
            base.update(_ZERO_SQL_PROXIES)
        features[t["id"]] = base

    for t in SEMANTIC_TASKS:
        prompt = t["prompt"]
        m = _N_REVIEWS_RE.search(prompt)
        base = {
            "task_type": t["type"],
            "task_tier": t["tier"],
            "task_type_tier": f"{t['type']}_{t['tier']}",
            "prompt_chars": len(prompt),
            "prompt_tokens": len(_TOKEN_RE.findall(prompt)),
            "n_reviews": int(m.group(1)) if m else 0,
            "answer_format": t["semantic_kind"],
        }
        base.update(_ZERO_SQL_PROXIES)
        features[t["id"]] = base

    return features


# Hand-verified expected lexical-proxy counts for a handful of tasks.py's
# SQL tasks, spanning simple/join/subquery/window/having shapes — catches
# regressions in the regexes above, not a claim of full SQL-parser coverage.
_EXPECTED = {
    "e1": {"n_joins": 1, "n_predicates": 1, "n_aggregations": 0, "n_groupby_cols": 0,
           "n_projected_cols": 1, "has_subquery": False, "has_window_fn": False},
    "e2": {"n_joins": 0, "n_predicates": 0, "n_aggregations": 1, "n_groupby_cols": 0,
           "n_projected_cols": 1, "has_subquery": False, "has_window_fn": False},
    "m1": {"n_joins": 1, "n_predicates": 0, "n_aggregations": 1, "n_groupby_cols": 1,
           "n_projected_cols": 2, "has_subquery": False, "has_window_fn": False},
    "h1": {"n_joins": 0, "n_predicates": 2, "n_aggregations": 1, "n_groupby_cols": 0,
           "n_projected_cols": 1, "has_subquery": True, "has_window_fn": False},
    "h2": {"n_joins": 0, "n_predicates": 0, "n_aggregations": 0, "n_groupby_cols": 0,
           "n_projected_cols": 3, "has_subquery": False, "has_window_fn": True},
    "h4": {"n_joins": 2, "n_predicates": 1, "n_aggregations": 1, "n_groupby_cols": 1,
           "n_projected_cols": 1, "has_subquery": False, "has_window_fn": False},
}


def _self_check():
    features = build_task_features()
    assert len(features) == 40, f"expected 40 tasks, got {len(features)}"
    n_sql = sum(1 for f in features.values() if f["task_type"] == "sql")
    n_concept = sum(1 for f in features.values() if f["task_type"] == "concept")
    n_semantic = sum(1 for f in features.values() if f["task_type"] == "semantic")
    assert (n_sql, n_concept, n_semantic) == (18, 6, 16), (n_sql, n_concept, n_semantic)

    failures = []
    for task_id, expected in _EXPECTED.items():
        got = features[task_id]
        for k, v in expected.items():
            if got[k] != v:
                failures.append(f"{task_id}.{k}: expected {v}, got {got[k]}")
    if failures:
        raise AssertionError("SQL lexical-proxy self-check failed:\n" + "\n".join(failures))

    # Every semantic task samples load_sembench_movie.SAMPLE_SIZE (24) reviews
    # regardless of the source movie's full corpus size (120 for taken_3, 256
    # for ant_man) — the corpus size only affects sampling diversity, not the
    # per-task n shown to the model.
    semantic_reviews = {f["n_reviews"] for tid, f in features.items() if f["task_type"] == "semantic"}
    assert semantic_reviews == {24}, semantic_reviews

    print(f"{len(features)} tasks OK ({n_sql} sql, {n_concept} concept, {n_semantic} semantic)")
    print(f"SQL lexical-proxy self-check passed for {len(_EXPECTED)} hand-verified tasks")
    print(f"semantic n_reviews values found: {sorted(semantic_reviews)}")


if __name__ == "__main__":
    _self_check()
