"""Grading for the three task types (sql / concept / semantic).

SQL tasks are graded by execution accuracy: run the candidate's extracted
query and the trusted gold query against the same DB connection, compare
result sets (order-insensitive, floats rounded to 2dp). Concept tasks are
graded by substring match against one or more accepted phrases. Semantic
tasks (SemBench-derived) ask the model to read embedded text and return a
JSON answer, graded against a gold answer precomputed from a hidden label.
"""

import json
import re
import sqlite3

_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text):
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    if ";" in text:
        text = text.split(";")[0]
    return text.strip()


def _is_select_only(sql):
    return sql.strip().upper().startswith(("SELECT", "WITH"))


def _normalize_rows(rows):
    norm = set()
    for row in rows:
        norm.add(tuple(round(v, 2) if isinstance(v, float) else v for v in row))
    return norm


def grade_sql(conn, response_text, gold_sql):
    candidate_sql = extract_sql(response_text)
    if not candidate_sql or not _is_select_only(candidate_sql):
        return False
    try:
        got = _normalize_rows(conn.execute(candidate_sql).fetchall())
    except sqlite3.Error:
        return False
    expected = _normalize_rows(conn.execute(gold_sql).fetchall())
    return got == expected


def _normalize_concept_text(s):
    """Lowercase and strip everything but letters/digits, so 'ReadCommitted',
    'read-committed', and 'read committed' all normalize the same way."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def grade_concept(response_text, accept_list):
    normalized_response = _normalize_concept_text(response_text)
    for phrase in accept_list:
        normalized_phrase = _normalize_concept_text(phrase)
        if normalized_phrase and normalized_phrase in normalized_response:
            return True
    return False


def extract_json(text):
    text = text.strip()
    m = _JSON_FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return None
    start = min(starts)
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj
    except json.JSONDecodeError:
        return None


def grade_semantic(response_text, task):
    obj = extract_json(response_text)
    if obj is None:
        return False
    kind = task["semantic_kind"]
    gold = task["gold"]

    if kind == "id_list_precision":
        if not isinstance(obj, list) or not obj:
            return False
        ids = {str(x) for x in obj}
        valid_ids = set(gold["sample_ids"])
        positive_ids = set(gold["positive_ids"])
        return ids.issubset(valid_ids) and ids.issubset(positive_ids)

    if kind == "count":
        if not isinstance(obj, dict):
            return False
        val = obj.get(gold["field"])
        if not isinstance(val, (int, float)):
            return False
        return abs(val - gold["value"]) <= gold.get("tol", 0)

    if kind == "ratio":
        if not isinstance(obj, dict):
            return False
        val = obj.get(gold["field"])
        if not isinstance(val, (int, float)):
            return False
        return abs(val - gold["value"]) <= gold.get("tol", 0.05)

    if kind == "breakdown":
        if not isinstance(obj, dict):
            return False
        tol = gold.get("tol", 0)
        for k, v in gold["value"].items():
            got = obj.get(k)
            if not isinstance(got, (int, float)) or abs(got - v) > tol:
                return False
        return True

    raise ValueError(f"unknown semantic_kind: {kind}")


def grade(conn, response_text, task):
    if task["type"] == "sql":
        return grade_sql(conn, response_text, task["gold_sql"])
    if task["type"] == "semantic":
        return grade_semantic(response_text, task)
    if task["type"] == "concept":
        return grade_concept(response_text, task["accept"])
    raise ValueError(f"unknown task type: {task['type']}")
