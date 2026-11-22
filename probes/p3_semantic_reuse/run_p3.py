"""P3: can the DBMS reuse semantic results in ways an LLM response cache cannot?

Table: SemBench Movie reviews that carry a graded critic score (1,418 of 2,000), so
ground truth exists for graded predicates. A dashboard-like workload of 7 semantic
filters is run in order:

  W1 positive            "clearly positive"                       (SemBench Q1 predicate)
  W2 rave                "enthusiastic rave"          rave  ⊑ positive   (declared)
  W3 negative            "negative"                   neg   ≡ ¬positive  (declared)
  W4 pan                 "harsh pan"                  pan   ⊑ negative   (declared)
  W5 liked               "the critic liked the movie" ≈ positive (NOT declared; paraphrase)
  W6 positive@taken_3    W1 restricted to one movie (SemBench Q2): relational selection
  W7 positive (refresh)  exact repeat of W1

Reuse strategies (each runs the whole workload from an empty store):
  R0 recompute    large model for every (query, tuple)
  R1 exact        tuple-level cache keyed by (predicate text, tuple)  ~ what an LLM cache does
  R2 logical      R1 + rules from the declared lattice: subsumption prunes the narrower
                  predicate to tuples that passed the broader one; complement derives
                  negative from positive without any call
  R3 similarity   R2 + an LLM equivalence check between predicate texts (one call per new
                  predicate); an equivalent predicate reuses the earlier answers
  R4 transform    R3, but residual evaluations of narrower predicates use a small model

Reported: energy, calls (large/small), tokens, reuse rate, F1 per query vs ground truth,
and model consistency (does recomputation by the model itself respect the lattice?).

  python -m probes.p3_semantic_reuse.run_p3 --out probes/results/p3 --reps 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from probes.common.data import load_reviews
from probes.common.energy import EnergyMeter
from probes.common.llm import LLM, from_env

# ------------------------------------------------------------------ ground truth
_LETTER = {"A+": 1.0, "A": .95, "A-": .9, "B+": .85, "B": .8, "B-": .75, "C+": .65, "C": .6, "C-": .55,
           "D+": .45, "D": .4, "D-": .35, "F": .1}


def norm_score(s: str) -> Optional[float]:
    s = s.strip().upper().replace(" ", "")
    if not s:
        return None
    if s in _LETTER:
        return _LETTER[s]
    m = re.fullmatch(r"([0-9.]+)/([0-9.]+)", s)
    if m:
        try:
            n, d = float(m.group(1)), float(m.group(2))
            return n / d if d > 0 and 0 <= n <= d else None
        except ValueError:
            return None
    return None


RAVE_T, PAN_T = 0.9, 0.3


def build_table() -> List[Dict]:
    rows = []
    for r in load_reviews():
        x = norm_score(r["originalScore"])
        if x is None:
            continue
        r = dict(r, score=x)
        r["gt"] = {
            "positive": r["label_positive"],
            "negative": not r["label_positive"],
            # graded labels are nested inside the binary label so the lattice holds in ground truth
            "rave": r["label_positive"] and x >= RAVE_T,
            "pan": (not r["label_positive"]) and x <= PAN_T,
        }
        rows.append(r)
    return rows


# ------------------------------------------------------------------ workload
PRED = {
    "positive": "The following movie review is clearly positive.",
    "rave": "The following movie review is an enthusiastic rave: the critic loved the movie.",
    "negative": "The following movie review is negative.",
    "pan": "The following movie review is a harsh pan: the critic strongly disliked the movie.",
    "liked": "The critic who wrote the following movie review liked the movie.",
}
# (query name, predicate key, ground-truth concept, optional movie filter)
WORKLOAD = [
    ("W1_positive", "positive", "positive", None),
    ("W2_rave", "rave", "rave", None),
    ("W3_negative", "negative", "negative", None),
    ("W4_pan", "pan", "pan", None),
    ("W5_liked", "liked", "positive", None),
    ("W6_positive_taken3", "positive", "positive", "taken_3"),
    ("W7_positive_refresh", "positive", "positive", None),
]
# Declared lattice: what a semantic catalog would state. 'liked' is deliberately absent.
SUBSUMED_BY = {"rave": "positive", "pan": "negative"}  # narrower -> broader
COMPLEMENT = {"negative": "positive", "positive": "negative"}


def prompt(pkey_or_text: str, r: Dict) -> str:
    text = PRED.get(pkey_or_text, pkey_or_text)
    return f"{text}\n\nReview:\n{r['reviewText']}\n\nDoes the statement hold? Answer yes or no."


# ------------------------------------------------------------------ semantic store
class Store:
    """Materialised semantic results: (predicate text, reviewId) -> (answer, provenance)."""

    def __init__(self):
        self.ans: Dict[Tuple[str, str], Tuple[bool, str]] = {}
        self.alias: Dict[str, str] = {}  # predicate text -> canonical predicate text (similarity)

    def get(self, p, rid):
        return self.ans.get((self.alias.get(p, p), rid))

    def put(self, p, rid, a, prov):
        self.ans[(self.alias.get(p, p), rid)] = (a, prov)

    def known_predicates(self):
        return sorted({p for p, _ in self.ans})


async def equivalent(llm: LLM, a: str, b: str) -> bool:
    q = (f"Statement A: {a}\nStatement B: {b}\n\nFor any movie review, does statement A hold exactly "
         f"when statement B holds? Answer yes or no.")
    return await llm.yes_no(q)


async def run_query(strategy, store: Store, big: LLM, small: Optional[LLM], pkey, rows, counters):
    ptext = PRED[pkey]
    declared = pkey in SUBSUMED_BY or pkey in COMPLEMENT
    # The catalog is consulted first; similarity is only a fallback for undeclared predicates.
    if strategy >= 3 and not declared and ptext not in store.alias and store.known_predicates():
        for known in store.known_predicates():
            if known != ptext and await equivalent(big, ptext, known):
                store.alias[ptext] = known
                counters["equiv_hits"] += 1
                break
    todo, out = [], {}
    for r in rows:
        rid = r["reviewId"]
        if strategy >= 1:
            hit = store.get(ptext, rid)
            if hit is not None:
                out[rid] = hit[0]
                counters["reused_exact_or_alias"] += 1
                continue
        if strategy >= 2:
            comp = COMPLEMENT.get(pkey)
            if comp is not None and store.get(PRED[comp], rid) is not None:
                out[rid] = not store.get(PRED[comp], rid)[0]
                store.put(ptext, rid, out[rid], f"complement({comp})")
                counters["reused_logical"] += 1
                continue
            broader = SUBSUMED_BY.get(pkey)
            if broader is not None:
                b = store.get(PRED[broader], rid)
                if b is None and COMPLEMENT.get(broader) and store.get(PRED[COMPLEMENT[broader]], rid) is not None:
                    b = (not store.get(PRED[COMPLEMENT[broader]], rid)[0], "derived")
                if b is not None and not b[0]:
                    out[rid] = False  # fails the broader predicate -> fails the narrower one
                    store.put(ptext, rid, False, f"subsumption({broader})")
                    counters["reused_logical"] += 1
                    continue
        todo.append(r)
    use_small = strategy >= 4 and small is not None and pkey in SUBSUMED_BY
    model = small if use_small else big
    res = await model.yes_no_many([prompt(pkey, r) for r in todo])
    for r, a in zip(todo, res):
        out[r["reviewId"]] = a
        if strategy >= 1:
            store.put(ptext, r["reviewId"], a, "small" if use_small else "large")
    counters["computed_small" if use_small else "computed_large"] += len(todo)
    return out


def f1(pred: Dict[str, bool], rows, concept) -> float:
    P = {k for k, v in pred.items() if v}
    T = {r["reviewId"] for r in rows if r["gt"][concept]}
    tp = len(P & T)
    p, rc = (tp / len(P) if P else 0), (tp / len(T) if T else 0)
    return 2 * p * rc / (p + rc) if p + rc else 0.0


async def run_workload(strategy, big, small, table):
    store = Store()
    counters = {k: 0 for k in ["computed_large", "computed_small", "reused_exact_or_alias",
                               "reused_logical", "equiv_hits"]}
    quality, answers = {}, {}
    for qname, pkey, concept, movie in WORKLOAD:
        rows = [r for r in table if movie is None or r["id"] == movie]
        out = await run_query(strategy, store, big, small, pkey, rows, counters)
        quality[qname] = f1(out, rows, concept)
        answers[qname] = out
    total = sum(len([r for r in table if m is None or r["id"] == m]) for _, _, _, m in WORKLOAD)
    counters["evaluations"] = total
    counters["reuse_rate"] = (counters["reused_exact_or_alias"] + counters["reused_logical"]) / total
    return counters, quality, answers


def consistency(answers) -> Dict[str, float]:
    """From R0 (independent recomputation): does the model itself respect the lattice?"""
    pos, rave, neg, pan, liked = (answers[k] for k in
                                  ["W1_positive", "W2_rave", "W3_negative", "W4_pan", "W5_liked"])
    ids = list(pos)
    frac = lambda xs: sum(xs) / len(xs) if xs else None
    return {
        "rave_implies_positive": frac([pos[i] for i in ids if rave[i]]),
        "pan_implies_negative": frac([neg[i] for i in ids if pan[i]]),
        "negative_is_not_positive": frac([neg[i] != pos[i] for i in ids]),
        "liked_agrees_positive": frac([liked[i] == pos[i] for i in ids]),
    }


async def amain(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = build_table()
    if args.rows:
        table = table[: args.rows]
    big = from_env()
    small = None
    if os.environ.get("PROBE_SMALL_MODEL"):
        small = LLM(base_url=os.environ.get("PROBE_SMALL_BASE_URL", big.base_url),
                    model=os.environ["PROBE_SMALL_MODEL"], max_concurrency=big.max_concurrency)
    meter = EnergyMeter()
    meta = {"model": big.model, "small_model": small.model if small else None, "rows": len(table),
            "gt_rates": {c: sum(r["gt"][c] for r in table) / len(table) for c in ["positive", "rave", "pan"]},
            "energy_available": meter.available, "args": vars(args)}
    meta["idle"] = meter.idle_power(args.idle_seconds)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    strategies = [0, 1, 2, 3] + ([4] if small else [])
    names = {0: "R0_recompute", 1: "R1_exact", 2: "R2_logical", 3: "R3_similarity", 4: "R4_transform"}
    with open(out / "records.jsonl", "a") as f:
        for rep in range(args.reps):
            for s in (strategies if rep % 2 == 0 else strategies[::-1]):
                for m in (big, small):
                    if m:
                        m.reset_usage()
                        m.reset_prefix_cache()
                t0 = meter.start()
                counters, quality, answers = await run_workload(s, big, small, table)
                w = meter.stop(t0)
                rec = {"rep": rep, "strategy": names[s], "window": w.to_dict(), "counters": counters,
                       "usage_large": big.usage.to_dict(), "usage_small": small.usage.to_dict() if small else None,
                       "quality_f1": quality}
                if s == 0:
                    rec["consistency"] = consistency(answers)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"rep{rep} {names[s]:14s} t={w.seconds:7.1f}s gpuJ={w.gpu_j} large={counters['computed_large']} "
                      f"small={counters['computed_small']} reuse={counters['reuse_rate']:.2f} "
                      f"f1={ {k: round(v, 3) for k, v in quality.items()} }"
                      + (f" consistency={rec['consistency']}" if s == 0 else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--idle-seconds", type=float, default=60.0)
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
