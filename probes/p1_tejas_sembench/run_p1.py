"""P1 (measurement): Tejas's model-selection question on a semantic operator.

For one model tier at a time (one vLLM server per tier), evaluate the SemBench Q1
predicate ("clearly positive") on every review, recording the answer, P(yes) from the
first-token log-probabilities, and prompt tokens. Energy is measured for the whole tier
pass (NVML + RAPL) and later attributed to tuples by token share. Policies are evaluated
offline by analyze_p1.py, as in Tejas (measured outcomes, held-out evaluation).

  python -m probes.p1_tejas_sembench.run_p1 --tier small --out probes/results/p1/<run>
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path

from probes.common.data import load_reviews
from probes.common.energy import EnergyMeter
from probes.common.llm import from_env

# SemBench Q1 wording ("clearly") and a control without the hedge word; the label is fresh/rotten.
PREDS = {
    "clearly": "The following movie review is clearly positive.",
    "plain": "The following movie review is positive.",
}
PRED = PREDS["clearly"]


def prompt(r):
    return f"{PRED}\n\nReview:\n{r['reviewText']}\n\nDoes the statement hold? Answer yes or no."


# SemBench Q5 predicate (binary join predicate over two reviews of the same movie).
SAME_SENT = "These two movie reviews express the same sentiment - either both are positive or both are negative."
PAIR_MOVIE = "ant_man_and_the_wasp_quantumania"


def items_q1(rows):
    return [{"key": r["reviewId"], "group": r["id"], "len": len(r["reviewText"]),
             "top": r["isTopCritic"] == "True", "label": r["label_positive"], "prompt": prompt(r)} for r in rows]


def items_q5pairs(rows, n=2000, seed=0):
    """n distinct unordered review pairs of the Q5 movie (fixed seed); label = same fresh/rotten label.
    group = first review, so cross-validation never splits pairs that share their anchor review."""
    import random
    rev = [r for r in rows if r["id"] == PAIR_MOVIE]
    rng = random.Random(seed)
    seen, out = set(), []
    while len(out) < n:
        a, b = rng.sample(range(len(rev)), 2)
        if (min(a, b), max(a, b)) in seen:
            continue
        seen.add((min(a, b), max(a, b)))
        r1, r2 = rev[a], rev[b]
        out.append({
            "key": f"{r1['reviewId']}|{r2['reviewId']}", "group": r1["reviewId"],
            "len": len(r1["reviewText"]) + len(r2["reviewText"]),
            "top": r1["isTopCritic"] == "True" or r2["isTopCritic"] == "True",
            "label": r1["label_positive"] == r2["label_positive"],
            "prompt": f"{SAME_SENT}\n\nReview 1:\n{r1['reviewText']}\n\nReview 2:\n{r2['reviewText']}"
                      f"\n\nDoes the statement hold? Answer yes or no.",
        })
    return out


async def amain(a):
    global PRED
    PRED = PREDS[a.pred]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = items_q5pairs(load_reviews(), n=a.rows or 2000) if a.task == "q5pairs" else \
        items_q1(load_reviews()[: a.rows] if a.rows else load_reviews())
    llm, meter = from_env(), EnergyMeter()
    idle = meter.idle_power(a.idle_seconds)
    passes = []
    for rep in range(a.reps):
        llm.reset_usage()
        llm.reset_prefix_cache()
        s = meter.start()
        res = await asyncio.gather(*(llm.yes_no_prob(r["prompt"]) for r in rows))
        w = meter.stop(s)
        passes.append({"rep": rep, "window": w.to_dict(), "usage": llm.usage.to_dict()})
        print(f"{a.tier} rep{rep}: {w.seconds:.1f}s gpuJ={w.gpu_j} acc="
              f"{sum(x['answer'] == r['label'] for x, r in zip(res, rows)) / len(rows):.3f}", flush=True)
    with open(out / f"tuples_{a.tier}.csv", "w", newline="") as f:  # answers from the last pass (temp 0)
        wr = csv.writer(f)
        wr.writerow(["reviewId", "movie", "len_chars", "isTopCritic", "label", "answer", "p_yes", "prompt_tokens"])
        for r, x in zip(rows, res):
            wr.writerow([r["key"], r["group"], r["len"], r["top"], int(r["label"]),
                         int(x["answer"]), round(x["p_yes"], 5), x["prompt_tokens"]])
    (out / f"energy_{a.tier}.json").write_text(json.dumps(
        {"tier": a.tier, "model": llm.model, "task": a.task, "pred": PRED if a.task == "q1" else SAME_SENT, "idle": idle, "passes": passes, "energy_available": meter.available},
        indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, help="label, e.g. small/medium/large")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--idle-seconds", type=float, default=60.0)
    ap.add_argument("--pred", choices=sorted(PREDS), default="clearly", help="q1 wording")
    ap.add_argument("--task", choices=["q1", "q5pairs"], default="q1",
                    help="q1: positive review; q5pairs: same-sentiment pairs (SemBench Q5 predicate)")
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
