"""P2: what does query-plan knowledge save beyond a state-of-the-art inference server?

Same model, same vLLM server, same answer semantics. Three execution regimes:

  A  decoupled:     one request per tuple, sequential, whole table, LIMIT applied after
  B  serving-aware: same requests, issued concurrently (vLLM continuous batching +
                    automatic prefix caching do what a serving layer can do on its own)
  C  plan-aware:    B plus levers that need knowledge only the DBMS has. Each lever is
                    run alone on top of B (attribution) and all together (C_all):
       proj        send only the attribute the predicate reads, not the serialised row
       limit       pipeline LIMIT: issue tuples in waves, stop once LIMIT is satisfied
       proxy       order tuples by a cheap CPU proxy so likely qualifying tuples come first
       join_order  order join pairs by the outer tuple, so its KV prefix is reused
       pred_order  order conjunctive semantic predicates by cost / (1 - selectivity)

Energy: NVML GPU counter + RAPL package counter per window. Each (query, strategy)
execution is repeated inside one window until it lasts >= --min-window seconds, with
vLLM's prefix cache reset before every execution, and reported per execution.

Usage (on the GPU node, vLLM already serving):
  python -m probes.p2_plan_aware.run_p2 --out probes/results/p2 --reps 5
Local smoke test (Ollama, no energy counters):
  PROBE_BASE_URL=http://localhost:11434/v1 PROBE_MODEL=llama3.2:1b \
    python -m probes.p2_plan_aware.run_p2 --smoke --out /tmp/p2_smoke
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from probes.common.data import load_reviews, row_text
from probes.common.energy import EnergyMeter
from probes.common.llm import from_env

POSITIVE = "The following movie review is clearly positive."
SAME_SENT = "These two movie reviews express the same sentiment - either both are positive or both are negative."
ACTING = "The following movie review praises the acting or the performances."

# ------------------------------------------------------------------ cheap CPU proxy
_POS = set("great good excellent best fun brilliant terrific wonderful enjoyable funny smart love loved "
           "delight delightful superb masterpiece entertaining charming strong solid thrilling beautiful".split())
_NEG = set("bad worst boring dull mess awful terrible weak tedious lazy fails failed waste disappointing "
           "stupid poor bland forgettable messy painful flat incoherent".split())


def proxy_score(text: str) -> float:
    w = re.findall(r"[a-z']+", text.lower())
    return (sum(x in _POS for x in w) - sum(x in _NEG for x in w)) / (1 + len(w) ** 0.5)


# ------------------------------------------------------------------ prompts
def p_filter(instr: str, r: Dict, projection: str) -> str:
    return f"{instr}\n\nReview:\n{row_text(r, projection)}\n\nDoes the statement hold? Answer yes or no."


def p_join(r1: Dict, r2: Dict, projection: str) -> str:
    # Outer tuple first: its tokens form the shareable prefix.
    return (f"{SAME_SENT}\n\nReview 1:\n{row_text(r1, projection)}\n\nReview 2:\n{row_text(r2, projection)}"
            f"\n\nDoes the statement hold? Answer yes or no.")


# ------------------------------------------------------------------ strategies
@dataclass
class Strategy:
    name: str
    concurrency: int = 64
    projection: str = "full_row"  # 'full_row' | 'needed'
    limit_stop: bool = False
    proxy: bool = False
    join_order: str = "random"  # 'random' | 'grouped'
    pred_order: str = "as_written"  # 'as_written' | 'rank'
    wave: int = 16  # tuples issued per round when LIMIT is pipelined


def strategies_for(query: str) -> List[Strategy]:
    S = [Strategy("A", concurrency=1), Strategy("B"), Strategy("C_proj", projection="needed")]
    if "limit" in query:
        S += [Strategy("C_limit", limit_stop=True), Strategy("C_limit_proxy", limit_stop=True, proxy=True)]
    if "join" in query:
        S += [Strategy("B_nested", join_order="grouped"), Strategy("C_join_order", join_order="grouped")]
    if "twopred" in query:
        S += [Strategy("C_pred_order", pred_order="rank")]
    S.append(Strategy("C_all", projection="needed", limit_stop="limit" in query, proxy="limit" in query,
                      join_order="grouped", pred_order="rank"))
    return S


# ------------------------------------------------------------------ query executors
async def run_filter(llm, rows, instr, limit: Optional[int], st: Strategy):
    order = list(rows)
    if st.proxy:
        order.sort(key=lambda r: -proxy_score(r["reviewText"]))
    out: List[Dict] = []
    if st.limit_stop and limit:
        wave = max(min(st.wave, st.concurrency), 1)
        for i in range(0, len(order), wave):
            chunk = order[i:i + wave]
            res = await llm.yes_no_many([p_filter(instr, r, st.projection) for r in chunk])
            out += [r for r, ok in zip(chunk, res) if ok]
            if len(out) >= limit:
                break
        return out[:limit]
    res = await llm.yes_no_many([p_filter(instr, r, st.projection) for r in order])
    out = [r for r, ok in zip(order, res) if ok]
    return out[:limit] if limit else out


async def run_join(llm, rows, limit: Optional[int], st: Strategy, seed: int):
    pairs = [(a, b) for a in rows for b in rows if a["reviewId"] != b["reviewId"]]
    if st.join_order == "random":
        random.Random(seed).shuffle(pairs)
    else:
        pairs.sort(key=lambda p: (p[0]["reviewId"], p[1]["reviewId"]))
    if st.proxy:  # likely same-sentiment pairs first; stable sort keeps grouping within ties
        pairs.sort(key=lambda p: -abs(proxy_score(p[0]["reviewText"]) + proxy_score(p[1]["reviewText"])))
    if st.limit_stop and limit:
        out, wave = [], max(min(st.wave, st.concurrency), 1)
        for i in range(0, len(pairs), wave):
            chunk = pairs[i:i + wave]
            res = await llm.yes_no_many([p_join(a, b, st.projection) for a, b in chunk])
            out += [p for p, ok in zip(chunk, res) if ok]
            if len(out) >= limit:
                break
        return out[:limit]
    res = await llm.yes_no_many([p_join(a, b, st.projection) for a, b in pairs])
    out = [p for p, ok in zip(pairs, res) if ok]
    return out[:limit] if limit else out


async def run_twopred(llm, rows, st: Strategy, sample_n: int = 40):
    preds = [POSITIVE, ACTING]
    if st.pred_order == "rank":
        # Estimate selectivity on a sample (these calls are part of the measured energy).
        sample = random.Random(7).sample(rows, min(sample_n, len(rows)))
        sel = []
        for instr in preds:
            res = await llm.yes_no_many([p_filter(instr, r, st.projection) for r in sample])
            sel.append(sum(res) / len(res))
        # Equal per-call cost assumed (same model, similar prompt length): rank = 1/(1-sel).
        preds = [p for _, p in sorted(zip([1 / max(1e-6, 1 - s) for s in sel], preds))]
    surv = list(rows)
    for instr in preds:
        res = await llm.yes_no_many([p_filter(instr, r, st.projection) for r in surv])
        surv = [r for r, ok in zip(surv, res) if ok]
    return surv


# ------------------------------------------------------------------ quality
def f1(pred: set, truth: set) -> Dict:
    tp = len(pred & truth)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(truth) if truth else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def quality(query: str, result, rows) -> Dict:
    if query.startswith("q1"):
        ids = {r["reviewId"] for r in result}
        truth = {r["reviewId"] for r in rows if r["label_positive"]}
        q = f1(ids, truth)
        return q if "full" in query else {"precision": q["precision"], "returned": len(ids)}
    if query.startswith("q5"):
        got = {(a["reviewId"], b["reviewId"]) for a, b in result}
        truth = {(a["reviewId"], b["reviewId"]) for a in rows for b in rows
                 if a["reviewId"] != b["reviewId"] and a["label_positive"] == b["label_positive"]}
        q = f1(got, truth)
        return q if "full" in query else {"precision": q["precision"], "returned": len(got)}
    return {"returned": len(result), "result_ids": sorted(r["reviewId"] for r in result)}


# ------------------------------------------------------------------ driver
def build_queries(reviews, smoke: bool, join_n: int, join_limit_n: int, n_rows: Optional[int]):
    rows = reviews[:n_rows] if n_rows else reviews
    am = [r for r in reviews if r["id"] == "ant_man_and_the_wasp_quantumania"]
    jn, jl = (6, 8) if smoke else (join_n, join_limit_n)
    q = {
        "q1_limit": (rows, 5),
        "q1_full": (rows, None),
        # SemBench Q5 joins all ~270 Ant-Man reviews (~73k pairs). Strategies without
        # pipelined LIMIT evaluate every pair, so the table is capped to keep A tractable.
        "q5_join_limit": (am[:jl], 10),
        "q5_join_full": (am[:jn], None),
        "q6_twopred_full": (rows, None),
    }
    return q


async def execute(llm, query, rows, limit, st, seed):
    if query.startswith("q1"):
        return await run_filter(llm, rows, POSITIVE, limit, st)
    if query.startswith("q5"):
        return await run_join(llm, rows, limit, st, seed)
    return await run_twopred(llm, rows, st)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--min-window", type=float, default=30.0)
    ap.add_argument("--queries", default="q1_limit,q1_full,q5_join_limit,q5_join_full,q6_twopred_full")
    ap.add_argument("--join-n", type=int, default=40, help="outer/inner size for the full self-join")
    ap.add_argument("--join-limit-n", type=int, default=80, help="table size for the LIMIT self-join")
    ap.add_argument("--rows", type=int, default=None, help="limit table size (debug)")
    ap.add_argument("--idle-seconds", type=float, default=30.0)
    ap.add_argument("--smoke", action="store_true", help="tiny inputs, 1 rep, no min window")
    return ap.parse_args()


async def amain(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reviews = load_reviews()
    queries = build_queries(reviews, args.smoke, args.join_n, args.join_limit_n, 60 if args.smoke else args.rows)
    if args.smoke:
        args.reps, args.min_window, args.idle_seconds = 1, 0.0, 1.0

    meter = EnergyMeter()
    llm = from_env()
    meta = {"model": llm.model, "base_url": llm.base_url, "energy_available": meter.available,
            "prefix_cache_reset_supported": llm.reset_prefix_cache(), "args": vars(args)}
    meta["idle"] = meter.idle_power(args.idle_seconds)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    rec_f = open(out / "records.jsonl", "a")
    for rep in range(args.reps):
        for query in args.queries.split(","):
            rows, limit = queries[query]
            strats = strategies_for(query)
            random.Random(1000 + rep).shuffle(strats)  # interleave to spread thermal drift
            for st in strats:
                llm.max_concurrency = st.concurrency
                llm.sem = asyncio.Semaphore(st.concurrency)
                llm.reset_usage()
                n_exec, result, s = 0, None, meter.start()
                t_first = time.perf_counter()
                while True:
                    llm.reset_prefix_cache()
                    result = await execute(llm, query, rows, limit, st, seed=rep)
                    n_exec += 1
                    if time.perf_counter() - t_first >= args.min_window:
                        break
                w = meter.stop(s)
                idle = meta["idle"]
                rec = {
                    "rep": rep, "query": query, "strategy": st.name, "lever_config": st.__dict__,
                    "executions_in_window": n_exec, "window": w.to_dict(),
                    "per_exec": {
                        "seconds": w.seconds / n_exec,
                        "gpu_j": None if w.gpu_j is None else w.gpu_j / n_exec,
                        "gpu_j_above_idle": None if w.gpu_j is None or idle["gpu_w"] is None
                        else (w.gpu_j - idle["gpu_w"] * w.seconds) / n_exec,
                        "cpu_pkg_j": None if w.cpu_pkg_j is None else w.cpu_pkg_j / n_exec,
                        **{k: v / n_exec for k, v in llm.usage.to_dict().items()},
                    },
                    "quality": quality(query, result, rows),
                }
                rec_f.write(json.dumps(rec) + "\n")
                rec_f.flush()
                pe = rec["per_exec"]
                print(f"rep{rep} {query:16s} {st.name:14s} calls={pe['calls']:7.0f} "
                      f"tok={pe['prompt_tokens']:9.0f} cached={pe['cached_prompt_tokens']:9.0f} "
                      f"t={pe['seconds']:7.2f}s gpuJ={pe['gpu_j']} q={ {k: v for k, v in rec['quality'].items() if k != 'result_ids'} }",
                      flush=True)
    rec_f.close()


def main():
    asyncio.run(amain(parse()))


if __name__ == "__main__":
    main()
