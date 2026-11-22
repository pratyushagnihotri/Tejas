"""P2b: wave-size sweep for pipelined LIMIT — does the best batch size need both layers' knowledge?

A semantic filter with LIMIT k is evaluated in waves of w concurrent requests; after each wave the
DBMS counts matches and stops once it has k. Small w wastes few calls but batches poorly; large w
batches well but overshoots. We sweep w and vary
  DBMS-side knowledge:    predicate selectivity (POSITIVE ~ high, ACTING ~ low) and LIMIT k (5, 50)
  serving-side knowledge: background load on the same server (0 or BG concurrent requests)
and report energy, latency, calls and wasted calls per execution. If the energy-optimal w moves with
both, neither layer can choose it alone.

Energy under load is *marginal*: window energy minus background-only power x window time, with the
background power calibrated right before each rep. Without load, total and above-idle are reported.
Prompts use the projected attribute only (as C_all); prefix cache is reset before every execution.
Background requests carry a unique nonce at the start, so they never share cached prefixes.

  python -m probes.p2_plan_aware.wave_sweep --out probes/results/p2_wave/<run>
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random
import time
from pathlib import Path

from probes.common.data import load_reviews
from probes.common.energy import EnergyMeter
from probes.common.llm import from_env
from probes.p2_plan_aware.run_p2 import ACTING, POSITIVE, p_filter

PREDS = {"positive": POSITIVE, "acting": ACTING}
BG_INSTR = "The following movie review mentions the director by name."


class Background:
    """Keeps `n` requests in flight against the same server until stopped."""

    def __init__(self, n: int, rows):
        self.n, self.rows = n, rows
        self.llm = from_env(concurrency=max(n, 1))
        self.stop_ev = asyncio.Event()
        self.tasks = []
        self.ctr = itertools.count()

    async def _worker(self, wid: int):
        rng = random.Random(wid)
        while not self.stop_ev.is_set():
            r = rng.choice(self.rows)
            await self.llm.yes_no(f"[request {next(self.ctr)}]\n" + p_filter(BG_INSTR, r, "full_row"))

    async def start(self):
        if self.n:
            self.stop_ev.clear()
            self.tasks = [asyncio.create_task(self._worker(i)) for i in range(self.n)]
            await asyncio.sleep(5)  # reach steady state

    async def stop(self):
        self.stop_ev.set()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks = []


async def limit_filter(llm, order, instr, k, wave):
    out, calls = [], 0
    for i in range(0, len(order), wave):
        chunk = order[i:i + wave]
        res = await llm.yes_no_many([p_filter(instr, r, "needed") for r in chunk])
        calls += len(chunk)
        for r, ok in zip(chunk, res):
            if ok:
                out.append((i + chunk.index(r), r))
        if len(out) >= k:
            break
    out.sort(key=lambda t: t[0])
    needed = out[k - 1][0] + 1 if len(out) >= k else calls  # calls a perfect (sequential) plan needs
    return [r for _, r in out[:k]], calls, needed


async def amain(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_reviews()
    meter = EnergyMeter()
    llm = from_env(concurrency=max(a.waves))
    meta = {"model": llm.model, "energy_available": meter.available, "args": vars(a),
            "prefix_cache_reset_supported": llm.reset_prefix_cache()}
    meta["idle"] = meter.idle_power(a.idle_seconds)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    rec_f = open(out / "records.jsonl", "a")

    for rep in range(a.reps):
        order = list(rows)
        random.Random(rep).shuffle(order)  # same tuple order for every wave size within a rep
        for bg_n in a.bg:
            bg = Background(bg_n, rows)
            await bg.start()
            if bg_n:
                s = meter.start()
                await asyncio.sleep(a.bg_calib)
                w = meter.stop(s)
                p_bg = None if w.gpu_j is None else w.gpu_j / w.seconds
            else:
                p_bg = meta["idle"]["gpu_w"]
            configs = [(p, k, wv) for p in a.preds for k in a.limits for wv in a.waves]
            random.Random(100 + rep).shuffle(configs)
            for pred, k, wave in configs:
                llm.reset_usage()
                n_exec, s, t0 = 0, meter.start(), time.perf_counter()
                lat, needed = [], 0
                while True:
                    llm.reset_prefix_cache()
                    t1 = time.perf_counter()
                    res, calls, needed = await limit_filter(llm, order, PREDS[pred], k, wave)
                    lat.append(time.perf_counter() - t1)
                    n_exec += 1
                    if time.perf_counter() - t0 >= a.min_window:
                        break
                w = meter.stop(s)
                gpu = w.gpu_j
                rec = {
                    "rep": rep, "pred": pred, "limit": k, "wave": wave, "bg": bg_n,
                    "executions_in_window": n_exec, "window": w.to_dict(), "bg_power_w": p_bg,
                    "per_exec": {
                        "latency_s": sum(lat) / len(lat),
                        "gpu_j": None if gpu is None else gpu / n_exec,
                        "gpu_j_marginal": None if gpu is None or p_bg is None else (gpu - p_bg * w.seconds) / n_exec,
                        "calls": llm.usage.calls / n_exec,
                        "needed_calls": needed,
                        "wasted_calls": llm.usage.calls / n_exec - needed,
                        "prompt_tokens": llm.usage.prompt_tokens / n_exec,
                    },
                    "returned": len(res),
                    "precision_vs_label": (sum(r["label_positive"] for r in res) / len(res))
                    if pred == "positive" and res else None,
                }
                rec_f.write(json.dumps(rec) + "\n")
                rec_f.flush()
                pe = rec["per_exec"]
                print(f"rep{rep} bg={bg_n:3d} {pred:8s} k={k:3d} w={wave:4d} calls={pe['calls']:6.1f} "
                      f"wasted={pe['wasted_calls']:6.1f} lat={pe['latency_s']:6.2f}s "
                      f"gpuJ={pe['gpu_j'] if pe['gpu_j'] is None else round(pe['gpu_j'], 1)} "
                      f"marg={pe['gpu_j_marginal'] if pe['gpu_j_marginal'] is None else round(pe['gpu_j_marginal'], 1)}",
                      flush=True)
            await bg.stop()
    rec_f.close()


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--waves", type=lambda s: [int(x) for x in s.split(",")], default=[1, 4, 8, 16, 64, 256])
    ap.add_argument("--limits", type=lambda s: [int(x) for x in s.split(",")], default=[5, 50])
    ap.add_argument("--preds", type=lambda s: s.split(","), default=["positive", "acting"])
    ap.add_argument("--bg", type=lambda s: [int(x) for x in s.split(",")], default=[0, 32])
    ap.add_argument("--min-window", type=float, default=30.0)
    ap.add_argument("--bg-calib", type=float, default=30.0)
    ap.add_argument("--idle-seconds", type=float, default=60.0)
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(amain(parse()))
