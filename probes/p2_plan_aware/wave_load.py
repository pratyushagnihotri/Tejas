"""P2c: wave-size sweep under shared load, with work-based energy attribution.

The first sweep (wave_sweep.py) attributed energy under load as window energy minus background power x time.
That assumes the background does the same work with and without the foreground query; it does not (the query
displaces background requests from shared batches), so marginal energy came out negative.

Here each configuration is measured as a pair of windows on a server with a closed-loop background of
`bg` concurrent requests:
  1. background alone for `calib` seconds  -> e_bg = (E - P_idle*T) / requests completed   (J per bg request)
  2. background + foreground LIMIT query, repeated until `min_window` seconds have passed
     -> marginal per execution = [(E - P_idle*T) - e_bg * bg requests completed] / executions
i.e. the foreground is charged everything except the background work actually done, priced at its
stand-alone rate. Background requests draw rows at random and carry a unique nonce (no shared prefixes).

  python -m probes.p2_plan_aware.wave_load --out probes/results/p2_wave_load/<run>
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
from probes.p2_plan_aware.run_p2 import p_filter
from probes.p2_plan_aware.wave_sweep import BG_INSTR, PREDS, limit_filter


class CountingBackground:
    """Closed loop of `n` workers; counts requests completed (and their prompt tokens) since `mark()`."""

    def __init__(self, n: int, rows):
        self.n, self.rows = n, rows
        self.llm = from_env(concurrency=n)
        self.stop_ev = asyncio.Event()
        self.tasks = []
        self.ctr = itertools.count()
        self.done = 0
        self.done_tokens = 0

    async def _worker(self, wid: int):
        rng = random.Random(wid)
        while not self.stop_ev.is_set():
            r = rng.choice(self.rows)
            p = f"[request {next(self.ctr)}]\n" + p_filter(BG_INSTR, r, "full_row")
            await self.llm.yes_no(p)
            self.done += 1
            self.done_tokens += len(p) // 4  # rough; the analysis uses request counts

    def mark(self):
        return self.done, self.done_tokens

    async def start(self):
        self.stop_ev.clear()
        self.tasks = [asyncio.create_task(self._worker(i)) for i in range(self.n)]
        await asyncio.sleep(10)  # reach steady state

    async def stop(self):
        self.stop_ev.set()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks = []


async def amain(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_reviews()
    meter = EnergyMeter()
    llm = from_env(concurrency=max(a.waves))
    meta = {"model": llm.model, "energy_available": meter.available, "args": vars(a),
            "method": "work-based attribution (paired calibration window per configuration)"}
    meta["idle"] = meter.idle_power(a.idle_seconds)
    p_idle = meta["idle"]["gpu_w"]
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    rec_f = open(out / "records.jsonl", "a")

    for rep in range(a.reps):
        order = list(rows)
        random.Random(rep).shuffle(order)
        for bg_n in a.bg:
            bg = CountingBackground(bg_n, rows)
            await bg.start()
            configs = [(p, k, w) for p in a.preds for k in a.limits for w in a.waves]
            random.Random(200 + rep).shuffle(configs)
            for pred, k, wave in configs:
                # 1) background alone
                c0, s = bg.mark(), meter.start()
                await asyncio.sleep(a.calib)
                wc = meter.stop(s)
                c1 = bg.mark()
                n_cal = c1[0] - c0[0]
                e_bg = (wc.gpu_j - p_idle * wc.seconds) / max(n_cal, 1)
                # 2) background + foreground
                llm.reset_usage()
                c2, s, t0 = bg.mark(), meter.start(), time.perf_counter()
                n_exec, lat, needed = 0, [], 0
                while True:
                    llm.reset_prefix_cache()  # as in wave_sweep: executions must not share cached prefixes
                    t1 = time.perf_counter()
                    res, calls, needed = await limit_filter(llm, order, PREDS[pred], k, wave)
                    lat.append(time.perf_counter() - t1)
                    n_exec += 1
                    if time.perf_counter() - t0 >= a.min_window:
                        break
                w = meter.stop(s)
                c3 = bg.mark()
                n_bg = c3[0] - c2[0]
                active = w.gpu_j - p_idle * w.seconds
                marg = (active - e_bg * n_bg) / n_exec
                bg_rate_alone = n_cal / wc.seconds
                bg_rate_with = n_bg / w.seconds
                rec = {
                    "rep": rep, "pred": pred, "limit": k, "wave": wave, "bg": bg_n,
                    "calib": {"window": wc.to_dict(), "bg_done": n_cal, "e_bg_j": e_bg},
                    "window": w.to_dict(), "bg_done": n_bg, "executions_in_window": n_exec,
                    "bg_slowdown": 1 - bg_rate_with / bg_rate_alone if bg_rate_alone else None,
                    "per_exec": {
                        "latency_s": sum(lat) / len(lat),
                        "gpu_j_marginal": marg,
                        "calls": llm.usage.calls / n_exec,
                        "needed_calls": needed,
                        "wasted_calls": llm.usage.calls / n_exec - needed,
                    },
                }
                rec_f.write(json.dumps(rec) + "\n")
                rec_f.flush()
                pe = rec["per_exec"]
                print(f"rep{rep} bg={bg_n:3d} {pred:8s} k={k:3d} w={wave:4d} calls={pe['calls']:6.1f} "
                      f"lat={pe['latency_s']:6.2f}s marg={marg:8.1f}J e_bg={e_bg:6.2f}J "
                      f"bg_slowdown={rec['bg_slowdown']:.2f} n_exec={n_exec}", flush=True)
            await bg.stop()
    rec_f.close()


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--waves", type=lambda s: [int(x) for x in s.split(",")], default=[1, 4, 16, 64, 256])
    ap.add_argument("--limits", type=lambda s: [int(x) for x in s.split(",")], default=[5, 50])
    ap.add_argument("--preds", type=lambda s: s.split(","), default=["positive", "acting"])
    ap.add_argument("--bg", type=lambda s: [int(x) for x in s.split(",")], default=[32])
    ap.add_argument("--min-window", type=float, default=30.0)
    ap.add_argument("--calib", type=float, default=20.0)
    ap.add_argument("--idle-seconds", type=float, default=60.0)
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(amain(parse()))
