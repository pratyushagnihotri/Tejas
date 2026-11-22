"""P4: can flexible semantic maintenance exploit time-varying energy conditions?

Trace-driven simulation. Jobs are semantic maintenance tasks (re-running a semantic
filter or classification over a table, refreshing a semantic view, re-embedding), each
with a release time, a deadline (freshness bound), a measured energy and duration.
Job energy/duration should come from P2/P3 measurements (--job-energy-wh/--job-hours).

Policies (one GPU, jobs do not overlap):
  immediate        run each job as soon as it is released (today's behaviour)
  deadline_aware   EDF-like: place each job in the feasible slot [release, deadline-duration]
                   with the lowest mean carbon intensity (or price), respecting the GPU timeline

Trace CSV columns: timestamp (ISO, hourly), carbon_g_per_kwh, price_eur_per_mwh.
Real traces: Energy-Charts / ENTSO-E / Electricity Maps (Germany). For code testing only:
--synthetic writes a clearly labelled synthetic trace.

  python -m probes.p4_temporal.simulate_p4 --trace de_2025.csv --out probes/results/p4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List


def load_trace(path: Path) -> List[Dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [{"t": datetime.fromisoformat(r["timestamp"]), "carbon": float(r["carbon_g_per_kwh"]),
             "price": float(r.get("price_eur_per_mwh") or "nan")} for r in rows]


def synthetic_trace(days: int, seed: int = 0) -> List[Dict]:
    """NOT REAL DATA: diurnal solar dip + noise, only for testing the simulator."""
    rng, t0, out = random.Random(seed), datetime(2025, 6, 1), []
    for h in range(days * 24):
        hod = h % 24
        solar = max(0.0, math.sin(math.pi * (hod - 6) / 12))
        c = 450 - 250 * solar + rng.gauss(0, 30)
        out.append({"t": t0 + timedelta(hours=h), "carbon": max(50, c), "price": max(-20, 0.25 * c + rng.gauss(0, 10))})
    return out


def make_jobs(trace, n_per_day, slack_h, energy_wh, hours, seed=1):
    rng = random.Random(seed)
    days = len(trace) // 24 - math.ceil((slack_h + hours) / 24) - 1
    jobs = []
    for d in range(days):
        for _ in range(n_per_day):
            rel = d * 24 + rng.randrange(24)
            jobs.append({"release": rel, "deadline": rel + slack_h + hours, "hours": hours, "energy_wh": energy_wh})
    return sorted(jobs, key=lambda j: j["release"])


def cost(trace, start, hours, energy_wh, key):
    """Energy split evenly over the job's hours; returns g CO2e (carbon) or EUR (price)."""
    total, per_h = 0.0, energy_wh / 1000.0 / hours  # kWh per hour
    h, rem = start, hours
    while rem > 1e-9:
        frac = min(1.0, rem, math.floor(h) + 1 - h)
        v = trace[int(h)][key]
        total += per_h * frac * (v if key == "carbon" else v / 1000.0)
        h += frac
        rem -= frac
    return total


SLOT = 0.25  # hours


def simulate(trace, jobs, policy, key):
    """immediate: FIFO, as soon as the GPU is free. deadline_aware: jobs in deadline order,
    each placed in the cheapest free slot window inside [release, deadline - duration];
    if none is free, the earliest free window after release (counted as a miss)."""
    horizon = int(len(trace) / SLOT)
    busy = [False] * horizon
    placed = {}

    def free(s0, k):
        return s0 + k <= horizon and not any(busy[s0:s0 + k])

    order = jobs if policy == "immediate" else sorted(jobs, key=lambda j: j["deadline"])
    for j in order:
        k = int(math.ceil(j["hours"] / SLOT))
        r = int(math.ceil(j["release"] / SLOT))
        if policy == "immediate":
            s0 = next(s for s in range(r, horizon) if free(s, k))
        else:
            last = int((j["deadline"] - j["hours"]) / SLOT)
            cands = [s for s in range(r, last + 1) if free(s, k)]
            s0 = (min(cands, key=lambda s: cost(trace, s * SLOT, j["hours"], j["energy_wh"], key)) if cands
                  else next(s for s in range(r, horizon) if free(s, k)))
        for x in range(s0, s0 + k):
            busy[x] = True
        placed[id(j)] = s0 * SLOT
    res = []
    for j in jobs:
        start = placed[id(j)]
        res.append({"start": start, "delay_h": start - j["release"], "missed": start + j["hours"] > j["deadline"] + 1e-9,
                    "carbon_g": cost(trace, start, j["hours"], j["energy_wh"], "carbon"),
                    "price_eur": cost(trace, start, j["hours"], j["energy_wh"], "price")})
    n = len(res)
    return {"jobs": n, "carbon_kg": sum(r["carbon_g"] for r in res) / 1000, "price_eur": sum(r["price_eur"] for r in res),
            "mean_delay_h": sum(r["delay_h"] for r in res) / n, "deadline_misses": sum(r["missed"] for r in res)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace")
    ap.add_argument("--synthetic", type=int, default=0, help="days of synthetic trace (testing only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs-per-day", type=int, default=6)
    ap.add_argument("--job-energy-wh", type=float, default=100.0, help="use measured P2/P3 energy")
    ap.add_argument("--job-hours", type=float, default=0.5)
    ap.add_argument("--slacks", default="0,2,4,8,12,24")
    ap.add_argument("--objective", default="carbon", choices=["carbon", "price"])
    a = ap.parse_args()
    trace = synthetic_trace(a.synthetic) if a.synthetic else load_trace(Path(a.trace))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in [float(x) for x in a.slacks.split(",")]:
        jobs = make_jobs(trace, a.jobs_per_day, s, a.job_energy_wh, a.job_hours)
        base = simulate(trace, jobs, "immediate", a.objective)
        for pol in ["immediate", "deadline_aware"]:
            r = base if pol == "immediate" else simulate(trace, jobs, pol, a.objective)
            r.update(slack_h=s, policy=pol, carbon_vs_immediate=r["carbon_kg"] / base["carbon_kg"],
                     price_vs_immediate=r["price_eur"] / base["price_eur"] if base["price_eur"] else None)
            rows.append(r)
            print(f"slack={s:5.1f}h {pol:15s} carbon={r['carbon_kg']:8.2f}kg ({r['carbon_vs_immediate']:.2f}x) "
                  f"price={r['price_eur']:8.2f}EUR delay={r['mean_delay_h']:5.2f}h misses={r['deadline_misses']}")
    meta = {"trace": a.trace or f"SYNTHETIC ({a.synthetic} days) - not for the paper", **vars(a)}
    (out / "p4.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
