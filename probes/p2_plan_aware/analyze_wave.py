"""Summarise the wave-size sweep: mean per (pred, limit, bg, wave) and the energy-optimal wave per setting.

  python -m probes.p2_plan_aware.analyze_wave probes/results/p2_wave/<run>

Energy metric: gpu_j_marginal (above idle without load, above background power with load);
falls back to latency when no energy counter was available.
"""

from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def main():
    run = Path(sys.argv[1])
    recs = [json.loads(l) for l in open(run / "records.jsonl")]
    use_e = recs[0]["per_exec"]["gpu_j_marginal"] is not None
    metric = "gpu_j_marginal" if use_e else "latency_s"
    g = defaultdict(list)
    for r in recs:
        g[(r["pred"], r["limit"], r["bg"], r["wave"])].append(r["per_exec"])
    rows = []
    for (p, k, bg, w), xs in sorted(g.items()):
        m = lambda f: st.mean(x[f] for x in xs if x[f] is not None) if any(x[f] is not None for x in xs) else None
        sd = lambda f: st.stdev([x[f] for x in xs]) if len(xs) > 1 and all(x[f] is not None for x in xs) else 0.0
        rows.append({"pred": p, "limit": k, "bg": bg, "wave": w, "n": len(xs),
                     "energy_j": m(metric), "energy_sd": sd(metric), "total_gpu_j": m("gpu_j"),
                     "latency_s": m("latency_s"), "calls": m("calls"), "needed": m("needed_calls"),
                     "wasted": m("wasted_calls")})
    with open(run / "wave_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"metric={metric}")
    best = {}
    for key in sorted({(r["pred"], r["limit"], r["bg"]) for r in rows}):
        rs = [r for r in rows if (r["pred"], r["limit"], r["bg"]) == key]
        b = min(rs, key=lambda r: r["energy_j"])
        best[key] = b["wave"]
        u = "J" if use_e else "s"
        curve = "  ".join(f"w{r['wave']}={r['energy_j']:.1f}{u}/{r['latency_s']:.2f}s" for r in rs)
        print(f"{key[0]:8s} k={key[1]:3d} bg={key[2]:3d}  best w={b['wave']:4d} "
              f"(worst/best {max(r['energy_j'] for r in rs) / max(b['energy_j'], 1e-9):.1f}x)  {curve}")
    (run / "wave_best.json").write_text(json.dumps({f"{p}|k={k}|bg={bg}": w for (p, k, bg), w in best.items()}, indent=2))


if __name__ == "__main__":
    main()
