"""Summarize wave_load.py output: mean marginal GPU energy per execution and latency per (pred, limit, bg, wave).

  python -m probes.p2_plan_aware.analyze_wave_load <run dir>
"""
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

run = Path(sys.argv[1])
g = defaultdict(list)
for line in open(run / "records.jsonl"):
    r = json.loads(line)
    g[(r["pred"], r["limit"], r["bg"], r["wave"])].append(r)
rows = []
for (p, k, b, w), v in sorted(g.items()):
    m = [x["per_exec"]["gpu_j_marginal"] for x in v]
    rows.append({"pred": p, "limit": k, "bg": b, "wave": w, "n": len(v), "marginal_j": st.mean(m),
                 "marginal_sd": st.stdev(m) if len(m) > 1 else 0.0,
                 "latency_s": st.mean(x["per_exec"]["latency_s"] for x in v),
                 "calls": st.mean(x["per_exec"]["calls"] for x in v),
                 "bg_slowdown": st.mean(x["bg_slowdown"] for x in v),
                 "e_bg_j": st.mean(x["calib"]["e_bg_j"] for x in v)})
with open(run / "wave_load_summary.csv", "w") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
by = defaultdict(list)
for r in rows:
    by[(r["pred"], r["limit"], r["bg"])].append(r)
for key, v in by.items():
    best = min(v, key=lambda r: r["marginal_j"])
    print(f"{key[0]:8s} k={key[1]:3d} bg={key[2]:3d} best w={best['wave']:4d}  " + "  ".join(
        f"w{r['wave']}={r['marginal_j']:.0f}±{r['marginal_sd']:.0f}J/{r['latency_s']:.2f}s" for r in v))
