"""Summarise P2 records: per (query, strategy) means and std over reps, normalised to A
and to B, and the A->B (serving) vs B->C (DBMS knowledge) decomposition.

  python -m probes.p2_plan_aware.analyze_p2 probes/results/p2/<run>
Writes summary.csv and fig_p2.pdf next to records.jsonl.
"""

from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

METRICS = ["gpu_j", "gpu_j_above_idle", "cpu_pkg_j", "seconds", "calls", "prompt_tokens", "cached_prompt_tokens"]


def load(run: Path):
    return [json.loads(l) for l in open(run / "records.jsonl")]


def summarise(recs):
    g = defaultdict(list)
    for r in recs:
        g[(r["query"], r["strategy"])].append(r)
    rows = []
    for (q, s), rs in sorted(g.items()):
        row = {"query": q, "strategy": s, "n": len(rs)}
        for m in METRICS:
            v = [r["per_exec"][m] for r in rs if r["per_exec"].get(m) is not None]
            row[m] = st.mean(v) if v else None
            row[m + "_sd"] = st.stdev(v) if len(v) > 1 else 0.0 if v else None
        for k in ("f1", "precision"):
            v = [r["quality"][k] for r in rs if k in r["quality"]]
            row[k] = st.mean(v) if v else None
        rows.append(row)
    # normalise energy (GPU if measured, else wall time) to A and B of the same query
    key = "gpu_j" if any(r["gpu_j"] for r in rows) else "seconds"
    by_q = defaultdict(dict)
    for r in rows:
        by_q[r["query"]][r["strategy"]] = r
    for r in rows:
        a, b = by_q[r["query"]].get("A"), by_q[r["query"]].get("B")
        r["metric"] = key
        r["vs_A"] = r[key] / a[key] if a and a[key] else None
        r["vs_B"] = r[key] / b[key] if b and b[key] else None
    return rows, key


def plot(rows, key, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    queries = sorted({r["query"] for r in rows})
    fig, axes = plt.subplots(1, len(queries), figsize=(3.2 * len(queries), 3), squeeze=False)
    for ax, q in zip(axes[0], queries):
        rs = [r for r in rows if r["query"] == q]
        order = sorted(rs, key=lambda r: ({"A": 0, "B": 1, "B_nested": 2}.get(r["strategy"], 3), r["strategy"]))
        names = [r["strategy"] for r in order]
        vals = [r["vs_A"] or 0 for r in order]
        cols = ["#888780" if n == "A" else "#378ADD" if n.startswith("B") else "#1D9E75" for n in names]
        ax.bar(range(len(names)), vals, color=cols)
        ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=7)
        ax.set_title(q, fontsize=9)
        ax.set_ylabel(f"{key} / A" if ax is axes[0][0] else "")
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out / "fig_p2.pdf")


def main():
    run = Path(sys.argv[1])
    rows, key = summarise(load(run))
    with open(run / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot(rows, key, run)
    print(f"metric={key}")
    for r in rows:
        print(f"{r['query']:16s} {r['strategy']:14s} vs_A={r['vs_A'] or 0:6.3f} vs_B={r['vs_B'] or 0:6.3f} "
              f"calls={r['calls']:8.0f} tok={r['prompt_tokens']:9.0f} f1={r['f1']} prec={r['precision']}")


if __name__ == "__main__":
    main()
