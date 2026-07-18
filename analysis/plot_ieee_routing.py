"""IEEE-style paper figures for the ORIGINAL energy-aware routing system
(static-cost-router / scarce / scarce, calibrated — the hand-designed
policies from docs/RESEARCH_LOG.md Findings 1-3, predating the learned
cost model). For the learned cost model's figures, see
plot_ieee_cost_model.py.

Requires the tejas package installed (`pip install -e .` from the repo
root). Run after tejas.routing.static_policies has produced
results/policy_summary*.csv:
    .venv/bin/python3 analysis/plot_ieee_routing.py

Figures written to results/figures/paper/routing/.
"""

import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogLocator, NullFormatter

from tejas.benchmarking.results_loader import load_raw_results
from tejas.paths import RESULTS_DIR as _RESULTS_DIR

from ieee_style import FIGSIZE, MODEL_COLORS, MODEL_TIER_ORDER, apply, save
from plot_results import DATASET_LABELS, DATASET_ORDER, _pareto_points, dataset_of, load_filtered_rows

RESULTS_DIR = str(_RESULTS_DIR)
FIG_DIR = os.path.join(RESULTS_DIR, "figures", "paper", "routing")

EXPERIMENTS = [
    ("mac", "Mac\n(3b/7b/14b)", os.path.join(RESULTS_DIR, "raw_results.csv"),
     os.path.join(RESULTS_DIR, "policy_summary.csv")),
    ("dgx_qwen", "DGX-Qwen\n(3b/7b/32b)", os.path.join(RESULTS_DIR, "raw_results_dgx_qwen.csv"),
     os.path.join(RESULTS_DIR, "policy_summary_dgx_qwen.csv")),
    ("dgx_llama", "DGX-Llama\n(3b/8b/70b)", os.path.join(RESULTS_DIR, "raw_results_dgx_llama.csv"),
     os.path.join(RESULTS_DIR, "policy_summary_dgx_llama.csv")),
]

TASK_TIER_ORDER = ["easy", "medium", "hard"]
TASK_TIER_MARKERS = {"easy": "o", "medium": "s", "hard": "^"}

POLICIES = ["static-cost-router", "energy-aware (scarce)", "energy-aware (scarce, calibrated)"]
POLICY_LABELS = {"static-cost-router": "static-cost-router",
                  "energy-aware (scarce)": "scarce",
                  "energy-aware (scarce, calibrated)": "scarce, calibrated"}
POLICY_COLORS = {"static-cost-router": "#eda100", "energy-aware (scarce)": "#e34948",
                  "energy-aware (scarce, calibrated)": "#eb6834"}


def fig_tier_pareto():
    """Quality vs. energy by model size x task difficulty, one panel per
    experiment — motivates "not every task needs the largest model" (the
    premise the whole routing system rests on) with real measured data
    across all three capability gaps."""
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE["row3"])
    for ax, (key, label, raw_path, _) in zip(axes, EXPERIMENTS):
        by_run_task_model, task_tier, runs = load_raw_results(raw_path)
        groups = {}
        for (run, task_id, model_tier), row in by_run_task_model.items():
            tier = task_tier[task_id]
            g = groups.setdefault((model_tier, tier), {"n": 0, "correct": 0, "energy": []})
            g["n"] += 1
            g["correct"] += 1 if row["correct"] == "True" else 0
            if row["energy_j"]:
                g["energy"].append(float(row["energy_j"]))

        for model_tier in MODEL_TIER_ORDER:
            xs, ys = [], []
            for tier in TASK_TIER_ORDER:
                g = groups.get((model_tier, tier))
                if not g or not g["energy"]:
                    continue
                avg_energy = sum(g["energy"]) / len(g["energy"])
                avg_quality = 100 * g["correct"] / g["n"]
                xs.append(avg_energy)
                ys.append(avg_quality)
                ax.scatter(avg_energy, avg_quality, s=70, color=MODEL_COLORS[model_tier],
                           marker=TASK_TIER_MARKERS[tier], edgecolor="white",
                           linewidth=1.0, zorder=3)
            if len(xs) > 1:
                order = sorted(range(len(xs)), key=lambda i: xs[i])
                ax.plot([xs[i] for i in order], [ys[i] for i in order],
                        color=MODEL_COLORS[model_tier], alpha=0.35, linewidth=1.4, zorder=2)

        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=5))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Avg. energy / request (J)")
        ax.set_title(label.replace("\n", " "))
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("Accuracy (%)")

    model_handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
                                 color=MODEL_COLORS[m], label=m.capitalize())
                      for m in MODEL_TIER_ORDER]
    tier_handles = [plt.Line2D([0], [0], marker=TASK_TIER_MARKERS[t], linestyle="",
                                markersize=8, color="gray", label=t.capitalize())
                     for t in TASK_TIER_ORDER]
    fig.legend(handles=model_handles + tier_handles, ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, 1.06), frameon=False, fontsize=10)
    fig.suptitle("Quality vs. energy by model size and task difficulty", y=1.14, fontsize=13)
    save(fig, FIG_DIR, "fig_routing_tier_pareto")


def fig_policy_comparison():
    """The core original finding (Findings 1-3): static-cost-router vs.
    scarce vs. scarce-calibrated, quality and energy, across all three
    capability gaps."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["row2"])
    x = range(len(EXPERIMENTS))
    width = 0.25

    for i, policy in enumerate(POLICIES):
        q_vals, q_stds, e_vals, e_stds = [], [], [], []
        for _, _, _, summary_path in EXPERIMENTS:
            df = pd.read_csv(summary_path).set_index("policy")
            q_vals.append(df.loc[policy, "quality_pct_mean"])
            q_stds.append(df.loc[policy, "quality_pct_std"])
            e_vals.append(df.loc[policy, "total_energy_j_mean"])
            e_stds.append(df.loc[policy, "total_energy_j_std"])
        offset = (i - 1) * width
        xpos = [xi + offset for xi in x]
        axes[0].bar(xpos, q_vals, width=width, yerr=q_stds, capsize=3,
                    color=POLICY_COLORS[policy], label=POLICY_LABELS[policy],
                    edgecolor="white", error_kw={"linewidth": 1})
        axes[1].bar(xpos, e_vals, width=width, yerr=e_stds, capsize=3,
                    color=POLICY_COLORS[policy], label=POLICY_LABELS[policy],
                    edgecolor="white", error_kw={"linewidth": 1})

    axes[0].set_ylabel("Quality (%)")
    axes[0].set_title("Quality")
    axes[1].set_ylabel("Total energy (J, log scale)")
    axes[1].set_yscale("log")
    axes[1].set_title("Energy")
    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels([label for _, label, _, _ in EXPERIMENTS])
    axes[0].legend(loc="lower right", fontsize=9.5)
    fig.suptitle("Hand-designed routing policies across capability gaps", fontsize=13)
    save(fig, FIG_DIR, "fig_routing_policy_comparison")


def fig_metrics_grid(key, label, raw_path):
    """Quality / energy / latency by model tier, one panel per dataset —
    consolidates the original fig3_metrics_{dataset} (3 separate files per
    experiment) into one 3x3 grid per experiment: same information, fewer
    near-duplicate files."""
    rows = load_filtered_rows(raw_path)
    fig, axes = plt.subplots(3, 3, figsize=(FIGSIZE["row3"][0], FIGSIZE["row3"][1] * 2.1))
    metric_specs = [("quality", "Accuracy (%)"), ("energy", "Avg. energy / req. (J)"),
                    ("latency", "Avg. latency / req. (s)")]

    for col, ds in enumerate(DATASET_ORDER):
        ds_rows = [r for r in rows if dataset_of(r["task_id"]) == ds]
        if not ds_rows:
            continue
        groups = defaultdict(lambda: {"correct": 0, "n": 0, "energy": [], "latency": []})
        for r in ds_rows:
            g = groups[r["model_tier"]]
            g["n"] += 1
            g["correct"] += 1 if r["correct"] == "True" else 0
            if r["energy_j"]:
                g["energy"].append(float(r["energy_j"]))
            g["latency"].append(float(r["latency_s"]))

        tiers = [t for t in MODEL_TIER_ORDER if groups[t]["n"] > 0]
        colors = [MODEL_COLORS[t] for t in tiers]
        values = {
            "quality": [100 * groups[t]["correct"] / groups[t]["n"] for t in tiers],
            "energy": [sum(groups[t]["energy"]) / len(groups[t]["energy"]) if groups[t]["energy"] else 0 for t in tiers],
            "latency": [sum(groups[t]["latency"]) / len(groups[t]["latency"]) for t in tiers],
        }
        for row_i, (metric, ylabel) in enumerate(metric_specs):
            ax = axes[row_i, col]
            ax.bar(range(len(tiers)), values[metric], color=colors, width=0.55, edgecolor="white")
            ax.set_xticks(range(len(tiers)))
            ax.set_xticklabels([t.capitalize() for t in tiers])
            if metric == "quality":
                ax.set_ylim(0, 100)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row_i == 0:
                ax.set_title(DATASET_LABELS[ds], fontsize=11)

    fig.suptitle(f"{label.replace(chr(10), ' ')} — metrics by model size and dataset", fontsize=13)
    fig.tight_layout()
    save(fig, FIG_DIR, f"fig_routing_metrics_grid_{key}")


def fig_pareto_dataset_comparison(key, label, raw_path):
    """Quality-vs-energy Pareto frontier, one panel per dataset, shared
    y-axis — direct IEEE port of the original fig5_pareto_comparison."""
    rows = load_filtered_rows(raw_path)
    present_datasets = [ds for ds in DATASET_ORDER if any(dataset_of(r["task_id"]) == ds for r in rows)]
    fig, axes = plt.subplots(1, len(present_datasets), figsize=FIGSIZE["row3"], sharey=True)
    if len(present_datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, present_datasets):
        ds_rows = [r for r in rows if dataset_of(r["task_id"]) == ds]
        points = _pareto_points(ds_rows)
        present = [t for t in MODEL_TIER_ORDER if t in points]
        xs = [points[t][0] for t in present]
        ys = [points[t][1] for t in present]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ax.plot([xs[i] for i in order], [ys[i] for i in order], color="#c3c2b7", linewidth=1.5, zorder=2)
        for t in present:
            x, y = points[t]
            ax.scatter(x, y, s=110, color=MODEL_COLORS[t], edgecolor="white", linewidth=1.2, zorder=3)
        ax.set_xlabel("Avg. energy / req. (J)")
        ax.set_title(DATASET_LABELS[ds], fontsize=11)

    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 100)
    model_handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=9,
                                 color=MODEL_COLORS[m], label=m.capitalize())
                      for m in MODEL_TIER_ORDER]
    fig.legend(handles=model_handles, title="Model size", loc="upper center",
               bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=10)
    fig.suptitle(f"Quality-vs-energy frontier by dataset — {label.replace(chr(10), ' ')}", y=1.2, fontsize=13)
    save(fig, FIG_DIR, f"fig_routing_pareto_by_dataset_{key}")


SEQ_BLUE = ["#f9f9f7", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#0d366b"]


def fig_query_heatmap(key, label, raw_path):
    """Per-task_id x model_tier breakdown — accuracy and (per-column
    normalized) energy. IEEE port of the original fig6_query_heatmap."""
    rows = load_filtered_rows(raw_path)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

    acc = defaultdict(lambda: [0, 0])
    energy = defaultdict(list)
    task_tier_of = {}
    for r in rows:
        k = (r["task_id"], r["model_tier"])
        acc[k][1] += 1
        if r["correct"] == "True":
            acc[k][0] += 1
        if r["energy_j"]:
            energy[k].append(float(r["energy_j"]))
        task_tier_of[r["task_id"]] = r["task_tier"]

    tier_rank = {"easy": 0, "medium": 1, "hard": 2}
    task_ids = sorted(task_tier_of, key=lambda t: (tier_rank[task_tier_of[t]], dataset_of(t), t))
    n = len(task_ids)

    q_grid = [[100 * acc[(t, mt)][0] / acc[(t, mt)][1] if acc[(t, mt)][1] else float("nan")
               for mt in MODEL_TIER_ORDER] for t in task_ids]
    e_grid = [[sum(energy[(t, mt)]) / len(energy[(t, mt)]) if energy[(t, mt)] else float("nan")
               for mt in MODEL_TIER_ORDER] for t in task_ids]
    e_norm = [[float("nan")] * len(MODEL_TIER_ORDER) for _ in range(n)]
    for j in range(len(MODEL_TIER_ORDER)):
        col = [e_grid[i][j] for i in range(n) if e_grid[i][j] == e_grid[i][j]]
        if not col:
            continue
        lo, hi = min(col), max(col)
        for i in range(n):
            v = e_grid[i][j]
            if v == v:
                e_norm[i][j] = (v - lo) / (hi - lo) if hi > lo else 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, max(6.5, n * 0.24)))
    ax1.imshow(q_grid, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax2.imshow(e_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for ax, grid, norm_grid in [(ax1, q_grid, [[v / 100 for v in row] for row in q_grid]),
                                 (ax2, e_grid, e_norm)]:
        ax.set_xticks(range(len(MODEL_TIER_ORDER)))
        ax.set_xticklabels([t.capitalize() for t in MODEL_TIER_ORDER])
        ax.set_yticks(range(n))
        ax.set_yticklabels(task_ids, fontsize=7)
        for i in range(n):
            for j in range(len(MODEL_TIER_ORDER)):
                v = grid[i][j]
                if v != v:
                    continue
                norm_v = norm_grid[i][j]
                text_color = "#ffffff" if (norm_v == norm_v and norm_v > 0.55) else "#0b0b0b"
                ax.text(j, i, "{:.0f}".format(v), ha="center", va="center", fontsize=6.5, color=text_color)
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax1.set_title("Accuracy (%)")
    ax2.set_title("Avg. energy (J, normalized per column)")
    fig.suptitle(f"{label.replace(chr(10), ' ')} — per-query breakdown ({n} tasks)", fontsize=13)
    fig.tight_layout()
    save(fig, FIG_DIR, f"fig_routing_query_heatmap_{key}")


def main():
    apply()
    fig_tier_pareto()
    fig_policy_comparison()
    for key, label, raw_path, _ in EXPERIMENTS:
        fig_metrics_grid(key, label, raw_path)
        fig_pareto_dataset_comparison(key, label, raw_path)
        fig_query_heatmap(key, label, raw_path)


if __name__ == "__main__":
    main()
