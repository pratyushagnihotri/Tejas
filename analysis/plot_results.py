"""Generates static, paper-ready figures (PNG + PDF) for a single experiment
(one raw_results CSV + its policy_summary CSV).

Requires the tejas package installed (`pip install -e .` from the repo
root). Run after tejas.routing.static_policies:
    .venv/bin/python3 analysis/plot_results.py
    .venv/bin/python3 analysis/plot_results.py --input results/raw_results_dgx_qwen.csv \\
        --policy-summary results/policy_summary_dgx_qwen.csv \\
        --output-dir results/figures/dgx_qwen --label "DGX Qwen (3b/7b/32b)"

For the Mac/DGX-Qwen/DGX-Llama comparison figures, see plot_cross_experiment.py.
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tejas.paths import RESULTS_DIR as _RESULTS_DIR

RESULTS_DIR = str(_RESULTS_DIR)
DEFAULT_RAW = os.path.join(RESULTS_DIR, "raw_results.csv")
DEFAULT_POLICY_SUMMARY = os.path.join(RESULTS_DIR, "policy_summary.csv")
DEFAULT_FIG_DIR = os.path.join(RESULTS_DIR, "figures")

MODEL_TIER_ORDER = ["small", "medium", "large"]
MODEL_COLORS = {"small": "#2a78d6", "medium": "#1baf7a", "large": "#4a3aa7"}
TASK_TIER_ORDER = ["easy", "medium", "hard"]
TASK_TIER_MARKERS = {"easy": "o", "medium": "s", "hard": "^"}
POLICY_COLORS = ["#4a3aa7", "#2a78d6", "#eda100", "#1baf7a", "#e34948", "#eb6834"]

DATASET_ORDER = ["db_sql", "sembench_taken3", "sembench_antman"]
DATASET_LABELS = {
    "db_sql": "DB / SQL benchmark",
    "sembench_taken3": "SemBench – taken_3 (skewed)",
    "sembench_antman": "SemBench – ant_man (balanced)",
}


def dataset_of(task_id):
    if task_id.startswith("sb_taken3"):
        return "sembench_taken3"
    if task_id.startswith("sb_antman"):
        return "sembench_antman"
    return "db_sql"


plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def load_filtered_rows(raw_path):
    """Same canonical-run filtering as tejas.benchmarking.results_loader:
    only aggregate runs that share the current task set, so old/new
    task-set versions never blend."""
    from tejas.benchmarking.results_loader import load_raw_results as load_analyze
    by_run_task_model, _task_tier, _runs = load_analyze(raw_path)
    return list(by_run_task_model.values())


def _pareto_points(rows):
    """{model_tier: (avg_energy_per_request, accuracy_pct)}"""
    groups = defaultdict(lambda: {"correct": 0, "n": 0, "energy": []})
    for r in rows:
        g = groups[r["model_tier"]]
        g["n"] += 1
        g["correct"] += 1 if r["correct"] == "True" else 0
        if r["energy_j"]:
            g["energy"].append(float(r["energy_j"]))
    points = {}
    for t in MODEL_TIER_ORDER:
        g = groups.get(t)
        if not g or not g["energy"]:
            continue
        points[t] = (sum(g["energy"]) / len(g["energy"]), 100 * g["correct"] / g["n"])
    return points


def fig_pareto(rows, fig_dir, label):
    """Quality vs. energy per (model_tier, task_tier), averaged across runs."""
    groups = defaultdict(lambda: {"correct": 0, "n": 0, "energy": []})
    for r in rows:
        g = groups[(r["model_tier"], r["task_tier"])]
        g["n"] += 1
        g["correct"] += 1 if r["correct"] == "True" else 0
        if r["energy_j"]:
            g["energy"].append(float(r["energy_j"]))

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for model_tier in MODEL_TIER_ORDER:
        xs, ys = [], []
        for task_tier in TASK_TIER_ORDER:
            g = groups.get((model_tier, task_tier))
            if not g or not g["energy"]:
                continue
            avg_energy = sum(g["energy"]) / len(g["energy"])
            avg_quality = 100 * g["correct"] / g["n"]
            xs.append(avg_energy)
            ys.append(avg_quality)
            ax.scatter(avg_energy, avg_quality, s=90,
                       color=MODEL_COLORS[model_tier],
                       marker=TASK_TIER_MARKERS[task_tier],
                       edgecolor="white", linewidth=1.2, zorder=3)
        if len(xs) > 1:
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            ax.plot([xs[i] for i in order], [ys[i] for i in order],
                    color=MODEL_COLORS[model_tier], alpha=0.35, linewidth=1.5, zorder=2)

    model_handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=9,
                                 color=MODEL_COLORS[m], label=m.capitalize())
                      for m in MODEL_TIER_ORDER]
    tier_handles = [plt.Line2D([0], [0], marker=TASK_TIER_MARKERS[t], linestyle="",
                                markersize=9, color="gray", label=t.capitalize())
                     for t in TASK_TIER_ORDER]
    leg1 = ax.legend(handles=model_handles, title="Model size", loc="lower right", frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=tier_handles, title="Task difficulty", loc="upper left", frameon=False)

    ax.set_xlabel("Avg. energy per request (J)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Quality vs. energy, by model size and task difficulty — {label}")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    _save(fig, fig_dir, "fig1_pareto")


def fig_policy_comparison(policy_summary_path, fig_dir, label):
    with open(policy_summary_path) as f:
        rows = list(csv.DictReader(f))

    def to_float(v):
        return float(v) if v not in ("", "None", None) else 0.0

    names = [r["policy"] for r in rows]
    q_mean = [to_float(r["quality_pct_mean"]) for r in rows]
    q_std = [to_float(r["quality_pct_std"]) for r in rows]
    e_mean = [to_float(r["total_energy_j_mean"]) for r in rows]
    e_std = [to_float(r["total_energy_j_std"]) for r in rows]
    n_runs = rows[0]["n_runs"] if rows else "1"

    colors = POLICY_COLORS[:len(names)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax1.bar(range(len(names)), q_mean, yerr=q_std, capsize=4, color=colors, width=0.6)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.set_ylabel("Quality (%)")
    ax1.set_title("Quality by routing policy")
    ax1.set_ylim(0, 100)

    ax2.bar(range(len(names)), e_mean, yerr=e_std, capsize=4, color=colors, width=0.6)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=30, ha="right")
    ax2.set_ylabel("Total energy (J)")
    ax2.set_title("Energy by routing policy")

    fig.suptitle(f"Routing policy comparison — {label} (n={n_runs} run{'s' if n_runs != '1' else ''})")
    fig.tight_layout()
    _save(fig, fig_dir, "fig2_policy_comparison")


def fig_metrics_by_dataset(rows, fig_dir, label):
    """Quality / energy / latency by model tier, one figure per dataset."""
    for ds in DATASET_ORDER:
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
        quality = [100 * groups[t]["correct"] / groups[t]["n"] for t in tiers]
        energy = [sum(groups[t]["energy"]) / len(groups[t]["energy"]) if groups[t]["energy"] else 0 for t in tiers]
        latency = [sum(groups[t]["latency"]) / len(groups[t]["latency"]) for t in tiers]
        colors = [MODEL_COLORS[t] for t in tiers]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.5, 4.2))
        for ax, values, ylabel, title in [
            (ax1, quality, "Accuracy (%)", "Quality"),
            (ax2, energy, "Avg. energy / request (J)", "Energy"),
            (ax3, latency, "Avg. latency / request (s)", "Latency"),
        ]:
            ax.bar(range(len(tiers)), values, color=colors, width=0.55)
            ax.set_xticks(range(len(tiers)))
            ax.set_xticklabels([t.capitalize() for t in tiers])
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        ax1.set_ylim(0, 100)

        fig.suptitle(f"{label} · {DATASET_LABELS[ds]} — metrics by model size ({len(ds_rows)} rows)")
        fig.tight_layout()
        _save(fig, fig_dir, f"fig3_metrics_{ds}")


def fig_pareto_by_dataset(rows, fig_dir, label):
    """Quality-vs-energy Pareto scatter, one figure per dataset."""
    for ds in DATASET_ORDER:
        ds_rows = [r for r in rows if dataset_of(r["task_id"]) == ds]
        if not ds_rows:
            continue
        points = _pareto_points(ds_rows)

        fig, ax = plt.subplots(figsize=(5.4, 4.4))
        present = [t for t in MODEL_TIER_ORDER if t in points]
        xs = [points[t][0] for t in present]
        ys = [points[t][1] for t in present]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ax.plot([xs[i] for i in order], [ys[i] for i in order],
                color="#c3c2b7", linewidth=1.5, zorder=2)
        for t in present:
            x, y = points[t]
            ax.scatter(x, y, s=110, color=MODEL_COLORS[t], edgecolor="white",
                       linewidth=1.2, zorder=3, label=t.capitalize())

        ax.set_xlabel("Avg. energy per request (J)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{label} · {DATASET_LABELS[ds]}")
        ax.set_ylim(0, 100)
        ax.legend(title="Model size", frameon=False, loc="best")
        fig.tight_layout()
        _save(fig, fig_dir, f"fig4_pareto_{ds}")


def fig_pareto_comparison(rows, fig_dir, label):
    """All datasets' Pareto frontiers side by side (shared y-axis) for direct comparison."""
    present_datasets = [ds for ds in DATASET_ORDER if any(dataset_of(r["task_id"]) == ds for r in rows)]
    fig, axes = plt.subplots(1, len(present_datasets),
                              figsize=(4.4 * len(present_datasets), 4.4), sharey=True)
    if len(present_datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, present_datasets):
        ds_rows = [r for r in rows if dataset_of(r["task_id"]) == ds]
        points = _pareto_points(ds_rows)
        present = [t for t in MODEL_TIER_ORDER if t in points]
        xs = [points[t][0] for t in present]
        ys = [points[t][1] for t in present]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ax.plot([xs[i] for i in order], [ys[i] for i in order],
                color="#c3c2b7", linewidth=1.5, zorder=2)
        for t in present:
            x, y = points[t]
            ax.scatter(x, y, s=110, color=MODEL_COLORS[t], edgecolor="white",
                       linewidth=1.2, zorder=3)
        ax.set_xlabel("Avg. energy / request (J)")
        ax.set_title(DATASET_LABELS[ds], fontsize=10.5)

    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 100)

    model_handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=9,
                                 color=MODEL_COLORS[m], label=m.capitalize())
                      for m in MODEL_TIER_ORDER]
    fig.legend(handles=model_handles, title="Model size", loc="upper center",
               bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False)
    fig.suptitle(f"Quality-vs-energy frontier, compared across datasets — {label}", y=1.16)
    fig.tight_layout()
    _save(fig, fig_dir, "fig5_pareto_comparison")


SEQ_BLUE = ["#f9f9f7", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#0d366b"]


def fig_query_heatmap(rows, fig_dir, label):
    """Fine-grained view: every individual task_id x model_tier cell (not
    aggregated by dataset/tier), two panels — accuracy and avg energy."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

    acc = defaultdict(lambda: [0, 0])
    energy = defaultdict(list)
    task_tier_of = {}
    for r in rows:
        key = (r["task_id"], r["model_tier"])
        acc[key][1] += 1
        if r["correct"] == "True":
            acc[key][0] += 1
        if r["energy_j"]:
            energy[key].append(float(r["energy_j"]))
        task_tier_of[r["task_id"]] = r["task_tier"]

    tier_rank = {"easy": 0, "medium": 1, "hard": 2}
    task_ids = sorted(task_tier_of, key=lambda t: (tier_rank[task_tier_of[t]], dataset_of(t), t))
    n = len(task_ids)

    q_grid = [[100 * acc[(t, mt)][0] / acc[(t, mt)][1] if acc[(t, mt)][1] else float("nan")
               for mt in MODEL_TIER_ORDER] for t in task_ids]
    e_grid = [[sum(energy[(t, mt)]) / len(energy[(t, mt)]) if energy[(t, mt)] else float("nan")
               for mt in MODEL_TIER_ORDER] for t in task_ids]

    # Energy is normalized per-column (per model tier): raw magnitudes differ
    # ~40x between small and large tiers, so a shared scale would flatten
    # every small/medium cell to the faintest color and hide within-tier
    # variation, which is the point of this view.
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, max(6, n * 0.26)))

    ax1.imshow(q_grid, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax2.imshow(e_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for ax, grid, norm_grid, fmt, vmax in [
        (ax1, q_grid, [[v / 100 for v in row] for row in q_grid], "{:.0f}", None),
        (ax2, e_grid, e_norm, "{:.0f}", None),
    ]:
        ax.set_xticks(range(len(MODEL_TIER_ORDER)))
        ax.set_xticklabels([t.capitalize() for t in MODEL_TIER_ORDER])
        ax.set_yticks(range(n))
        ax.set_yticklabels(task_ids, fontsize=7)
        for i in range(n):
            for j in range(len(MODEL_TIER_ORDER)):
                v = grid[i][j]
                if v != v:  # NaN
                    continue
                norm_v = norm_grid[i][j]
                text_color = "#ffffff" if (norm_v == norm_v and norm_v > 0.55) else "#0b0b0b"
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.5, color=text_color)
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax1.set_title("Accuracy (%)")
    ax2.set_title("Avg. energy / request (J, normalized per column)")
    fig.suptitle(f"{label} — per-query breakdown ({n} tasks)")
    fig.tight_layout()
    _save(fig, fig_dir, "fig6_query_heatmap")


def _save(fig, fig_dir, name):
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(fig_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {os.path.join(fig_dir, name)}.png / .pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_RAW)
    parser.add_argument("--policy-summary", default=None,
                         help="default: alongside --input, raw_results->policy_summary")
    parser.add_argument("--output-dir", default=None,
                         help="default: results/figures for the default input, else "
                              "results/figures/<input stem>")
    parser.add_argument("--label", default=None,
                         help="human label for chart titles (default: derived from --input)")
    args = parser.parse_args()

    policy_summary = args.policy_summary or os.path.join(
        os.path.dirname(args.input),
        os.path.basename(args.input).replace("raw_results", "policy_summary"))
    if args.output_dir:
        fig_dir = args.output_dir
    elif args.input == DEFAULT_RAW:
        fig_dir = DEFAULT_FIG_DIR
    else:
        stem = os.path.basename(args.input).replace("raw_results_", "").replace(".csv", "")
        fig_dir = os.path.join(RESULTS_DIR, "figures", stem)
    label = args.label or os.path.basename(args.input).replace(".csv", "")

    rows = load_filtered_rows(args.input)
    fig_pareto(rows, fig_dir, label)
    fig_policy_comparison(policy_summary, fig_dir, label)
    fig_metrics_by_dataset(rows, fig_dir, label)
    fig_pareto_by_dataset(rows, fig_dir, label)
    fig_pareto_comparison(rows, fig_dir, label)
    fig_query_heatmap(rows, fig_dir, label)


if __name__ == "__main__":
    main()
