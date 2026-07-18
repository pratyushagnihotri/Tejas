"""Compares all three experiments (Mac Qwen 3b/7b/14b, DGX Qwen 3b/7b/32b,
DGX Llama 3b/8b/70b) side by side: Pareto frontier shape, and how well the
scarce / calibrated-scarce policies do relative to static-cost-router.

Requires the tejas package installed (`pip install -e .` from the repo
root). Run after tejas.routing.static_policies + plot_results.py have been
run for all three:
    .venv/bin/python3 analysis/plot_cross_experiment.py
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

from tejas.paths import RESULTS_DIR as _RESULTS_DIR

from plot_results import MODEL_TIER_ORDER, MODEL_COLORS, _pareto_points, load_filtered_rows, _save

RESULTS_DIR = str(_RESULTS_DIR)
FIG_DIR = os.path.join(RESULTS_DIR, "figures", "cross_experiment")

EXPERIMENTS = [
    {"key": "mac", "label": "Mac\n(3b/7b/14b)",
     "raw": os.path.join(RESULTS_DIR, "raw_results.csv"),
     "policy": os.path.join(RESULTS_DIR, "policy_summary.csv")},
    {"key": "dgx_qwen", "label": "DGX Qwen\n(3b/7b/32b)",
     "raw": os.path.join(RESULTS_DIR, "raw_results_dgx_qwen.csv"),
     "policy": os.path.join(RESULTS_DIR, "policy_summary_dgx_qwen.csv")},
    {"key": "dgx_llama", "label": "DGX Llama\n(3b/8b/70b)",
     "raw": os.path.join(RESULTS_DIR, "raw_results_dgx_llama.csv"),
     "policy": os.path.join(RESULTS_DIR, "policy_summary_dgx_llama.csv")},
]

POLICIES_TO_SHOW = ["static-cost-router", "energy-aware (scarce)", "energy-aware (scarce, calibrated)"]
POLICY_COLORS = {"static-cost-router": "#eda100", "energy-aware (scarce)": "#e34948",
                  "energy-aware (scarce, calibrated)": "#eb6834"}

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


def load_policy_summary(path):
    with open(path) as f:
        rows = {r["policy"]: r for r in csv.DictReader(f)}
    return rows


def to_float(v):
    return float(v) if v not in ("", "None", None) else 0.0


def fig_pareto_across_experiments():
    """Small/medium/large Pareto frontier for each experiment, side by side
    on shared axes — shows how the frontier's shape changes as the
    model-capability gap widens from Mac -> DGX-Qwen -> DGX-Llama."""
    fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(4.6 * len(EXPERIMENTS), 4.6), sharey=True)
    for ax, exp in zip(axes, EXPERIMENTS):
        rows = load_filtered_rows(exp["raw"])
        points = _pareto_points(rows)
        present = [t for t in MODEL_TIER_ORDER if t in points]
        xs = [points[t][0] for t in present]
        ys = [points[t][1] for t in present]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ax.plot([xs[i] for i in order], [ys[i] for i in order],
                color="#c3c2b7", linewidth=1.5, zorder=2)
        for t in present:
            x, y = points[t]
            ax.scatter(x, y, s=120, color=MODEL_COLORS[t], edgecolor="white",
                       linewidth=1.2, zorder=3)
            ax.annotate(t, (x, y), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9, color="#52514e")
        ax.set_xlabel("Avg. energy / request (J), log scale")
        ax.set_xscale("log")
        # log-scale minor ticks (e.g. 2x10^2, 4x10^2, 6x10^2) collide when the
        # three points span a narrow range — cap to at most 4 labeled ticks.
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=6))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(exp["label"].replace("\n", " "), fontsize=11)

    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 100)
    fig.suptitle("Quality-vs-energy frontier widens as model-capability gap grows", y=1.03)
    fig.tight_layout()
    _save(fig, FIG_DIR, "fig_cross_pareto")


def fig_policy_across_experiments():
    """Quality and energy for static-cost-router vs scarce vs calibrated-scarce,
    grouped by experiment — the headline 'does the free lunch hold' figure."""
    n_exp = len(EXPERIMENTS)
    n_pol = len(POLICIES_TO_SHOW)
    width = 0.8 / n_pol
    x = list(range(n_exp))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))

    for i, policy in enumerate(POLICIES_TO_SHOW):
        q_means, q_stds, e_means, e_stds = [], [], [], []
        for exp in EXPERIMENTS:
            summary = load_policy_summary(exp["policy"])
            row = summary[policy]
            q_means.append(to_float(row["quality_pct_mean"]))
            q_stds.append(to_float(row["quality_pct_std"]))
            e_means.append(to_float(row["total_energy_j_mean"]))
            e_stds.append(to_float(row["total_energy_j_std"]))
        offsets = [xi + (i - (n_pol - 1) / 2) * width for xi in x]
        ax1.bar(offsets, q_means, yerr=q_stds, width=width * 0.9, capsize=3,
                color=POLICY_COLORS[policy], label=policy)
        ax2.bar(offsets, e_means, yerr=e_stds, width=width * 0.9, capsize=3,
                color=POLICY_COLORS[policy], label=policy)

    ax1.set_xticks(x)
    ax1.set_xticklabels([e["label"] for e in EXPERIMENTS])
    ax1.set_ylabel("Quality (%)")
    ax1.set_title("Quality by policy, across experiments")
    ax1.set_ylim(0, 100)
    ax1.legend(frameon=False, loc="lower left", fontsize=9)

    ax2.set_xticks(x)
    ax2.set_xticklabels([e["label"] for e in EXPERIMENTS])
    ax2.set_ylabel("Total energy (J), log scale")
    ax2.set_yscale("log")
    ax2.set_title("Energy by policy, across experiments")
    ax2.legend(frameon=False, loc="upper left", fontsize=9)

    fig.suptitle("Does the energy-aware 'free lunch' hold as the capability gap widens?", y=1.02)
    fig.tight_layout()
    _save(fig, FIG_DIR, "fig_cross_policy")


def main():
    fig_pareto_across_experiments()
    fig_policy_across_experiments()


if __name__ == "__main__":
    main()
