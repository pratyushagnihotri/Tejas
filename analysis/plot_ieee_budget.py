"""The budget-vs-quality frontier: achieved quality as a function of the
energy budget available (as a fraction of static-cost-router's total),
one panel per deployment. This is the figure that shows behavior actually
tracking *how much* energy is available, not just a single before/after
scarcity toggle — the bridge between Tejas and computation-by-energy's
claim that energy availability should be an input to the decision, not a
binary switch.

Requires the tejas package installed (`pip install -e .` from the repo
root). Run after tejas.routing.evaluate_budget has produced
results/budget_sweep_<experiment>.csv and
results/budget_sweep_scarce_point_<experiment>.csv for all three
experiments:
    .venv/bin/python3 analysis/plot_ieee_budget.py

Figure written to results/figures/paper/cost_model/.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from tejas.paths import RESULTS_DIR as _RESULTS_DIR

from ieee_style import FIGSIZE, LEARNED_COLOR, ORACLE_COLOR, HAND_DESIGNED_COLORS, apply, save

RESULTS_DIR = str(_RESULTS_DIR)
FIG_DIR = os.path.join(RESULTS_DIR, "figures", "paper", "cost_model")

EXPERIMENTS = [("mac", "Mac\n(3b/7b/14b)"), ("dgx_qwen", "DGX-Qwen\n(3b/7b/32b)"),
               ("dgx_llama", "DGX-Llama\n(3b/8b/70b)")]

SCARCE_COLOR = HAND_DESIGNED_COLORS["scarce"]


def fig_budget_frontier():
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE["row3"], sharey=True)
    handles_labels = None

    for ax, (key, label) in zip(axes, EXPERIMENTS):
        curve = pd.read_csv(os.path.join(RESULTS_DIR, f"budget_sweep_{key}.csv")).sort_values("budget_frac")
        scarce = pd.read_csv(os.path.join(RESULTS_DIR, f"budget_sweep_scarce_point_{key}.csv")).iloc[0]

        ax.plot(curve["oracle_energy_pct"], curve["oracle_quality_pct"], color=ORACLE_COLOR,
                linewidth=1.6, linestyle="--", marker="D", markersize=5, label="Measured oracle", zorder=2)
        ax.fill_between(curve["tejas_energy_pct_mean"], curve["tejas_quality_pct_ci_lo"],
                         curve["tejas_quality_pct_ci_hi"], color=LEARNED_COLOR, alpha=0.15, zorder=1)
        ax.plot(curve["tejas_energy_pct_mean"], curve["tejas_quality_pct_mean"], color=LEARNED_COLOR,
                linewidth=2.0, marker="o", markersize=6, label="Tejas", zorder=3)
        ax.scatter([scarce["energy_pct"]], [scarce["quality_pct"]], color=SCARCE_COLOR, marker="^",
                   s=110, edgecolor="white", linewidth=1.0, zorder=4, label="Fixed downgrade (scarce)")

        ax.set_xlabel("Energy budget (% of static)")
        ax.set_title(label.replace("\n", " "))
        ax.invert_xaxis()  # scarcer (lower budget) to the right, reads left-to-right as "energy gets scarcer"
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()

    axes[0].set_ylabel("Achieved quality (%)")
    fig.legend(*handles_labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12),
               fontsize=10, frameon=False)
    fig.suptitle("Quality achieved across energy budgets: does behavior track how much energy is available?",
                 y=1.22, fontsize=12.5)
    save(fig, FIG_DIR, "fig_cost_model_budget_frontier")


def main():
    apply()
    fig_budget_frontier()


if __name__ == "__main__":
    main()
