"""IEEE-style paper figures for the learned cost model. Same underlying
data as plot_cost_model_results.py (results/baseline_comparison_*.csv,
results/policy_decisions_learned_*.csv) but restyled for the paper (see
ieee_style.py) and combined into small-multiples where it saves space.

Requires the tejas package installed (`pip install -e .` from the repo
root). Run after tejas.routing.evaluate_policy:
    .venv/bin/python3 analysis/plot_ieee_cost_model.py

Figures written to results/figures/paper/cost_model/.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter

from tejas.paths import RESULTS_DIR as _RESULTS_DIR

from ieee_style import (FIGSIZE, HAND_DESIGNED_COLORS, LEARNED_COLOR, MODEL_COLORS,
                         MODEL_TIER_ORDER, ORACLE_COLOR, apply, save)

RESULTS_DIR = str(_RESULTS_DIR)
FIG_DIR = os.path.join(RESULTS_DIR, "figures", "paper", "cost_model")

EXPERIMENTS = [("mac", "Mac\n(3b/7b/14b)"), ("dgx_qwen", "DGX-Qwen\n(3b/7b/32b)"),
               ("dgx_llama", "DGX-Llama\n(3b/8b/70b)")]


def _policy_color(policy):
    if policy in HAND_DESIGNED_COLORS:
        return HAND_DESIGNED_COLORS[policy]
    if policy.startswith("always-"):
        return MODEL_COLORS[policy.replace("always-", "")]
    if policy == "learned cost-model policy":
        return LEARNED_COLOR
    return ORACLE_COLOR


def fig_baseline_pareto():
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE["row3"])
    handles_labels = None
    for ax, (key, label) in zip(axes, EXPERIMENTS):
        df = pd.read_csv(os.path.join(RESULTS_DIR, f"baseline_comparison_{key}.csv"))
        for _, row in df.iterrows():
            is_oracle = "oracle" in row["policy"]
            color = _policy_color(row["policy"])
            marker = "D" if is_oracle else ("*" if row["policy"] == "learned cost-model policy" else "o")
            size = 200 if row["policy"] == "learned cost-model policy" else (140 if is_oracle else 110)
            ax.scatter(row["energy_j"], row["quality_pct"], s=size, color=color, marker=marker,
                       edgecolor="white", linewidth=1.0, zorder=3, label=row["policy"])
        ax.set_xlabel("Total energy (J)")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=5))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(label.replace("\n", " "))
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()
    axes[0].set_ylabel("Quality (%)")
    fig.legend(*handles_labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.14),
               fontsize=9.5, frameon=False)
    fig.suptitle("Policy comparison: quality vs. energy", y=1.28, fontsize=13)
    save(fig, FIG_DIR, "fig_cost_model_pareto")


def fig_routing_behavior():
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE["row3"])
    for ax, (key, label) in zip(axes, EXPERIMENTS):
        decisions = pd.read_csv(os.path.join(RESULTS_DIR, f"policy_decisions_learned_{key}.csv"))
        order = decisions.groupby("task_type_tier")["task_id"].nunique().sort_values(ascending=False).index.tolist()
        mix = (decisions.groupby("task_type_tier")["chosen_tier"]
               .value_counts(normalize=True).unstack(fill_value=0)
               .reindex(columns=MODEL_TIER_ORDER, fill_value=0)).loc[order]

        bottom = np.zeros(len(mix))
        x = np.arange(len(mix))
        for tier in MODEL_TIER_ORDER:
            ax.bar(x, mix[tier], bottom=bottom, width=0.62, color=MODEL_COLORS[tier],
                   label=tier.capitalize(), edgecolor="white", linewidth=1.2)
            bottom += mix[tier].values
        ax.set_xticks(x)
        ax.set_xticklabels(mix.index, rotation=35, ha="right", fontsize=8.5)
        ax.set_title(label.replace("\n", " "))
    axes[0].set_ylabel("Fraction of decisions")
    axes[-1].legend(title="Chosen tier", loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=10)
    fig.suptitle("Learned-policy tier assignment by workload class", fontsize=13)
    save(fig, FIG_DIR, "fig_cost_model_routing_behavior")


def fig_cross_experiment_summary():
    policies = ["scarce", "scarce, calibrated", "learned cost-model policy"]
    colors = [HAND_DESIGNED_COLORS["scarce"], HAND_DESIGNED_COLORS["scarce, calibrated"], LEARNED_COLOR]

    energy_saved, quality_diff = {}, {}
    for key, _ in EXPERIMENTS:
        base = pd.read_csv(os.path.join(RESULTS_DIR, f"baseline_comparison_{key}.csv")).set_index("policy")
        energy_saved[key] = [base.loc[p, "pct_energy_saved_vs_static"] for p in policies]
        quality_diff[key] = [base.loc[p, "quality_diff_pp_vs_static"] for p in policies]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["row2"])
    x = np.arange(len(EXPERIMENTS))
    width = 0.25
    for i, policy in enumerate(policies):
        vals_e = [energy_saved[key][i] for key, _ in EXPERIMENTS]
        vals_q = [quality_diff[key][i] for key, _ in EXPERIMENTS]
        offset = (i - 1) * width
        axes[0].bar(x + offset, vals_e, width=width, color=colors[i], label=policy, edgecolor="white")
        axes[1].bar(x + offset, vals_q, width=width, color=colors[i], label=policy, edgecolor="white")

    axes[0].axhline(0, color="#8a8a80", linewidth=1)
    axes[0].set_ylabel("% energy saved vs. static-cost-router")
    axes[0].set_title("Energy savings")
    axes[1].axhline(0, color="#8a8a80", linewidth=1)
    axes[1].axhline(-2, color="#c0392b", linewidth=1, linestyle=":", label="-2pp budget")
    axes[1].set_ylabel("Quality diff. (pp)")
    axes[1].set_title("Quality change")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in EXPERIMENTS])
    axes[1].legend(fontsize=9.5, loc="lower left")
    fig.suptitle("scarce vs. scarce, calibrated vs. learned policy", fontsize=13)
    save(fig, FIG_DIR, "fig_cost_model_summary")


def main():
    apply()
    fig_baseline_pareto()
    fig_routing_behavior()
    fig_cross_experiment_summary()


if __name__ == "__main__":
    main()
