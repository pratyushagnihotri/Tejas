"""Paper-ready figures for the energy cost model results (docs/RESEARCH_LOG.md
§10). Reuses the project's existing palette/style conventions from
plot_results.py (MODEL_COLORS, rcParams, _save) rather than inventing a new
look. New categorical color (learned cost-model policy, #0891b2 teal) was
checked against the existing palette with dataviz's validate_palette.js —
passes lightness/chroma/CVD-separation. Oracles are drawn as dashed
neutral-gray reference markers, not categorical hues — they represent a
bound, not a competing policy, so they're deliberately excluded from the
categorical CVD check (a bound isn't meant to be "identified," just seen).

Requires the tejas package installed (`pip install -e .` from the repo
root). Run after tejas.routing.evaluate_policy has produced
results/baseline_comparison_<experiment>.csv and
results/policy_decisions_learned_<experiment>.csv for all three experiments:
    .venv/bin/python3 analysis/plot_cost_model_results.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tejas.paths import RESULTS_DIR as _RESULTS_DIR

from plot_results import MODEL_COLORS, MODEL_TIER_ORDER, _save

RESULTS_DIR = str(_RESULTS_DIR)
FIG_DIR = os.path.join(RESULTS_DIR, "figures", "cost_model")

EXPERIMENTS = [("mac", "Mac\n(3b/7b/14b)"), ("dgx_qwen", "DGX-Qwen\n(3b/7b/32b)"),
               ("dgx_llama", "DGX-Llama\n(3b/8b/70b)")]

HAND_DESIGNED_COLORS = {"static-cost-router": "#eda100", "scarce": "#e34948",
                         "scarce, calibrated": "#eb6834"}
LEARNED_COLOR = "#0891b2"
ORACLE_COLOR = "#6b6a63"

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


def _policy_color(policy):
    if policy in HAND_DESIGNED_COLORS:
        return HAND_DESIGNED_COLORS[policy]
    if policy.startswith("always-"):
        return MODEL_COLORS[policy.replace("always-", "")]
    if policy == "learned cost-model policy":
        return LEARNED_COLOR
    return ORACLE_COLOR  # both oracles


def plot_baseline_pareto(experiment_key, label):
    """Legend-keyed rather than direct-labeled: on DGX-Llama in particular,
    static-cost-router / scarce-calibrated / the learned policy sit almost
    on top of each other (the point of Finding 2/3 — the learned policy
    barely deviates from static there), so inline text labels collide
    unreadably. A legend stays unambiguous regardless of how tightly
    points cluster."""
    path = os.path.join(RESULTS_DIR, f"baseline_comparison_{experiment_key}.csv")
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for _, row in df.iterrows():
        is_oracle = "oracle" in row["policy"]
        color = _policy_color(row["policy"])
        marker = "D" if is_oracle else ("*" if row["policy"] == "learned cost-model policy" else "o")
        size = 240 if row["policy"] == "learned cost-model policy" else (170 if is_oracle else 140)
        ax.scatter(row["energy_j"], row["quality_pct"], s=size, color=color, marker=marker,
                   edgecolor="white", linewidth=1.2, zorder=3, label=row["policy"])

    ax.set_xlabel("Total energy (J, 40 tasks, mean of 4 runs)")
    ax.set_ylabel("Quality (%)")
    ax.set_title(f"{label.replace(chr(10), ' ')}: policy comparison — quality vs. energy")
    ax.margins(x=0.15, y=0.15)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _save(fig, FIG_DIR, f"fig_baseline_pareto_{experiment_key}")


def plot_routing_behavior(experiment_key, label):
    path = os.path.join(RESULTS_DIR, f"policy_decisions_learned_{experiment_key}.csv")
    decisions = pd.read_csv(path)
    order = decisions.groupby("task_type_tier")["task_id"].nunique().sort_values(ascending=False).index.tolist()
    mix = (decisions.groupby("task_type_tier")["chosen_tier"]
           .value_counts(normalize=True).unstack(fill_value=0).reindex(columns=MODEL_TIER_ORDER, fill_value=0))
    mix = mix.loc[order]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bottom = np.zeros(len(mix))
    x = np.arange(len(mix))
    for tier in MODEL_TIER_ORDER:
        ax.bar(x, mix[tier], bottom=bottom, width=0.6, color=MODEL_COLORS[tier],
               label=tier.capitalize(), edgecolor="white", linewidth=1.5)
        bottom += mix[tier].values
    ax.set_xticks(x)
    ax.set_xticklabels(mix.index, rotation=20, ha="right")
    ax.set_ylabel("Fraction of (repeat x task) decisions")
    ax.set_title(f"{label.replace(chr(10), ' ')}: learned-policy tier assignment by workload class")
    ax.legend(title="Chosen tier", frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _save(fig, FIG_DIR, f"fig_routing_behavior_{experiment_key}")


def plot_cross_experiment_summary():
    policies = ["scarce", "scarce, calibrated", "learned cost-model policy"]
    colors = [HAND_DESIGNED_COLORS["scarce"], HAND_DESIGNED_COLORS["scarce, calibrated"], LEARNED_COLOR]

    energy_saved, quality_diff = {}, {}
    for key, _ in EXPERIMENTS:
        base = pd.read_csv(os.path.join(RESULTS_DIR, f"baseline_comparison_{key}.csv")).set_index("policy")
        energy_saved[key] = [base.loc[p, "pct_energy_saved_vs_static"] for p in policies]
        quality_diff[key] = [base.loc[p, "quality_diff_pp_vs_static"] for p in policies]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
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
    axes[1].set_ylabel("Quality diff vs. static-cost-router (pp)")
    axes[1].set_title("Quality change")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in EXPERIMENTS])
    axes[1].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("scarce vs. scarce, calibrated vs. learned cost-model policy, across capability gaps")
    _save(fig, FIG_DIR, "fig_cross_experiment_summary")


def main():
    for key, label in EXPERIMENTS:
        plot_baseline_pareto(key, label)
        plot_routing_behavior(key, label)
    plot_cross_experiment_summary()


if __name__ == "__main__":
    main()
