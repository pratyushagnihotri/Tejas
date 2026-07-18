"""IEEE-style figures for three evaluation questions that previously had
numbers (docs/RESEARCH_LOG.md SS10) but no plot:

1. Does the cost model actually predict well? (predicted-vs-actual energy,
   quality calibration/reliability curve)
2. Does it transfer to a different model family? (cross_family_transfer.py)
3. How much target-deployment data is needed to adapt it? (fewshot_calibration.py)

Requires the tejas package installed (`pip install -e .` from the repo
root). Run after tejas.cost_model.evaluate_models,
tejas.cost_model.cross_family_transfer, and
tejas.cost_model.fewshot_calibration have all produced their CSVs:
    .venv/bin/python3 analysis/plot_ieee_evaluation.py

Figures written to results/figures/paper/cost_model/.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter
from sklearn.calibration import calibration_curve

from tejas.paths import RESULTS_DIR as _RESULTS_DIR

from ieee_style import FIGSIZE, MODEL_COLORS, MODEL_TIER_ORDER, apply, save

RESULTS_DIR = str(_RESULTS_DIR)
FIG_DIR = os.path.join(RESULTS_DIR, "figures", "paper", "cost_model")

EXPERIMENTS = [("mac", "Mac\n(3b/7b/14b)"), ("dgx_qwen", "DGX-Qwen\n(3b/7b/32b)"),
               ("dgx_llama", "DGX-Llama\n(3b/8b/70b)")]

TRANSFER_DIRECTIONS = [("dgx_qwen", "dgx_llama", "Qwen → Llama"),
                        ("dgx_llama", "dgx_qwen", "Llama → Qwen")]


def fig_prediction_accuracy():
    """Row 1: predicted vs. actual energy (out-of-fold, log-log, y=x
    reference). Row 2: quality reliability diagram (out-of-fold). Both
    pooled across all 5x5 repeated nested CV folds -- genuinely held-out
    predictions, not in-sample fit."""
    fig, axes = plt.subplots(2, 3, figsize=(FIGSIZE["row3"][0], FIGSIZE["row3"][1] * 1.9))

    for col, (key, label) in enumerate(EXPERIMENTS):
        oof = pd.read_csv(os.path.join(RESULTS_DIR, f"oof_predictions_{key}.csv"))

        ax = axes[0, col]
        for tier in MODEL_TIER_ORDER:
            sub = oof[oof["model_tier"] == tier]
            ax.scatter(sub["actual_energy_j"], sub["pred_energy_j"], s=10, alpha=0.35,
                       color=MODEL_COLORS[tier], label=tier.capitalize(), linewidths=0)
        lo = min(oof["actual_energy_j"].min(), oof["pred_energy_j"].min())
        hi = max(oof["actual_energy_j"].max(), oof["pred_energy_j"].max())
        ax.plot([lo, hi], [lo, hi], color="#8a8a80", linewidth=1.2, linestyle="--", zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(base=10, numticks=5))
            axis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=5))
            axis.set_minor_formatter(NullFormatter())
        ax.set_title(label.replace("\n", " "))
        ax.set_xlabel("Actual energy (J)")
        if col == 0:
            ax.set_ylabel("Predicted energy (J)")

        ax2 = axes[1, col]
        frac_pos, mean_pred = calibration_curve(oof["actual_correct"], oof["pred_quality"],
                                                  n_bins=8, strategy="quantile")
        ax2.plot([0, 1], [0, 1], color="#8a8a80", linewidth=1.2, linestyle="--", zorder=1)
        ax2.plot(mean_pred, frac_pos, marker="o", markersize=5, color="#0891b2", linewidth=1.6)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Predicted P(correct)")
        if col == 0:
            ax2.set_ylabel("Observed fraction correct")

    axes[0, -1].legend(title="Model tier", loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    fig.suptitle("Cost-model prediction accuracy (5x5 repeated nested CV): "
                 "(top) energy predicted vs. actual; (bottom) quality reliability diagram",
                 fontsize=12.5)
    fig.tight_layout()
    save(fig, FIG_DIR, "fig_cost_model_prediction_accuracy")


def _best_row(df, metric, minimize=True):
    return df.loc[df[metric].idxmin() if minimize else df[metric].idxmax()]


def fig_cross_family_transfer():
    """Within-deployment vs. matched-workload (Protocol A) vs.
    unseen-workload (Protocol B) transfer, both directions, energy MAE and
    quality accuracy -- the best-performing model kind per (direction,
    protocol), matching how the numbers are quoted in docs/RESEARCH_LOG.md."""
    conditions = ["within-\ndeployment", "matched-\nworkload (A)", "unseen-\nworkload (B)"]
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["row2"])
    width = 0.35
    x = np.arange(len(conditions))

    for i, (source, target, dlabel) in enumerate(TRANSFER_DIRECTIONS):
        model_eval = pd.read_csv(os.path.join(RESULTS_DIR, f"model_eval_{target}.csv"))
        within_mae = model_eval["energy_mae_j"].mean()
        within_acc = model_eval["quality_accuracy"].mean()

        a_energy = pd.read_csv(os.path.join(RESULTS_DIR, f"transfer_A_energy_{source}_to_{target}.csv"))
        a_quality = pd.read_csv(os.path.join(RESULTS_DIR, f"transfer_A_quality_{source}_to_{target}.csv"))
        b_energy = pd.read_csv(os.path.join(RESULTS_DIR, f"transfer_B_energy_{source}_to_{target}.csv"))
        b_quality = pd.read_csv(os.path.join(RESULTS_DIR, f"transfer_B_quality_{source}_to_{target}.csv"))

        mae_vals = [within_mae, _best_row(a_energy, "mae_j")["mae_j"],
                    _best_row(b_energy, "mae_j_mean")["mae_j_mean"]]
        # All three accuracy sources are on a 0-1 scale (confirmed against
        # the actual CSVs, not assumed) -- convert to percent for the plot.
        acc_vals = [within_acc * 100,
                    _best_row(a_quality, "accuracy", minimize=False)["accuracy"] * 100,
                    _best_row(b_quality, "accuracy_mean", minimize=False)["accuracy_mean"] * 100]

        offset = (i - 0.5) * width
        axes[0].bar(x + offset, mae_vals, width=width, label=dlabel,
                    color=["#2a78d6", "#e34948"][i], edgecolor="white")
        axes[1].bar(x + offset, acc_vals, width=width, label=dlabel,
                    color=["#2a78d6", "#e34948"][i], edgecolor="white")

    axes[0].set_ylabel("Energy MAE (J)")
    axes[0].set_title("Energy prediction error")
    axes[1].set_ylabel("Quality accuracy (%)")
    axes[1].set_title("Quality prediction accuracy")
    axes[1].set_ylim(0, 100)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
    axes[0].legend(fontsize=10, loc="upper left")
    fig.suptitle("Cross-family transfer: DGX-Qwen ↔ DGX-Llama", fontsize=13)
    save(fig, FIG_DIR, "fig_cost_model_cross_family_transfer")


def fig_fewshot_calibration():
    """Energy MAE and quality accuracy vs. number of target-deployment
    calibration tasks (k), one column per transfer direction -- the
    full-refit-vs-linear-calibration asymmetry is the point of this
    figure, so the two directions must be visually easy to compare."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "fewshot_calibration.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(FIGSIZE["row2"][0], FIGSIZE["row2"][1] * 1.9))
    method_colors = {"full refit": "#eb6834", "linear calibration": "#0891b2"}
    method_markers = {"full refit": "s", "linear calibration": "o"}

    for col, (source, target, dlabel) in enumerate(TRANSFER_DIRECTIONS):
        sub = df[(df["source"] == source) & (df["target"] == target)]
        zero_shot = sub[sub["method"] == "zero-shot (no calibration)"].iloc[0]

        ax_mae = axes[0, col]
        ax_acc = axes[1, col]
        ax_mae.axhline(zero_shot["mae_j"], color="#8a8a80", linewidth=1.2, linestyle=":",
                        label="zero-shot")
        ax_acc.axhline(zero_shot["accuracy"] * 100, color="#8a8a80", linewidth=1.2, linestyle=":",
                        label="zero-shot")

        for method in ("full refit", "linear calibration"):
            m = sub[sub["method"] == method].sort_values("k")
            ax_mae.errorbar(m["k"], m["mae_j"], yerr=m["mae_j_std"], marker=method_markers[method],
                             color=method_colors[method], label=method, capsize=3, linewidth=1.6)
            ax_acc.errorbar(m["k"], m["accuracy"] * 100, yerr=m["accuracy_std"] * 100,
                             marker=method_markers[method], color=method_colors[method],
                             label=method, capsize=3, linewidth=1.6)

        ax_mae.set_title(dlabel)
        ax_mae.set_xlabel("k (calibration tasks)")
        ax_acc.set_xlabel("k (calibration tasks)")
        ax_mae.set_xticks([1, 5, 10])
        ax_acc.set_xticks([1, 5, 10])
        if col == 0:
            ax_mae.set_ylabel("Energy MAE (J)")
            ax_acc.set_ylabel("Quality accuracy (%)")

    axes[0, -1].legend(fontsize=9.5, loc="upper right")
    fig.suptitle("Few-shot deployment calibration", fontsize=13)
    fig.tight_layout()
    save(fig, FIG_DIR, "fig_cost_model_fewshot_calibration")


def main():
    apply()
    fig_prediction_accuracy()
    fig_cross_family_transfer()
    fig_fewshot_calibration()


if __name__ == "__main__":
    main()
