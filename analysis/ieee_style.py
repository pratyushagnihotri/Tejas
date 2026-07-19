""" IEEE-style matplotlib config used by both
plot_ieee_routing.py (original hand-designed policies) and
plot_ieee_cost_model.py (learned cost model).
"""

import matplotlib

IEEE_RCPARAMS = {
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "legend.frameon": False,
}

# width, height in inches. "row3"/"row2" = a figure with that many
# side-by-side subplots; generous enough at the rcParams above that 13pt
# titles and 12pt axis labels don't crowd the panel edges.
FIGSIZE = {
    "single": (5.2, 4.0),
    "row2": (9.0, 4.0),
    "row3": (12.5, 4.2),
    "wide_single": (7.5, 4.2),
}

MODEL_TIER_ORDER = ["small", "medium", "large"]
MODEL_COLORS = {"small": "#2a78d6", "medium": "#1baf7a", "large": "#4a3aa7"}
HAND_DESIGNED_COLORS = {
    "static-cost-router": "#eda100",
    "energy-aware (scarce)": "#e34948",
    "scarce": "#e34948",
    "energy-aware (scarce, calibrated)": "#eb6834",
    "scarce, calibrated": "#eb6834",
}
LEARNED_COLOR = "#0891b2"
ORACLE_COLOR = "#6b6a63"


def apply():
    matplotlib.rcParams.update(IEEE_RCPARAMS)


def save(fig, fig_dir, name):
    import os
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(fig_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved {os.path.join(fig_dir, name)}.png / .pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)
