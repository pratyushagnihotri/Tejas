"""Builds the row-level training frame shared by energy_model.py,
quality_model.py, oracle.py, and the evaluate_*.py harnesses: one row per
(run, task_id, model_tier), combining results_loader.py's measured outcomes
with task_features.py's workload features and deployment_features.py's
deployment metadata.
"""

import math
import os

import pandas as pd

from tejas.benchmarking.results_loader import load_raw_results, STATIC_ROUTE
from tejas.cost_model.deployment_features import DEPLOYMENTS, deployment_for_csv, param_count_b
from tejas.cost_model.task_features import build_task_features
from tejas.paths import RESULTS_DIR

# Feature groups, used by evaluate_models.py's ablation (base -> +sql_semantic
# -> +interactions -> +deployment). "interactions" are meaningful only for the
# linear models (Ridge/ElasticNet/LogisticRegression) — tree models learn
# interactions natively from the base+sql_semantic+deployment columns already.
FEATURE_GROUPS = {
    "base": {
        "categorical": ["task_type_tier", "model_tier"],
        "numeric": ["prompt_chars", "prompt_tokens", "log_param_count"],
    },
    "sql_semantic": {
        "categorical": ["answer_format"],
        "numeric": ["n_joins", "n_predicates", "n_aggregations", "n_groupby_cols",
                    "n_projected_cols", "has_subquery", "has_window_fn", "n_reviews"],
    },
    "interactions": {
        "categorical": [],
        "numeric": ["log_param_count_x_is_semantic", "log_param_count_x_is_dgx"],
    },
    "deployment": {
        "categorical": ["platform", "device_class", "model_family"],
        "numeric": ["gpu_count", "params_per_gpu"],
    },
}


def feature_columns(groups=("base", "sql_semantic", "interactions")):
    """groups: which FEATURE_GROUPS to include, in order. Returns
    (categorical_cols, numeric_cols). "deployment" is only informative once
    pooling across experiments (Next tier) — including it within a single
    deployment just adds constant columns, harmless but not useful."""
    categorical, numeric = [], []
    for g in groups:
        categorical += FEATURE_GROUPS[g]["categorical"]
        numeric += FEATURE_GROUPS[g]["numeric"]
    return categorical, numeric


def load_experiment_frame(csv_path):
    """Returns (df, task_tier, runs). df has one row per (run, task_id,
    model_tier) with all features + targets (correct, energy_j, latency_s)."""
    by_run_task_model, task_tier, runs = load_raw_results(csv_path)
    task_feats = build_task_features()
    deployment_key = deployment_for_csv(csv_path)
    dep = DEPLOYMENTS[deployment_key]

    rows = []
    for (run, task_id, model_tier), row in by_run_task_model.items():
        tf = task_feats[task_id]
        pcount = param_count_b(row["model_name"])
        log_pcount = math.log1p(pcount)
        is_semantic = 1 if tf["task_type"] == "semantic" else 0
        is_dgx = 1 if dep["platform"] == "dgx" else 0
        params_per_gpu = (pcount / dep["gpu_count"]) if dep["gpu_count"] > 0 else 0.0

        rows.append({
            "run": run, "task_id": task_id, "model_tier": model_tier,
            "model_name": row["model_name"],
            "deployment": deployment_key,
            "reference_tier": STATIC_ROUTE[tf["task_tier"]],
            "param_count_b": pcount, "log_param_count": log_pcount,
            "platform": dep["platform"], "device_class": dep["device_class"],
            "model_family": dep["model_family"], "gpu_count": dep["gpu_count"],
            "measurement_scope": dep["measurement_scope"],
            "params_per_gpu": params_per_gpu,
            "log_param_count_x_is_semantic": log_pcount * is_semantic,
            "log_param_count_x_is_dgx": log_pcount * is_dgx,
            **tf,
            "has_subquery": int(tf["has_subquery"]),
            "has_window_fn": int(tf["has_window_fn"]),
            "correct": row["correct"] == "True",
            "energy_j": float(row["energy_j"]),
            "latency_s": float(row["latency_s"]),
        })

    df = pd.DataFrame(rows)
    return df, task_tier, runs


def _self_check():
    for csv_name in ("raw_results.csv", "raw_results_dgx_qwen.csv", "raw_results_dgx_llama.csv"):
        path = os.path.join(str(RESULTS_DIR), csv_name)
        df, task_tier, runs = load_experiment_frame(path)
        assert len(df) == 40 * 3 * len(runs), (csv_name, len(df), len(runs))
        assert df["task_id"].nunique() == 40
        assert df["energy_j"].gt(0).all()
        assert not df.isnull().any().any(), df.isnull().sum()[df.isnull().sum() > 0]
        cat, num = feature_columns(("base", "sql_semantic", "interactions", "deployment"))
        for c in cat + num:
            assert c in df.columns, f"{csv_name}: missing feature column {c}"
        print(f"{csv_name}: {len(df)} rows, {df['task_id'].nunique()} tasks, "
              f"{len(runs)} runs, measurement_scope={df['measurement_scope'].iloc[0]}")


if __name__ == "__main__":
    _self_check()
