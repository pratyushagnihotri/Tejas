"""Two spot-checks from the plan's verification section, run as a single
train/test split per experiment (not the full 5x5 repeated nested CV used
for headline numbers — this is a leakage/sanity check, not a result to
report with the same statistical weight):

1. Shuffling task features (keeping deployment/model features intact)
   should degrade task-aware prediction — if it doesn't, task features
   aren't actually contributing, or there's a leak making the model
   trivially solvable some other way.
2. Feature-group ablation (base -> +sql_semantic -> +interactions) shows
   which feature families matter, using the same held-out split.
"""

import os

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss

from tejas.cost_model.dataset import FEATURE_GROUPS, feature_columns, load_experiment_frame
from tejas.cost_model.energy_model import EnergyModel
from tejas.cost_model.quality_model import QualityModel
from tejas.paths import RESULTS_DIR

TASK_FEATURE_COLS = FEATURE_GROUPS["base"]["categorical"][:1] + FEATURE_GROUPS["base"]["numeric"][:2] \
    + FEATURE_GROUPS["sql_semantic"]["categorical"] + FEATURE_GROUPS["sql_semantic"]["numeric"]


def _single_split(df, seed=0):
    splitter = GroupKFold(n_splits=5)
    task_ids = df["task_id"].unique()
    train_pos, test_pos = next(splitter.split(task_ids, groups=task_ids))
    train_tasks, test_tasks = set(task_ids[train_pos]), set(task_ids[test_pos])
    return df[df["task_id"].isin(train_tasks)], df[df["task_id"].isin(test_tasks)]


def shuffle_check(csv_path, experiment_name):
    df, _, _ = load_experiment_frame(csv_path)
    train_df, test_df = _single_split(df)
    cat, num = feature_columns(("base", "sql_semantic", "interactions"))

    em = EnergyModel("histgb", cat, num).fit(train_df, train_df["energy_j"])
    qm = QualityModel("histgb", cat, num).fit(train_df, train_df["correct"], groups=train_df["task_id"])
    mae = float(np.mean(np.abs(em.predict(test_df) - test_df["energy_j"])))
    ll = float(log_loss(test_df["correct"], np.clip(qm.predict_proba(test_df), 1e-6, 1 - 1e-6), labels=[0, 1]))

    rng = np.random.default_rng(0)
    shuffled_train = train_df.copy()
    present_cols = [c for c in TASK_FEATURE_COLS if c in shuffled_train.columns]
    perm = rng.permutation(len(shuffled_train))
    shuffled_train[present_cols] = shuffled_train[present_cols].values[perm]

    em_s = EnergyModel("histgb", cat, num).fit(shuffled_train, shuffled_train["energy_j"])
    qm_s = QualityModel("histgb", cat, num).fit(shuffled_train, shuffled_train["correct"], groups=shuffled_train["task_id"])
    mae_s = float(np.mean(np.abs(em_s.predict(test_df) - test_df["energy_j"])))
    ll_s = float(log_loss(test_df["correct"], np.clip(qm_s.predict_proba(test_df), 1e-6, 1 - 1e-6), labels=[0, 1]))

    print(f"  {experiment_name}: energy MAE real={mae:.2f}J shuffled={mae_s:.2f}J "
          f"({'DEGRADED as expected' if mae_s > mae else 'NO DEGRADATION — investigate'})")
    print(f"  {experiment_name}: quality log-loss real={ll:.3f} shuffled={ll_s:.3f} "
          f"({'DEGRADED as expected' if ll_s > ll else 'NO DEGRADATION — investigate'})")
    return mae, mae_s, ll, ll_s


def ablation_check(csv_path, experiment_name):
    df, _, _ = load_experiment_frame(csv_path)
    train_df, test_df = _single_split(df)
    levels = [("base",), ("base", "sql_semantic"), ("base", "sql_semantic", "interactions")]
    print(f"  {experiment_name} feature-group ablation (single split, histgb):")
    for groups in levels:
        cat, num = feature_columns(groups)
        em = EnergyModel("histgb", cat, num).fit(train_df, train_df["energy_j"])
        qm = QualityModel("histgb", cat, num).fit(train_df, train_df["correct"], groups=train_df["task_id"])
        mae = float(np.mean(np.abs(em.predict(test_df) - test_df["energy_j"])))
        ll = float(log_loss(test_df["correct"], np.clip(qm.predict_proba(test_df), 1e-6, 1 - 1e-6), labels=[0, 1]))
        print(f"    {'+'.join(groups):<35} energy MAE={mae:7.2f}J  quality log-loss={ll:.3f}")


def main():
    from tejas.cost_model.deployment_features import CSV_TO_DEPLOYMENT
    print("=== shuffle-leakage check ===")
    for csv_name, key in CSV_TO_DEPLOYMENT.items():
        shuffle_check(os.path.join(str(RESULTS_DIR), csv_name), key)
    print("\n=== feature-group ablation ===")
    for csv_name, key in CSV_TO_DEPLOYMENT.items():
        ablation_check(os.path.join(str(RESULTS_DIR), csv_name), key)


if __name__ == "__main__":
    main()
