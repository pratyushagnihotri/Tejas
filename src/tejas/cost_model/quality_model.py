"""Quality prediction ladder: Q_hat(task, model, deployment) -> P(correct).

Calibrated probabilities matter more here than raw accuracy, since
policy_optimizer.py compares predicted quality *differences* against a
delta threshold — a systematically over/under-confident model would bias
every routing decision the same direction. histgb gets an optional
fold-safe Platt (sigmoid) scaling: fit the classifier on a group-safe
subset of the training fold, calibrate on the held-out remainder of that
same fold (never on validation/test rows). Isotonic calibration is skipped
as unsafe at this sample size (40 tasks).
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

QUALITY_MODEL_KINDS = ["tier_baseline", "logreg", "histgb"]

_CLASSIFIERS = {
    "logreg": lambda: LogisticRegression(max_iter=2000, C=1.0),
    "histgb": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=100, random_state=0),
}


def _column_transformer(categorical_cols, numeric_cols):
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("numeric", StandardScaler(), numeric_cols),
    ])


class QualityModel:
    def __init__(self, kind, categorical_cols, numeric_cols, calibrate=False, random_state=0):
        assert kind in QUALITY_MODEL_KINDS, kind
        self.kind = kind
        self.categorical_cols = list(categorical_cols)
        self.numeric_cols = list(numeric_cols)
        self.calibrate = calibrate and kind != "tier_baseline"
        self.random_state = random_state
        self._tier_acc = None
        self._global_acc = None
        self._pipeline = None
        self._platt = None

    def fit(self, df, correct, groups=None):
        y = np.asarray(correct, dtype=int)

        if self.kind == "tier_baseline":
            groups_acc = {}
            for k, c in zip(zip(df["deployment"], df["task_type_tier"], df["model_tier"]), y):
                groups_acc.setdefault(k, []).append(c)
            self._tier_acc = {k: float(np.mean(v)) for k, v in groups_acc.items()}
            self._global_acc = float(y.mean())
            return self

        pipeline = Pipeline([
            ("features", _column_transformer(self.categorical_cols, self.numeric_cols)),
            ("model", _CLASSIFIERS[self.kind]()),
        ])

        if self.calibrate:
            if groups is None:
                raise ValueError("calibrate=True needs task_id groups for a group-safe split")
            gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=self.random_state)
            fit_idx, calib_idx = next(gss.split(df, y, groups=groups))
            pipeline.fit(df.iloc[fit_idx], y[fit_idx])
            raw = pipeline.predict_proba(df.iloc[calib_idx])[:, 1].reshape(-1, 1)
            self._platt = LogisticRegression().fit(raw, y[calib_idx])
        else:
            pipeline.fit(df, y)

        self._pipeline = pipeline
        return self

    def predict_proba(self, df):
        if self.kind == "tier_baseline":
            key = list(zip(df["deployment"], df["task_type_tier"], df["model_tier"]))
            return np.array([self._tier_acc.get(k, self._global_acc) for k in key])
        raw = self._pipeline.predict_proba(df)[:, 1]
        if self._platt is not None:
            return self._platt.predict_proba(raw.reshape(-1, 1))[:, 1]
        return raw


def _self_check():
    import os

    from tejas.cost_model.dataset import feature_columns, load_experiment_frame
    from tejas.paths import RESULTS_DIR

    path = os.path.join(str(RESULTS_DIR), "raw_results.csv")
    df, _, _ = load_experiment_frame(path)
    cat, num = feature_columns(("base", "sql_semantic", "interactions"))

    for kind in QUALITY_MODEL_KINDS:
        model = QualityModel(kind, cat, num, calibrate=(kind == "histgb"))
        model.fit(df, df["correct"], groups=df["task_id"])
        proba = model.predict_proba(df)
        assert np.all((proba >= 0) & (proba <= 1)), f"{kind}: probabilities out of [0,1]"
        acc = np.mean((proba >= 0.5) == df["correct"])
        print(f"{kind:>13}: in-sample accuracy@0.5 = {acc:.3f}, mean predicted P = {proba.mean():.3f}, "
              f"actual accuracy = {df['correct'].mean():.3f}")


if __name__ == "__main__":
    _self_check()
