"""Energy prediction ladder: E_hat(task, model, deployment) -> joules.

global_mean / tier_lookup are baselines operating directly on energy_j
(arithmetic domain) — the bar the learned models (ridge/elasticnet/histgb)
must clear to be worth using. The learned models fit log(energy_j) (energy
is heavily right-skewed across a 3b-70b range) and back-transform with a
training-fold-only Duan smearing correction (`c = mean(exp(residuals))`,
`E_hat = c * exp(logE_hat)`) rather than bare exp(), which would
systematically underpredict the arithmetic mean (geometric-mean bias).
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ENERGY_MODEL_KINDS = ["global_mean", "tier_lookup", "ridge", "elasticnet", "histgb"]

_REGRESSORS = {
    "ridge": lambda: Ridge(alpha=1.0),
    "elasticnet": lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000),
    "histgb": lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=100, random_state=0),
}


def _column_transformer(categorical_cols, numeric_cols):
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("numeric", StandardScaler(), numeric_cols),
    ])


class EnergyModel:
    def __init__(self, kind, categorical_cols, numeric_cols):
        assert kind in ENERGY_MODEL_KINDS, kind
        self.kind = kind
        self.categorical_cols = list(categorical_cols)
        self.numeric_cols = list(numeric_cols)
        self._global_mean = None
        self._tier_means = None
        self._pipeline = None
        self._smear_c = 1.0

    def fit(self, df, energy_j):
        if df["measurement_scope"].nunique() > 1:
            raise ValueError(
                "EnergyModel (absolute-J) received rows spanning multiple "
                "measurement_scope values — absolute joules aren't comparable "
                "across measurement boundaries (see deployment_features.py). "
                "Use a ratio/relative target for cross-scope pooling instead."
            )
        energy_j = np.asarray(energy_j, dtype=float)

        if self.kind == "global_mean":
            self._global_mean = energy_j.mean()
            return self

        if self.kind == "tier_lookup":
            groups = {}
            for k, e in zip(zip(df["deployment"], df["model_tier"]), energy_j):
                groups.setdefault(k, []).append(e)
            self._tier_means = {k: float(np.mean(v)) for k, v in groups.items()}
            self._global_mean = float(energy_j.mean())
            return self

        pipeline = Pipeline([
            ("features", _column_transformer(self.categorical_cols, self.numeric_cols)),
            ("model", _REGRESSORS[self.kind]()),
        ])
        log_energy = np.log(energy_j)
        pipeline.fit(df, log_energy)
        resid = log_energy - pipeline.predict(df)
        self._smear_c = float(np.mean(np.exp(resid)))  # Duan smearing, training-fold-only
        self._pipeline = pipeline
        return self

    def predict(self, df):
        if self.kind == "global_mean":
            return np.full(len(df), self._global_mean)
        if self.kind == "tier_lookup":
            key = list(zip(df["deployment"], df["model_tier"]))
            return np.array([self._tier_means.get(k, self._global_mean) for k in key])
        log_pred = self._pipeline.predict(df)
        return self._smear_c * np.exp(log_pred)


def _self_check():
    import os

    from tejas.cost_model.dataset import feature_columns, load_experiment_frame
    from tejas.paths import RESULTS_DIR

    path = os.path.join(str(RESULTS_DIR), "raw_results.csv")
    df, _, _ = load_experiment_frame(path)
    cat, num = feature_columns(("base", "sql_semantic", "interactions"))

    for kind in ENERGY_MODEL_KINDS:
        model = EnergyModel(kind, cat, num)
        model.fit(df, df["energy_j"])
        pred = model.predict(df)
        assert np.all(np.isfinite(pred)), f"{kind}: non-finite predictions"
        assert np.all(pred > 0), f"{kind}: non-positive predictions"
        mae = np.mean(np.abs(pred - df["energy_j"]))
        print(f"{kind:>12}: in-sample MAE = {mae:.2f} J")

    # cross-scope pooling guard
    df2 = df.copy()
    df2.loc[df2.index[0], "measurement_scope"] = "gpu_die_only"
    try:
        EnergyModel("ridge", cat, num).fit(df2, df2["energy_j"])
        raise AssertionError("expected ValueError for mixed measurement_scope")
    except ValueError:
        print("measurement_scope pooling guard OK")


if __name__ == "__main__":
    _self_check()
