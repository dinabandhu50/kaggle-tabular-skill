"""Logistic/Linear + one-hot wrapper. Doubles as the Phase-2 LINEARITY PROBE: if competitive with
the GBDT baseline, the signal is near-linear -> prefer shallow trees + diversity. HR-1: the encoder
and scaler are fit on X_tr only, inside fit_fold."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_fit_fold(params: dict | None = None, task: str = "classification"):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    params = params or {}

    def fit_fold(X_tr: pd.DataFrame, y_tr: np.ndarray, X_val: pd.DataFrame,
                 X_test: pd.DataFrame, fold: int, seed: int):
        cat_cols = [c for c in X_tr.columns if pd.api.types.is_string_dtype(X_tr[c])]
        num_cols = [c for c in X_tr.columns if c not in cat_cols]
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ])
        est = (LogisticRegression(max_iter=2000, C=params.get("C", 1.0))
               if task == "classification" else Ridge(alpha=params.get("alpha", 1.0)))
        pipe = Pipeline([("pre", pre), ("est", est)])
        pipe.fit(X_tr, y_tr)
        if task == "classification":
            return pipe.predict_proba(X_val)[:, 1], pipe.predict_proba(X_test)[:, 1], pipe
        return pipe.predict(X_val), pipe.predict(X_test), pipe

    return fit_fold
