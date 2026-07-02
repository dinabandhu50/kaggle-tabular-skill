"""XGBoost wrapper. Same fit_fold contract as lgbm.py. HR-1: any target-aware transform is fit
inside fit_fold on X_tr only (see src/features.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..device import has_gpu


def make_fit_fold(params: dict | None = None, num_boost_round: int = 2000,
                  early_stopping: int = 100, task: str = "classification"):
    import xgboost as xgb

    default = dict(
        max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.0, reg_lambda=1.0,
        objective="binary:logistic" if task == "classification" else "reg:squarederror",
        eval_metric="auc" if task == "classification" else "rmse",
        tree_method="hist",
        device="cuda" if has_gpu() else "cpu",  # auto-detected; --gpu scaffold flag pins "cuda"
    )
    params = {**default, **(params or {})}

    def fit_fold(X_tr: pd.DataFrame, y_tr: np.ndarray, X_val: pd.DataFrame,
                 X_test: pd.DataFrame, fold: int, seed: int):
        cat_cols = [c for c in X_tr.columns if pd.api.types.is_string_dtype(X_tr[c])]
        X_tr, X_val, X_test = X_tr.copy(), X_val.copy(), X_test.copy()
        for c in cat_cols:  # enable_categorical=True requires actual category dtype
            X_tr[c] = X_tr[c].astype("category")
            X_val[c] = X_val[c].astype("category")
            X_test[c] = X_test[c].astype("category")

        dtr = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dval = xgb.DMatrix(X_val, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, enable_categorical=True)
        model = xgb.train({**params, "seed": seed}, dtr, num_boost_round=num_boost_round)
        return model.predict(dval), model.predict(dtest), model

    return fit_fold
