"""Example model wrapper: LightGBM.

Shows the `fit_fold` contract that `run_experiment` expects. The key discipline (HR-1): any
target-aware or cross-row preprocessing happens HERE, fit on the fold's training rows only, then
applied to validation and test. This wrapper uses LightGBM's native categorical handling, so it has
no separate encoder to leak; when you add OOF target encoding, scaling, or imputation, fit them
inside this function on X_tr only.

Copy this file per family (xgb.py, cat.py, ...) and swap the estimator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_fit_fold(params: dict | None = None, num_boost_round: int = 2000,
                  early_stopping: int = 100, task: str = "classification"):
    import lightgbm as lgb

    default = dict(
        objective="binary" if task == "classification" else "regression",
        learning_rate=0.03, num_leaves=63, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=1, min_child_samples=20, reg_alpha=0.0, reg_lambda=1.0,
        verbosity=-1,
        # device_type="gpu",  # uncomment if scaffolded with --gpu
    )
    params = {**default, **(params or {})}

    def fit_fold(X_tr: pd.DataFrame, y_tr: np.ndarray, X_val: pd.DataFrame,
                 X_test: pd.DataFrame, fold: int, seed: int):
        cat_cols = [c for c in X_tr.columns if str(X_tr[c].dtype) in ("object", "category")]
        for c in cat_cols:  # consistent category dtype across splits
            X_tr[c] = X_tr[c].astype("category")
            X_val[c] = X_val[c].astype("category")
            X_test = X_test.copy()
            X_test[c] = X_test[c].astype("category")

        dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols or "auto")
        dval = lgb.Dataset(X_val, label=None, reference=dtr)  # labels not needed for prediction
        model = lgb.train(
            {**params, "seed": seed},
            dtr, num_boost_round=num_boost_round,
            valid_sets=[dtr], callbacks=[lgb.log_evaluation(0)],
        )
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        return val_pred, test_pred, model

    return fit_fold
