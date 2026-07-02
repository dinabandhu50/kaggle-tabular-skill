"""CatBoost wrapper. Native categorical handling (Ordered TS) — the categorical specialist.
Same fit_fold contract as lgbm.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..device import has_gpu


def make_fit_fold(params: dict | None = None, iterations: int = 3000,
                  task: str = "classification"):
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    default = dict(
        depth=4, learning_rate=0.03, l2_leaf_reg=3.0, iterations=iterations,
        verbose=0, allow_writing_files=False,
        task_type="GPU" if has_gpu() else "CPU",  # auto-detected; --gpu scaffold flag pins "GPU"
    )
    params = {**default, **(params or {})}

    def fit_fold(X_tr: pd.DataFrame, y_tr: np.ndarray, X_val: pd.DataFrame,
                 X_test: pd.DataFrame, fold: int, seed: int):
        cat_cols = [c for c in X_tr.columns if pd.api.types.is_string_dtype(X_tr[c])]
        Model = CatBoostClassifier if task == "classification" else CatBoostRegressor
        model = Model(**params, random_seed=seed)
        model.fit(Pool(X_tr, y_tr, cat_features=cat_cols))
        pv = model.predict_proba(X_val)[:, 1] if task == "classification" else model.predict(X_val)
        pt = model.predict_proba(X_test)[:, 1] if task == "classification" else model.predict(X_test)
        return pv, pt, model

    return fit_fold
