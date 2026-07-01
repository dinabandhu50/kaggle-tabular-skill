"""Validation harness: fold generation (HR-2) and adversarial validation.

Generate folds ONCE at Phase 1 and persist to data/folds.parquet. Every model, experiment, and
stacking level then reuses the identical split. Re-deriving folds elsewhere is an HR-2 violation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit


def make_folds(
    df: pd.DataFrame,
    *,
    scheme: str,            # 'stratified' | 'kfold' | 'group' | 'time'
    n_folds: int = 5,
    target_col: str | None = None,
    group_col: str | None = None,
    time_col: str | None = None,
    seed: int = 42,
) -> pd.Series:
    """Return an integer fold label per row. Choose `scheme` by TEST STRUCTURE, not habit:

      stratified : i.i.d. classification (preserves class balance per fold)
      kfold      : i.i.d. regression
      group      : grouped entities (user/patient/store) — an entity never spans folds
      time       : temporal data — never shuffle across time
    """
    folds = np.full(len(df), -1, dtype=int)

    if scheme == "stratified":
        assert target_col, "stratified needs target_col"
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for i, (_, val) in enumerate(skf.split(df, df[target_col])):
            folds[val] = i
    elif scheme == "kfold":
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for i, (_, val) in enumerate(kf.split(df)):
            folds[val] = i
    elif scheme == "group":
        assert group_col, "group needs group_col"
        gkf = GroupKFold(n_splits=n_folds)
        for i, (_, val) in enumerate(gkf.split(df, groups=df[group_col])):
            folds[val] = i
    elif scheme == "time":
        assert time_col, "time needs time_col"
        order = df[time_col].argsort().to_numpy()
        tss = TimeSeriesSplit(n_splits=n_folds)
        for i, (_, val) in enumerate(tss.split(order)):
            folds[order[val]] = i  # earliest split has fold -1 (no train history); drop or keep
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    return pd.Series(folds, index=df.index, name="fold")


def save_folds(folds: pd.Series, path: str | Path = "data/folds.parquet") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_frame().to_parquet(path)


def load_folds(path: str | Path = "data/folds.parquet") -> pd.Series:
    return pd.read_parquet(path)["fold"]


def adversarial_validation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_cols: list[str],
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Train a classifier to distinguish train (0) from test (1).

    AUC ~ 0.5  -> train and test are drawn alike; random K-fold is trustworthy.
    AUC >> 0.5 -> distribution shift; inspect top features, prefer time/group folds, consider
                  weighting train rows by test-likeness or selecting a test-like validation subset.

    Returns {'auc': float, 'feature_importance': Series sorted desc}.
    """
    try:
        import lightgbm as lgb
    except ImportError as e:  # pragma: no cover
        raise ImportError("adversarial_validation needs lightgbm") from e
    from sklearn.metrics import roc_auc_score

    a = train[feature_cols].copy()
    b = test[feature_cols].copy()
    a["__is_test__"] = 0
    b["__is_test__"] = 1
    both = pd.concat([a, b], ignore_index=True)
    y = both.pop("__is_test__").to_numpy()
    X = both

    # encode object columns as categ: LightGBM handles them natively
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category")

    oof = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    importances = np.zeros(len(feature_cols))
    for tr, va in skf.split(X, y):
        m = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, verbosity=-1,
        )
        m.fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        importances += m.feature_importances_ / n_folds

    auc = roc_auc_score(y, oof)
    fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    return {"auc": float(auc), "feature_importance": fi}
