"""Leak-safe feature helpers (HR-1).

Every function here is designed to be called INSIDE a model's `fit_fold`, fit on the fold's
TRAINING rows only, then applied to validation and test. Fitting any of these on full data (or on
data that includes the validation fold) is an HR-1 violation and inflates CV.

Competition-agnostic: pass the column lists explicitly; nothing here hardcodes a dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class OOFTargetEncoder:
    """Smoothed target encoding, fit on training rows only.

    Usage inside fit_fold:
        te = OOFTargetEncoder(cols, smoothing=20.0).fit(X_tr, y_tr)
        X_tr = te.transform(X_tr); X_val = te.transform(X_val); X_test = te.transform(X_test)
    """

    def __init__(self, cols: list[str], smoothing: float = 20.0, suffix: str = "_te"):
        self.cols = cols
        self.smoothing = smoothing
        self.suffix = suffix
        self.global_mean_: float = 0.0
        self.maps_: dict[str, pd.Series] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "OOFTargetEncoder":
        y = np.asarray(y, dtype=float)
        self.global_mean_ = float(y.mean())
        df = X[self.cols].copy()
        df["__y__"] = y
        for c in self.cols:
            stats = df.groupby(c)["__y__"].agg(["mean", "count"])
            smooth = (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_) / (
                stats["count"] + self.smoothing
            )
            self.maps_[c] = smooth
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.cols:
            X[c + self.suffix] = X[c].map(self.maps_[c]).astype(float).fillna(self.global_mean_)
        return X


def frequency_encode(X_tr: pd.DataFrame, X_others: list[pd.DataFrame], cols: list[str],
                     suffix: str = "_freq") -> list[pd.DataFrame]:
    """Map each category to its TRAIN-row frequency. Returns transformed [X_tr, *X_others]."""
    frames = [X_tr] + list(X_others)
    frames = [f.copy() for f in frames]
    for c in cols:
        freq = X_tr[c].value_counts()
        for f in frames:
            f[c + suffix] = f[c].map(freq).fillna(0).astype(float)
    return frames


def all_cats(X: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Cast columns (default: all) to pandas 'category' so trees split on levels."""
    X = X.copy()
    cols = cols or list(X.columns)
    for c in cols:
        X[c] = X[c].astype(str).astype("category")
    return X


def quantile_bins(X_tr: pd.DataFrame, X_others: list[pd.DataFrame], cols: list[str], q: int = 10,
                  suffix: str = "_qbin") -> list[pd.DataFrame]:
    """Percentile bins with edges learned on TRAIN rows only. Returns [X_tr, *X_others]."""
    frames = [f.copy() for f in ([X_tr] + list(X_others))]
    for c in cols:
        _, edges = pd.qcut(X_tr[c], q=q, retbins=True, duplicates="drop", labels=False)
        for f in frames:
            f[c + suffix] = pd.cut(f[c], bins=edges, labels=False, include_lowest=True)
            f[c + suffix] = f[c + suffix].fillna(-1).astype(int)
    return frames


def digit_features(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Extract units/tens digits of integer-valued numerics (exposes generator structure)."""
    X = X.copy()
    for c in cols:
        v = X[c].fillna(0).astype(int)
        X[c + "_units"] = v % 10
        X[c + "_tens"] = (v // 10) % 10
    return X


def categorical_interactions(X: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Concatenate categorical pairs into new categorical columns (then target-encode in-fold)."""
    X = X.copy()
    for a, b in pairs:
        X[f"{a}_{b}"] = X[a].astype(str) + "_" + X[b].astype(str)
    return X
