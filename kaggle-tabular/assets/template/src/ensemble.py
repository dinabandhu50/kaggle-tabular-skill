"""Ensembling over the OOF ledger: hill climbing + stacking.

Reads `kept` experiments, loads their OOF/test arrays, and combines them. Hill climbing is the
robust workhorse; stacking adds a meta-model when base models capture different structure. All
operates on OOF predictions aligned to the frozen folds (HR-2) — never on in-fold-fit predictions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import ledger
from .metric import GREATER_IS_BETTER, competition_metric, is_improvement


def load_oof_matrix(ledger_path="experiments/ledger.parquet"):
    """Return (exp_ids, OOF matrix [n_rows x n_models], TEST matrix [n_test x n_models])."""
    df = ledger.kept(ledger_path)
    if df.empty:
        raise ValueError("No 'kept' experiments. Run the Guardian audit first.")
    oof_cols, test_cols, ids = [], [], []
    for _, r in df.iterrows():
        oof_cols.append(np.load(r["oof_path"]))
        test_cols.append(np.load(r["pred_path"]))
        ids.append(r["exp_id"])
    return ids, np.vstack(oof_cols).T, np.vstack(test_cols).T


def hill_climb(oof: np.ndarray, test: np.ndarray, y: np.ndarray, *, n_iter=100, init_best=True):
    """Greedy weighted blend (with replacement). Adds the member that most improves OOF each step.

    Returns (weights[n_models], oof_blend, test_blend, history). Robust and hard to overfit.
    """
    n_models = oof.shape[1]
    scores = [competition_metric(y, oof[:, j]) for j in range(n_models)]
    chosen = [int(np.argmax(scores) if GREATER_IS_BETTER else np.argmin(scores))] if init_best else []

    def blend(idx):
        return oof[:, idx].mean(axis=1) if idx else np.zeros(len(y))

    best = competition_metric(y, blend(chosen)) if chosen else (-np.inf if GREATER_IS_BETTER else np.inf)
    history = [best]
    for _ in range(n_iter):
        cand_scores = []
        for j in range(n_models):
            s = competition_metric(y, blend(chosen + [j]))
            cand_scores.append(s)
        j_best = int(np.argmax(cand_scores) if GREATER_IS_BETTER else np.argmin(cand_scores))
        if is_improvement(cand_scores[j_best], best):
            chosen.append(j_best)
            best = cand_scores[j_best]
            history.append(best)
        else:
            break

    weights = np.bincount(chosen, minlength=n_models).astype(float)
    weights /= weights.sum()
    oof_blend = oof @ weights
    test_blend = test @ weights
    return weights, oof_blend, test_blend, history


def stack(oof: np.ndarray, test: np.ndarray, y: np.ndarray, folds: np.ndarray, *, kind="ridge"):
    """Level-2 meta-model on the OOF matrix, CV'd on the SAME folds (HR-2).

    kind: 'ridge' (regression) | 'logistic' (classification) | 'lgbm'. Start simple; escalate only
    if each step beats the prior on OOF.
    """
    folds = np.asarray(folds)
    meta_oof = np.full(len(y), np.nan)
    test_acc = np.zeros(test.shape[0])
    fold_ids = [f for f in sorted(np.unique(folds)) if f >= 0]

    def make():
        if kind == "ridge":
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0)
        if kind == "logistic":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, C=1.0)
        if kind == "lgbm":
            import lightgbm as lgb
            return lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                     num_leaves=15, verbosity=-1)
        raise ValueError(kind)

    for f in fold_ids:
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = make()
        if kind == "logistic":
            m.fit(oof[tr], y[tr])
            meta_oof[va] = m.predict_proba(oof[va])[:, 1]
            test_acc += m.predict_proba(test)[:, 1] / len(fold_ids)
        else:
            m.fit(oof[tr], y[tr])
            meta_oof[va] = m.predict(oof[va])
            test_acc += m.predict(test) / len(fold_ids)

    valid = folds >= 0
    return {
        "kind": kind,
        "oof_score": float(competition_metric(y[valid], meta_oof[valid])),
        "meta_oof": meta_oof,
        "test_pred": test_acc,
    }


def save_spec(path, *, method, members, weights=None, meta_kind=None, oof_score=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "member": members,
        "weight": weights if weights is not None else [None] * len(members),
    }).to_csv(path, index=False)
    print(f"[ensemble] method={method} meta={meta_kind} OOF={oof_score} -> {path}")
