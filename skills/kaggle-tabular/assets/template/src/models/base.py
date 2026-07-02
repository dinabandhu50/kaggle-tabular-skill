"""The OOF-contract keystone.

Build EVERY model through `run_experiment`. It loads the frozen folds (HR-2), runs CV, saves the
OOF and test-prediction arrays (HR-4), and appends a ledger row. Doing this by hand is how leakage
and fold-mismatch creep in — route everything through here.

The leakage boundary (HR-1) lives in your `fit_fold` callable: it receives RAW per-fold frames and
must do all target-aware / cross-row preprocessing (encoders, scalers, imputers, target encoding)
*inside* the function on the training rows only, then apply to validation and test.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .. import ledger, tracking
from ..metric import competition_metric, cv_score

# fit_fold(X_tr, y_tr, X_val, X_test, fold, seed) -> (val_pred, test_pred_for_this_fold, model)
FitFold = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame, int, int], tuple]


def run_experiment(
    exp_id: str,
    model_family: str,
    fit_fold: FitFold,
    *,
    train: pd.DataFrame,
    y: np.ndarray,
    test: pd.DataFrame,
    folds: pd.Series,
    feature_groups: list[str],
    config: dict[str, Any] | None = None,
    seed: int = 42,
    notes: str = "",
    oof_dir: str | Path = "oof",
    preds_dir: str | Path = "preds",
    ledger_path: str | Path = "experiments/ledger.parquet",
) -> dict:
    """Run one CV experiment under the frozen folds and persist all artifacts.

    Returns a summary dict (also written to the ledger). Test predictions are averaged across folds;
    for a final full-data refit, do that separately in Phase 7.
    """
    oof_dir, preds_dir = Path(oof_dir), Path(preds_dir)
    oof_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    folds = folds.to_numpy() if isinstance(folds, pd.Series) else np.asarray(folds)
    oof = np.full(len(train), np.nan, dtype=float)
    test_pred = np.zeros(len(test), dtype=float)
    fold_ids = [f for f in sorted(np.unique(folds)) if f >= 0]

    pbar = tqdm(fold_ids, desc=exp_id, unit="fold")
    for f in pbar:
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]
        # NOTE (HR-1): fit_fold must fit every target-aware/cross-row transform on tr rows only.
        val_pred, test_pred_f, _ = fit_fold(
            train.iloc[tr_idx], y[tr_idx], train.iloc[va_idx], test, int(f), seed
        )
        oof[va_idx] = val_pred
        test_pred += np.asarray(test_pred_f, dtype=float) / len(fold_ids)
        pbar.set_postfix(fold_score=f"{competition_metric(y[va_idx], val_pred):.5f}")

    scores = cv_score(oof, y, folds)

    oof_path = oof_dir / f"train_oof_{exp_id}.npy"
    pred_path = preds_dir / f"test_preds_{exp_id}.npy"
    np.save(oof_path, oof)
    np.save(pred_path, test_pred)

    row = {
        "exp_id": exp_id,
        "model_family": model_family,
        "cv_score": scores["cv_score"],
        "cv_fold_scores": scores["cv_fold_scores"],
        "cv_std": scores["cv_std"],
        "feature_groups": feature_groups,
        "n_features": int(train.shape[1]),
        "params_hash": ledger.params_hash(config),
        "config_path": (config or {}).get("__path__"),
        "oof_path": str(oof_path),
        "pred_path": str(pred_path),
        "seed": seed,
        "notes": notes,
    }
    ledger.append_row(ledger_path, row)
    tracking.log_run(exp_id, model_family, scores, config)

    print(f"[{exp_id}] {model_family}  CV={scores['cv_score']:.6f}  "
          f"±{scores['cv_std']:.6f}  folds={scores['cv_fold_scores']}")
    return {**row, **scores, "oof": oof, "test_pred": test_pred}
