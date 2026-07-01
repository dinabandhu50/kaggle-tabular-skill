"""Competition metric (HR-3).

Fill in `competition_metric` with the EXACT metric the leaderboard uses, then prove it: a trivial
submission (all-mean / all-majority) computed here must reproduce its public-LB score within
tolerance. A CV computed with the wrong metric optimizes the wrong thing.

`GREATER_IS_BETTER` must be set correctly — the ensembler and gates rely on it.
"""
from __future__ import annotations

import numpy as np

# ---- SET THESE FOR THE COMPETITION -----------------------------------------------------------
GREATER_IS_BETTER = True   # AUC/accuracy/MAP -> True ; RMSE/LogLoss/MAE -> False
# ----------------------------------------------------------------------------------------------


def competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Replace the body with the competition's exact metric.

    Examples (uncomment / adapt one):

        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_pred)              # AUC (GREATER_IS_BETTER=True)

        from sklearn.metrics import mean_squared_error
        return mean_squared_error(y_true, y_pred, squared=False)  # RMSE (False)

        from sklearn.metrics import log_loss
        return log_loss(y_true, y_pred)                   # LogLoss (False)
    """
    raise NotImplementedError("Implement the exact competition metric, then verify against the LB.")


def cv_score(oof: np.ndarray, y_true: np.ndarray, folds: np.ndarray) -> dict:
    """Compute overall and per-fold OOF scores under the competition metric."""
    per_fold = []
    for f in sorted(np.unique(folds)):
        if f < 0:
            continue  # time-scheme rows with no training history
        mask = folds == f
        per_fold.append(competition_metric(y_true[mask], oof[mask]))
    valid = folds >= 0
    overall = competition_metric(y_true[valid], oof[valid])
    return {
        "cv_score": float(overall),
        "cv_fold_scores": [float(s) for s in per_fold],
        "cv_std": float(np.std(per_fold)) if per_fold else 0.0,
    }


def is_improvement(new: float, best: float) -> bool:
    return new > best if GREATER_IS_BETTER else new < best
