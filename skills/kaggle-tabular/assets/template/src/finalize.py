"""Phase-7 full-data refit (no validation fold at the very end).

Once features and params are frozen, retrain on 100% of train at ~1.25x the average per-fold
best_iteration, averaged over many seeds. More data -> examples seen proportionally fewer times ->
afford slightly more rounds. Evidence: top tabular solutions beat K-fold averaging with a ~1.25x
best-iteration, ~20-seed refit.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

# fit_full(X, y, X_test, n_rounds, seed) -> test_pred (np.ndarray, shape [n_test])
FitFull = Callable[..., np.ndarray]


def full_data_refit(fit_full: FitFull, best_iters: Iterable[int], X, y, X_test, *,
                    mult: float = 1.25, seeds: Iterable[int] = range(20)) -> np.ndarray:
    """Average test predictions over seeds from a full-data refit at mult x avg best_iteration."""
    best_iters = list(best_iters)
    n_rounds = int(round(float(np.mean(best_iters)) * mult)) if best_iters else 0
    seeds = list(seeds)
    preds = np.zeros(len(X_test), dtype=float)
    for s in seeds:
        preds += np.asarray(fit_full(X, y, X_test, n_rounds, s), dtype=float) / len(seeds)
    return preds
