"""Reusable Optuna harness for Phase 5 (light tuning).

Trials score on the frozen folds directly — no per-trial ledger/OOF artifacts (HR-4 artifacts are
for the final config only, logged separately through `run_experiment`). Search spaces are declared
in `configs/optimize/<model>.yaml` and turned into a param_space via `param_space_from_config`, so
switching what's searched is a config change, not a code change.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ..metric import GREATER_IS_BETTER, cv_score

ParamSpace = Callable[[Any], dict[str, Any]]
FitFoldBuilder = Callable[[dict[str, Any]], Callable]


def param_space_from_config(cfg: dict[str, dict[str, Any]]) -> ParamSpace:
    """Build param_space(trial) -> dict from a YAML search-space config.

    Each entry: `name: {type: int|float|categorical, ...suggest_* kwargs}`, e.g.
        learning_rate: {type: float, low: 0.01, high: 0.1, log: true}
        num_leaves: {type: int, low: 15, high: 255}
        booster: {type: categorical, choices: [gbdt, dart]}
    """
    def _space(trial):
        params = {}
        for name, spec in cfg.items():
            spec = dict(spec)
            kind = spec.pop("type")
            if kind == "int":
                params[name] = trial.suggest_int(name, **spec)
            elif kind == "float":
                params[name] = trial.suggest_float(name, **spec)
            elif kind == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(f"unknown param type: {kind}")
        return params
    return _space


def _cv_for_params(fit_fold_builder: FitFoldBuilder, params: dict, train: pd.DataFrame,
                   y: np.ndarray, folds: np.ndarray, seed: int) -> float:
    fit_fold = fit_fold_builder(params)
    oof = np.full(len(train), np.nan)
    for f in sorted(np.unique(folds)):
        if f < 0:
            continue
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]
        val_pred, _, _ = fit_fold(
            train.iloc[tr_idx], y[tr_idx], train.iloc[va_idx], train.iloc[va_idx], int(f), seed
        )
        oof[va_idx] = val_pred
    return cv_score(oof, y, folds)["cv_score"]


def tune(
    fit_fold_builder: FitFoldBuilder,
    param_space: ParamSpace,
    *,
    train: pd.DataFrame,
    y: np.ndarray,
    folds: pd.Series | np.ndarray,
    n_trials: int = 30,
    seed: int = 42,
    study_name: str | None = None,
):
    """Search `param_space` over `n_trials`, each scored on the frozen folds. Returns the study —
    read `study.best_params` / `study.best_value`, then log the winning config via `run_experiment`.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    folds_arr = folds.to_numpy() if isinstance(folds, pd.Series) else np.asarray(folds)

    def objective(trial):
        return _cv_for_params(fit_fold_builder, param_space(trial), train, y, folds_arr, seed)

    direction = "maximize" if GREATER_IS_BETTER else "minimize"
    study = optuna.create_study(
        direction=direction, study_name=study_name, sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study
