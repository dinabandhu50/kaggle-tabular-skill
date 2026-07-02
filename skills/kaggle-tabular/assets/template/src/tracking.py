"""Optional Weights & Biases logging — off unless WANDB_PROJECT is set.

The ledger (HR-4) is the durable experiment record regardless of this flag; wandb is a visualization
layer on top, never the source of truth, so a wandb outage/misconfiguration must never block a run.
To disable without unsetting the project (e.g. wandb is slowing runs down), set `WANDB_MODE=disabled`
or `WANDB_MODE=offline` — wandb reads that itself.
"""
from __future__ import annotations

import os
from typing import Any


def log_run(exp_id: str, model_family: str, scores: dict[str, Any],
           config: dict[str, Any] | None = None) -> None:
    project = os.environ.get("WANDB_PROJECT")
    if not project:
        return
    try:
        import wandb
        run = wandb.init(project=project, name=exp_id, config=config or {}, reinit=True)
        run.log({
            "cv_score": scores["cv_score"],
            "cv_std": scores["cv_std"],
            "model_family": model_family,
            **{f"fold_{i}": s for i, s in enumerate(scores["cv_fold_scores"])},
        })
        run.finish()
    except Exception as e:
        print(f"[tracking] wandb logging skipped: {e}")
