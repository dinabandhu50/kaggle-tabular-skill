"""Append-only experiment ledger — the agent-coordination layer.

Every experiment appends exactly one row here (HR-4). Parallel agents append without collision.
The ensembler/summarizer read this to discover models. The ledger is committed to git: it is the
memory of the competition run.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

LEDGER_COLUMNS = [
    "exp_id", "timestamp", "model_family", "cv_score", "cv_fold_scores", "cv_std",
    "feature_groups", "n_features", "params_hash", "config_path", "oof_path", "pred_path",
    "seed", "git_sha", "lb_public", "status", "notes",
]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


def params_hash(config: dict[str, Any] | None) -> str:
    blob = json.dumps(config or {}, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def append_row(ledger_path: str | Path, row: dict[str, Any]) -> None:
    """Append one experiment row. Creates the ledger if missing."""
    ledger_path = Path(ledger_path)
    row = {**row}
    row.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    row.setdefault("git_sha", _git_sha())
    row.setdefault("status", "unaudited")  # Guardian flips to 'kept' or 'rejected:<reason>'
    row.setdefault("lb_public", None)
    # normalize list-valued fields to JSON strings for stable parquet round-trips
    for k in ("cv_fold_scores", "feature_groups"):
        if isinstance(row.get(k), (list, tuple)):
            row[k] = json.dumps(list(row[k]))
    full = {c: row.get(c) for c in LEDGER_COLUMNS}

    if ledger_path.exists():
        df = pd.read_parquet(ledger_path)
        df = pd.concat([df, pd.DataFrame([full])], ignore_index=True)
    else:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([full], columns=LEDGER_COLUMNS)
    df.to_parquet(ledger_path, index=False)


def load(ledger_path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(ledger_path)


def kept(ledger_path: str | Path) -> pd.DataFrame:
    """Experiments the Validation Guardian has accepted — the ensembling candidate pool."""
    df = load(ledger_path)
    return df[df["status"] == "kept"].reset_index(drop=True)


def set_status(ledger_path: str | Path, exp_id: str, status: str) -> None:
    """Guardian audit outcome: 'kept' or 'rejected:<reason>'."""
    df = load(ledger_path)
    df.loc[df["exp_id"] == exp_id, "status"] = status
    df.to_parquet(ledger_path, index=False)


def set_lb(ledger_path: str | Path, exp_id: str, lb_public: float) -> None:
    df = load(ledger_path)
    df.loc[df["exp_id"] == exp_id, "lb_public"] = lb_public
    df.to_parquet(ledger_path, index=False)
