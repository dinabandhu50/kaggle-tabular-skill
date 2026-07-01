#!/usr/bin/env python3
"""Scaffold a new tabular-competition repo that already enforces the OOF contract and leakage rules.

Usage:
    python scripts/scaffold_competition.py <comp-name> [--dest PATH] [--gpu]

Creates the directory layout from the skill's playbook, copies working template code into src/,
substitutes the competition name, and stubs the empty data/artifact dirs. After scaffolding, follow
the gated workflow starting at Phase 0 (see SKILL.md).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent.parent / "assets" / "template"

# directories that are created empty (artifacts/data are gitignored; ledger/folds are committed)
EMPTY_DIRS = [
    "data/raw", "data/processed", "oof", "preds", "experiments", "notebooks",
    "scripts", "configs/features",
]


def substitute(path: Path, comp: str) -> None:
    try:
        text = path.read_text()
    except (UnicodeDecodeError, IsADirectoryError):
        return
    if "{{COMP_NAME}}" in text:
        path.write_text(text.replace("{{COMP_NAME}}", comp))


def enable_gpu(dest: Path) -> None:
    """Uncomment GPU toggles in the model wrappers (exact commented markers -> live lines)."""
    swaps = {
        "src/models/lgbm.py": (
            '        # device_type="gpu",  # uncomment if scaffolded with --gpu',
            '        device_type="gpu",',
        ),
        "src/models/xgb.py": (
            '        # device="cuda",  # uncomment if scaffolded with --gpu',
            '        device="cuda",',
        ),
        "src/models/cat.py": (
            '        # task_type="GPU",  # uncomment if scaffolded with --gpu',
            '        task_type="GPU",',
        ),
    }
    for rel, (old, new) in swaps.items():
        p = dest / rel
        if p.exists():
            p.write_text(p.read_text().replace(old, new))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("comp_name", help="competition slug, e.g. playground-series-s6e3")
    ap.add_argument("--dest", default=".", help="parent directory to create the repo in")
    ap.add_argument("--gpu", action="store_true", help="default templates to GPU (RAPIDS) backends")
    args = ap.parse_args()

    if not TEMPLATE.exists():
        raise SystemExit(f"template not found at {TEMPLATE}")

    root = Path(args.dest).resolve() / f"comp-{args.comp_name}"
    if root.exists():
        raise SystemExit(f"refusing to overwrite existing {root}")

    # copy template tree (includes dotfiles)
    shutil.copytree(TEMPLATE, root)

    # create empty working dirs with .gitkeep
    for d in EMPTY_DIRS:
        p = root / d
        p.mkdir(parents=True, exist_ok=True)
        (p / ".gitkeep").touch()

    # substitute competition name everywhere
    for path in root.rglob("*"):
        if path.is_file():
            substitute(path, args.comp_name)

    if args.gpu:
        enable_gpu(root)

    print(f"scaffolded {root}")
    print("next:")
    print("  cd", root)
    print("  uv sync                 # Phase 0")
    print("  # add Kaggle creds to .envrc, then: just download")
    print("  # Phase 1: implement src/metric.py, then `just folds` and `just adval`")
    print("  # follow the gated workflow in the kaggle-tabular skill (references/workflow-phases.md)")


if __name__ == "__main__":
    main()
