# AGENTS.md — {{COMP_NAME}}

Cross-harness instructions (Claude Code / Codex / OpenCode). Read this, then follow the skill.

## Operating manual

This repo follows the **kaggle-tabular** skill. The authoritative workflow, hard rules, and phase
gates live in that skill's `SKILL.md` and `references/`. Obey them. In short:

- **Two foundations:** trustworthy local validation, and fast experiment throughput.
- **Validation-first:** build and verify the CV harness (Phase 1) BEFORE feature engineering.
- **OOF contract (HR-4):** build every model through `src/models/base.py::run_experiment`, which
  saves `oof/` + `preds/` arrays and appends an `experiments/ledger.parquet` row. No artifacts =
  it didn't happen. It also logs to wandb if `WANDB_PROJECT` is set — that's a visualization layer,
  not the source of truth; the ledger always wins. Set `WANDB_MODE=disabled` (or unset
  `WANDB_PROJECT`) if wandb is slowing runs down; the ledger keeps tracking regardless.
- **Frozen folds (HR-2):** always use `data/folds.parquet`. Never re-derive folds.
- **Leakage boundary (HR-1):** all target-aware / cross-row preprocessing is fit inside the fold on
  training rows only — put it in the model's `fit_fold`, never on full data.
- **Decide on CV, never the public LB (HR-5).**

## ROI order (don't get this wrong)

Trustworthy CV → diverse baselines → feature engineering → *light* tuning → *serious* ensembling.
Do not heavily tune before FE has plateaued. Ensembling is where the leaderboard is won.

## Engineering conventions

- **No throwaway code.** Never use `python -c "..."` or a heredoc for anything that produces a
  result you'll rely on. Every FE idea is planned first (`experiments/<model>/specs/NNN_<slug>.md`),
  then implemented in `src/feature_engineering/<model>/NNN_<slug>.py`, with tunable knobs in
  `configs/features/<model>/NNN_<slug>.yaml` — see the skill's `references/feature-engineering.md`.
- **`scripts/` is standalone utilities only** (download, folds, adversarial, eda, baseline, fe, tune,
  ensemble, submit, audit, summary) — no feature-engineering logic there; that's
  `src/feature_engineering/`.
- **Reuse `src/`.** Phase scripts are thin wrappers over `run_experiment`, `src/features.py`,
  `src/feature_engineering/`, `src/optimize/optuna_search.py`, `src/ensemble.py`. Don't reimplement
  CV loops, encoders, or Optuna boilerplate inline.
- **GPU by default.** `src/device.py::has_gpu()` auto-detects CUDA; no flag needed. `--gpu` at
  scaffold time only pins it (skips the runtime check).
- **Parallelize.** Baselines and per-family FE loops are embarrassingly parallel — one subagent per
  model family, not serial.
- **Visible progress.** Training shows a fold-level progress bar (already wired into
  `run_experiment`); long-running agents print one line per experiment start/finish.
- **Track progress.** After every kept/rejected experiment and every submission, append one terse
  line to `PROGRESS.md`.
- **Terse code.** No file-header docstrings restating what the code does; comment only a non-obvious
  constraint.

## Agent roles (see the skill's references/orchestration.md)

- **Validation Guardian** — owns folds + metric; audits every experiment against the Hard Rules
  before it is marked `kept`. This is an adversarial role: the agent that produced an experiment
  must not be the one that certifies it.
- **Baseline / FE-explorer agents** — one per model family, run in parallel; coordinate only through
  the append-only ledger.
- **Ensembler** — reads `kept` OOFs; hill-climbs and stacks.

## Commands

`just setup` · `just download` · `just folds` · `just adval` · `just eda` ·
`just baseline model=lgbm` · `just fe model=lgbm group=<g>` · `just tune model=lgbm` ·
`just ensemble` · `just submit spec=best` · `just audit exp=<id>` · `just summary`

## Competition facts

See `COMPETITION.md` (task, exact metric, CV decision, submission limit, deadlines). See
`PROGRESS.md` for the running log of what's been tried, kept, dropped, and submitted.
