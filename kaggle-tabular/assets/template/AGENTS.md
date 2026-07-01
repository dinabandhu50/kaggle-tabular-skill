# AGENTS.md — {{COMP_NAME}}

Cross-harness instructions (Claude Code / Codex / OpenCode). Read this, then follow the skill.

## Operating manual

This repo follows the **kaggle-tabular** skill. The authoritative workflow, hard rules, and phase
gates live in that skill's `SKILL.md` and `references/`. Obey them. In short:

- **Two foundations:** trustworthy local validation, and fast experiment throughput.
- **Validation-first:** build and verify the CV harness (Phase 1) BEFORE feature engineering.
- **OOF contract (HR-4):** build every model through `src/models/base.py::run_experiment`, which
  saves `oof/` + `preds/` arrays and appends an `experiments/ledger.parquet` row. No artifacts =
  it didn't happen.
- **Frozen folds (HR-2):** always use `data/folds.parquet`. Never re-derive folds.
- **Leakage boundary (HR-1):** all target-aware / cross-row preprocessing is fit inside the fold on
  training rows only — put it in the model's `fit_fold`, never on full data.
- **Decide on CV, never the public LB (HR-5).**

## ROI order (don't get this wrong)

Trustworthy CV → diverse baselines → feature engineering → *light* tuning → *serious* ensembling.
Do not heavily tune before FE has plateaued. Ensembling is where the leaderboard is won.

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

See `COMPETITION.md` (task, exact metric, CV decision, submission limit, deadlines).
