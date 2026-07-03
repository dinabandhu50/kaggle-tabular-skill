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

- **No throwaway code for anything that drives a decision.** Never use `python -c "..."` or a
  heredoc for FE/modeling/ensembling exploration. Every FE idea is planned first
  (`experiments/<model>/specs/NNN_<slug>.md`), then implemented in
  `src/feature_engineering/<model>/NNN_<slug>.py`, with tunable knobs in
  `configs/features/<model>/NNN_<slug>.yaml` — see the skill's `references/feature-engineering.md`.
  `python -c` is fine only for a tiny test-and-forget check. Anything worth keeping goes in
  `localdev/tmp/*.py`; captured output/logs go through `localdev/logs/`.
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
- **Commit at meaningful steps.** After a new FE group/model variant (`feat: ...`), an ensemble
  analysis or notable result (`exp: ...`), a bug fix (`fix: ...`), or a `PROGRESS.md` update
  (`docs: ...`) — see the skill's `SKILL.md` → "Commit at meaningful steps" for the full convention.
- **Plan before executing, in small batches.** See the skill's `references/orchestration.md` →
  "Daily experiment planning cadence": review state, pick a small batch of hypotheses, spec each one,
  dispatch subagents in parallel where independent, verify against `cv_std` + the Guardian audit,
  then decide the next batch from evidence.
- **Mine the competition's own community.** Overview/Discussion/Code URLs live in `COMPETITION.md`;
  periodically search the web for them and write notes to `localdev/external/*.md` — see
  `references/orchestration.md` → "External intel gathering".
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

See `COMPETITION.md` (task, exact metric, CV decision, submission limit, deadlines, external links).
See `PROGRESS.md` for the running log of what's been tried, kept, dropped, and submitted.

## Scratch space

`localdev/tmp/` (disposable scripts), `localdev/logs/` (captured output), `localdev/external/`
(notes distilled from the competition's Discussion/Code pages — committed; `tmp/` and `logs/` are
not). None of it is part of the OOF contract.
