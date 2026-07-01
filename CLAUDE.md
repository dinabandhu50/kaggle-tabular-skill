# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is **not** a competition entry — it is the **authoring workspace for a portable Claude Code skill** plus supporting research. There is no top-level `src/`, `data/`, or notebooks here; the modeling code lives inside a template that gets *scaffolded out* into a separate `comp-<name>/` repo.

Two products:

- **`plugins/kaggle-tabular/`** — a self-contained, cross-harness (Claude Code / Codex / OpenCode) skill: a grandmaster-grade, validation-first workflow for solo tabular Kaggle competitions. This is the deliverable.
- **`learnings/`** — distilled research: top-solution write-ups (S6E2), a prioritized action-item roadmap, and the user's own data-science framework (`my-pipeline.md`). Source material that informs the skill.

There is no build/lint/test step — this is Markdown + a template. Do not add tests (per project norms, correctness is model performance, not unit tests).

### Plugin packaging (root-is-plugin, like superpowers)

The repo root **is** the plugin. Per-harness manifests live at the root:
- `.claude-plugin/plugin.json` + `.claude-plugin/marketplace.json` (source `./`) — Claude Code.
- `.codex-plugin/plugin.json` (`skills: ./skills/`) — Codex; the skill is also symlinked into
  `~/.codex/skills/` for native discovery.
- `.opencode/plugins/kaggle-tabular.js` + `.opencode/INSTALL.md` + root `package.json` — OpenCode; the
  plugin's `config` hook registers `skills/` into `config.skills.paths`.

The skill itself lives at `skills/kaggle-tabular/` (`SKILL.md`, `references/`, `scripts/`, `assets/`).
Install everywhere with `install.sh` (Claude marketplace install + Codex symlink + OpenCode plugin
registration), or per-harness per `USAGE.md` / `.opencode/INSTALL.md`. Paths shown in the sections
below are relative to the skill directory `skills/kaggle-tabular/`.

## The `kaggle-tabular` skill — architecture

`kaggle-tabular/SKILL.md` is a **router**, not a monolith. It states the philosophy and the 8-phase gated workflow, then delegates detail to `references/*.md` that are loaded **on demand per phase**:

- `references/hard-rules.md` — the leakage constitution (HR-1…HR-7) + Validation Guardian audit. Read at Phase 1 and before accepting any experiment.
- `references/workflow-phases.md` — detailed Phase 0–7 actions/gates/owners.
- `references/feature-engineering.md` (Phase 4), `references/model-menu.md` (Phase 3), `references/ensembling.md` (Phase 6–7), `references/orchestration.md` (multi-agent planning).

The workflow is **phase-gated**: phases 0→7 do not advance until the current gate is met and logged. Bold/high-ROI phases are **1 (validation harness)**, **4 (feature engineering)**, and **6 (ensembling)** — *not* tuning. The ROI order is a deliberate correction of the common mistake of over-tuning: `trustworthy CV → diverse baselines → feature engineering → light tuning → serious ensembling`.

### The scaffolder + template (the second half of the skill)

`scripts/scaffold_competition.py <comp-name> [--dest PATH] [--gpu]` copies `assets/template/` into a new `comp-<comp-name>/` dir, creates empty gitignored working dirs (`data/`, `oof/`, `preds/`, `experiments/`, …), and substitutes the `{{COMP_NAME}}` placeholder everywhere.

The template ships **library code in `src/` that *encodes* the hard rules** so agents can't easily violate them — this is the central design idea:

- `src/models/base.py::run_experiment(...)` — **the keystone / OOF contract.** Every model must be built through it. It loads the frozen folds (HR-2), runs the CV loop, saves `oof/train_oof_<id>.npy` + `preds/test_preds_<id>.npy` (HR-4), and appends one ledger row. The per-fold leakage boundary (HR-1) lives in the caller's `fit_fold` callable, which receives raw per-fold frames and must fit all target-aware/cross-row transforms on train rows only.
- `src/ledger.py` — append-only `experiments/ledger.parquet`. This is the **agent-coordination layer**: parallel agents append without collision; the Validation Guardian flips each row's `status` to `kept` / `rejected:<reason>`; the ensembler reads only `kept` OOFs. Committed to git = the competition's memory.
- `src/cv.py` (folds + adversarial validation), `src/metric.py` (metric registry + `cv_score`, must be filled in per competition — HR-3), `src/ensemble.py` (hill climbing + stacking over OOFs).

The template `justfile` is the phase runner (`just folds`, `just baseline model=lgbm`, `just fe`, `just tune`, `just ensemble`, `just submit`, `just audit exp=<id>`). Note: these recipes call `scripts.make_folds`, `scripts.baseline`, etc. which are **not** in the template — the agent writes those thin phase scripts on top of the `src/` library during the competition.

## Editing conventions for this repo

- **Keep the skill and its template in sync.** When you change a rule or workflow step in `SKILL.md` / `references/`, also update where it is *encoded*: the template `src/` code and `assets/template/AGENTS.md` (the cross-harness in-repo instructions). The skill describes the discipline; the template enforces it — they must agree.
- **Template files are placeholders, not live code.** Use `{{COMP_NAME}}` for the competition slug; never hardcode a competition. The `--gpu` flag works by *uncommenting* specific marker lines (e.g. `# device_type="gpu",  # uncomment if scaffolded with --gpu`) — preserve that exact commented form so `enable_gpu()` in the scaffolder can find and flip it.
- **Hard rules are the invariant.** HR-1 (no target-aware preprocessing touching the validation fold), HR-2 (folds frozen once in `data/folds.parquet`), HR-4 (no artifacts = it didn't happen), HR-5 (decide on CV, never public LB). Any code or guidance change must not weaken these.
- The `learnings/` files are updated *after* implementing a technique — record what worked, what didn't, and the analysis. Add new learnings files as insights are discovered.