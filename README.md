# kaggle-tabular

A **grandmaster-grade, validation-first workflow** for solo tabular Kaggle competitions, packaged as a
**cross-harness skill/plugin** for **Claude Code, Codex, and OpenCode**. It ships the discipline (leakage
hard-rules, a gated 8-phase loop, an OOF artifact contract) *and* the code that encodes it (a scaffolder
plus a template `src/` that agents build every experiment through).

This repo is **root-is-plugin** (same layout as [superpowers](https://github.com/obra/superpowers)).

## What's inside

| Path | What it is |
|---|---|
| `skills/kaggle-tabular/SKILL.md` | The router: philosophy, ROI order, HR-1…HR-7, the 8-phase gated workflow |
| `skills/kaggle-tabular/references/` | On-demand detail per phase (hard-rules, workflow-phases, feature-engineering, model-menu, ensembling, orchestration, what-works) |
| `skills/kaggle-tabular/scripts/scaffold_competition.py` | Scaffolds a `comp-<name>/` repo that encodes the rules |
| `skills/kaggle-tabular/assets/template/` | The competition template: `run_experiment` OOF keystone, append-only ledger, frozen folds, leak-safe `features.py`, `ensemble.py`, `finalize.py`, and `lgbm`/`xgb`/`cat`/`logreg` wrappers |
| `.claude-plugin/` · `.codex-plugin/` · `.opencode/` | Per-harness manifests |
| `learnings/` | Research: distilled top-solution write-ups + the refined pipeline (not shipped as part of the skill) |

## Install

```bash
bash install.sh          # wires Claude Code + Codex + OpenCode (use --dry-run to preview)
```

Per-harness detail and the per-competition flow are in [`USAGE.md`](USAGE.md); OpenCode specifics in
[`.opencode/INSTALL.md`](.opencode/INSTALL.md).

## The workflow in one line

Trustworthy CV → diverse baselines → hypothesis-driven feature engineering (per family, OOF-logged) →
*light* tuning → hill-climb + stack the diverse OOFs → seed-ensemble + full-data refit → **submit by CV**.
Every step emits OOF + test preds + a ledger row, audited against the hard rules.

## The hard rules (never weakened)

HR-1 no target-aware preprocessing touching the validation fold · HR-2 frozen folds · HR-3 exact metric
proven vs a known LB point · HR-4 every experiment saves OOF+preds+ledger row · HR-5 decide on CV, never
public LB · HR-6 log seeds/versions · HR-7 inference-time-available features.

## License

MIT
