# Design: `kaggle-tabular` Grandmaster Plugin

**Date:** 2026-07-01
**Status:** Approved (design phase) — pending spec review before writing-plans

## Purpose

Turn the existing `kaggle-tabular/` skill draft plus the distilled `learnings/refined-pipeline.md`
into a **reusable, installable plugin** — packaged like `superpowers` — that gives an AI coding agent
a grandmaster-grade, validation-first workflow for solo tabular Kaggle competitions, and works across
**Claude Code, Codex, and OpenCode**.

The plugin is competition-agnostic: it ships the *discipline* (hard rules, phase gates, OOF contract)
and *reusable code that encodes the discipline* (a scaffolder + template `src/`), never a specific
competition's data or feature names. Competition-specific evidence (S6E2 heart-disease numbers) is
retained only as **cited proof** inside references, not as hardcoded logic.

## Decisions (locked during brainstorming)

1. **Skill architecture:** one router skill (`kaggle-tabular`) + on-demand reference files. Matches
   the gated, sequential nature of a competition. Not decomposed into many independent skills.
2. **Packaging:** restructure this repo into a plugin + marketplace, **mirroring superpowers**
   (plugin nested under `plugins/kaggle-tabular/`, marketplace manifest at repo root). Installable via
   `claude plugin marketplace add` → `claude plugin install`.
3. **Cross-harness:** a global `install.sh` (Claude plugin install + Codex `~/.codex` pointer +
   OpenCode config pointer) **plus** the scaffolder writing a per-competition `AGENTS.md` into each
   `comp-<name>/` repo.
4. **Template scope:** ship the **full encoded toolkit** — `features.py`, upgraded `ensemble.py`,
   `finalize.py`, and model wrappers `lgbm.py`/`xgb.py`/`cat.py`/`logreg.py`.
5. **Content:** fold `refined-pipeline.md`'s concrete recipes into the references as **generalized**
   techniques; keep S6E2 numbers as illustrative evidence. `learnings/` stays as research, not shipped.

## Non-goals (YAGNI)

- No NLP/CV/audio support (tabular only, per the skill description).
- No unit tests for modeling code (correctness = CV/model performance, per repo norms).
- No auto-publishing to a public marketplace registry; local/git marketplace install only.
- No unrelated refactors of the existing `src/` keystone code (`run_experiment`, `ledger`, `cv`,
  `metric` are already correct and stay as-is).

---

## Architecture: final repo layout

```
kaggle-pipline/                          # repo root = the marketplace
  .claude-plugin/
    marketplace.json                     # lists the kaggle-tabular plugin (mirrors superpowers)
  plugins/
    kaggle-tabular/
      .claude-plugin/
        plugin.json                      # name, version, author, description, keywords
      install.sh                         # cross-harness installer (Claude / Codex / OpenCode)
      USAGE.md                           # step-by-step per harness + per-competition flow
      skills/
        kaggle-tabular/
          SKILL.md                       # the router (moved from kaggle-tabular/SKILL.md)
          references/
            hard-rules.md                # upgraded
            workflow-phases.md           # upgraded (gate enrichment)
            feature-engineering.md       # upgraded (Tier-1/Tier-2 recipes + anti-patterns)
            model-menu.md                # upgraded (linearity probe, low-depth, RealMLP)
            ensembling.md                # upgraded (rank, logit hill-climb, dedup, refit)
            orchestration.md             # kept (light enrichment)
            what-works.md                # NEW — cross-cutting patterns + "what does NOT work"
          scripts/
            scaffold_competition.py      # upgraded to also emit new src files
          assets/
            template/                    # the per-competition repo template
              ...                        # (see "Template" section)
  learnings/                             # research — NOT shipped in the plugin
  docs/superpowers/specs/                # this design doc lives here
  CLAUDE.md                             # updated to describe the new layout
```

### `marketplace.json` (repo root)
Mirrors the superpowers marketplace manifest: `name`, `owner`, and a `plugins` array with one entry
pointing at `./plugins/kaggle-tabular` (source = relative path).

### `plugin.json`
Standard plugin manifest: `name: kaggle-tabular`, `version`, `description` (reuse the skill's tuned
trigger description), `author`, `keywords` (kaggle, tabular, cross-validation, ensembling, feature
engineering, GBDT). Skills auto-discovered from `skills/`.

---

## Component 1 — The router skill (`SKILL.md`)

Keep the current router (it is already well-formed): the two foundations, the ROI order, HR-1…HR-7,
the 8-phase gated table, "what the scaffold gives you," and the on-demand reference index. Changes:

- Update the reference index to include the new `what-works.md`.
- Update the scaffolder invocation path (now under the skill's `scripts/`).
- Keep the frontmatter `description` (it is a strong trigger surface) — minor tightening only.

**Interface:** the skill is the single entry point. It reads fully, then loads exactly one reference
per phase. It depends on nothing outside its own `references/`, `scripts/`, and `assets/`.

## Component 2 — References (distilled, reusable knowledge)

Each reference is independently loadable and single-purpose. Upgrades fold in `refined-pipeline.md`:

| Reference | Upgrade |
|---|---|
| `hard-rules.md` | Add adversarial-AUC interpretation table (≈0.5 / 0.51–0.55 / >0.55 → action); add CV–LB gap tracking (reject experiments that widen CV−LB beyond the healthy band). Keep HR-1…HR-7 and the Guardian audit. |
| `workflow-phases.md` | Enrich Phase-1 gate (metric reproduces a known LB point; adversarial AUC logged) and Phase-7 gate (1.25× refit + seed ensemble; final picks by CV). |
| `feature-engineering.md` | Add **Tier-1** (in-fold TE, ALL_CATS, external-data TE, frequency) and **Tier-2 diversity generators** (quantile binning, digit extraction, n-gram categorical interactions, periodic/sinusoidal for NN only, domain composites) as generic recipes; add the anti-patterns table; keep the "diversity over individual accuracy" principle. |
| `model-menu.md` | Add "run LogReg first as a linearity probe" heuristic; low-depth GBDT (2–3) insight for near-linear signal; RealMLP as best neural member; keep TabPFN/AutoGluon guidance. |
| `ensembling.md` | Add OOF dedup (corr > 0.9999), multi-seed averaging, Optuna subset selection, rank transform, **logit-space hill climbing with negative weights**, meta-model choice table, full-data refit at 1.25× best_iteration + ~20 seeds, and the CV–LB relation as the final-decision rule. |
| `orchestration.md` | Kept; light enrichment of the role map and CV–LB contract. |
| `what-works.md` (NEW) | Cross-cutting patterns seen in 3+ top solutions + the evidence-backed "what does NOT work" table (pseudo-labeling, averaging all OOFs, 800+ poly interactions, deep trees on synthetic data, public-LB selection, …). One high-value loadable memory file. |

All feature names abstracted to generic form; S6E2 numbers cited as evidence (e.g. "XGBoost + in-fold
TE reached 0.955663 OOF — 1st place"), never as code constants.

## Component 3 — Template `src/` (rules encoded as code)

The central design idea: encode the rules so agents can't easily violate them. Extends the existing
template. **Unchanged (already correct):** `models/base.py::run_experiment` (the OOF keystone),
`ledger.py`, `cv.py`, `metric.py`.

**New / upgraded:**

- **`src/features.py` (NEW)** — leak-safe helpers designed to be called *inside* `fit_fold` (HR-1),
  each fit on train rows only then applied to val/test:
  - `OOFTargetEncoder` — smoothed target encoding fit per fold on train rows.
  - `frequency_encode`, `all_cats` (everything → category), `quantile_bins`, `digit_features`
    (units/tens), `categorical_interactions` (pairwise concat).
  - Docstrings state the HR-1 boundary explicitly.
- **`src/ensemble.py` (UPGRADE)** — add: `rank_transform` option; `hill_climb_logit` (logit space,
  **negative weights allowed**, tolerance/iteration stop); `dedup_oofs` (drop corr > 0.9999);
  `average_seeds`. Keep existing `hill_climb`, `stack`, `save_spec`, `load_oof_matrix`.
- **`src/finalize.py` (NEW)** — Phase-7 full-data refit: record avg best_iteration across folds,
  retrain on 100% data at 1.25× rounds over N seeds, average predictions.
- **Model wrappers** — `models/xgb.py`, `models/cat.py`, `models/logreg.py` alongside `lgbm.py`, each
  implementing the same `make_fit_fold(...) -> fit_fold` contract. `logreg.py` doubles as the Phase-2
  linearity probe (OHE + LogisticRegression). GPU toggles follow the existing commented-marker pattern
  that `enable_gpu()` flips.

**Interface contract (unchanged keystone):** every model is a `fit_fold(X_tr, y_tr, X_val, X_test,
fold, seed) -> (val_pred, test_pred, model)` passed to `run_experiment`, which owns the CV loop, OOF
+ preds artifacts, and the ledger row. `features.py` helpers are called *within* `fit_fold`. This is
how HR-1/HR-2/HR-4 stay enforced by construction.

## Component 4 — Scaffolder

`scaffold_competition.py` keeps its current behavior (copy template, substitute `{{COMP_NAME}}`, make
gitignored working dirs, `--gpu` uncomments markers) and additionally:
- Copies the new `features.py`, `finalize.py`, and the three new model wrappers.
- Continues to drop `AGENTS.md` into `comp-<name>/` (the per-competition cross-harness rules).
- `enable_gpu()` extended to flip the GPU markers in `xgb.py`/`cat.py` too.

## Component 5 — Cross-harness distribution

- **`install.sh`** (one markdown/skill source of truth, three targets):
  - **Claude Code:** `claude plugin marketplace add <repo>` then `claude plugin install kaggle-tabular`.
  - **Codex:** symlink `skills/kaggle-tabular` into `~/.codex/skills/` (or copy) and append a pointer
    block to `~/.codex/AGENTS.md` that references the skill's SKILL.md.
  - **OpenCode:** add an instruction/rule pointer to the OpenCode global config referencing the skill.
  - Idempotent; detects which harnesses are installed and only wires those.
- **Per-competition:** the scaffolder's `AGENTS.md` gives Codex/OpenCode local rules automatically
  when working inside a `comp-<name>/` repo (no global install strictly required for per-repo use).

## Component 6 — `USAGE.md`

The explicit step-by-step the user asked for:
1. **One-time install** per harness (via `install.sh`, with the manual fallback commands shown).
2. **Per-competition flow**, shown for each harness: `scaffold_competition.py <slug>` →
   `uv sync` → add Kaggle creds → `just download` → Phase 0→7 following the skill, with the gate that
   advances each phase.
3. How each harness *discovers* the skill (Claude: Skill tool; Codex/OpenCode: AGENTS.md + pointer).

---

## Data flow (per competition, unchanged core)

```
scaffold → template repo → Phase 1 freezes data/folds.parquet (HR-2) + proves metric (HR-3)
  → every experiment: fit_fold (HR-1 boundary) → run_experiment → oof/ + preds/ + ledger row (HR-4)
  → Guardian audits ledger rows → status kept/rejected
  → ensembler reads kept OOFs → hill-climb/stack → finalize.py full-data refit → submit by CV (HR-5)
```

## Testing / verification

Per repo norms, no unit tests for modeling code. Verification for **this build** is structural:
- `claude plugin validate <plugin path>` passes for `plugin.json` and `marketplace.json`.
- `python scripts/scaffold_competition.py demo --dest /tmp/...` produces a repo whose new `src/`
  files import cleanly (`python -c "import src.features, src.finalize, src.ensemble"`).
- `install.sh --dry-run` prints the correct per-harness actions.
- No hardcoded competition constants remain in shipped files (grep for S6E2 feature names / scores in
  `plugins/` returns only reference-doc citations, never code).

## Risks / open considerations

- **Codex/OpenCode skill discovery** differs from Claude's plugin system; the reliable mechanism is
  AGENTS.md pointers. `install.sh` must degrade gracefully if a harness's config path isn't found.
- **Git not initialized** in this repo yet — the design doc cannot be committed until `git init`.
  Marketplace install from a local path works without git; git is only needed for remote install.
- Keep skill prose and encoded template in sync (existing repo convention) — any rule change touches
  both the reference and the `src/` that encodes it.
