# Workflow Phases (detailed)

Each phase: **Goal → Actions → Gate (must pass to advance) → Owner agent → Save.** Do not advance
until the gate is met and the result is logged. Bold phases (1, 4, 6) are where competitions are won.

---

## Phase 0 — Setup & ingestion
- **Goal:** reproducible env + raw data on disk + competition facts captured.
- **Actions:** create the `uv` env; download data via the Kaggle API into `data/raw/`; write
  `COMPETITION.md` capturing task type, the **exact** metric, row/column counts, target format,
  group/time structure, submission format, daily submission limit, and timeline. Ask the user for
  the competition's Overview, Discussion, and Code URLs and record them in `COMPETITION.md` —
  see `orchestration.md` → "External intel gathering" for how these get mined periodically.
- **Gate:** `just setup` reproduces the env from scratch; raw-data hashes recorded; `COMPETITION.md`
  states the **CV decision** (which fold scheme, and why) as a hypothesis to verify in Phase 1; the
  three external links are recorded (or explicitly noted as unavailable).
- **Owner:** Setup agent.
- **Save:** `COMPETITION.md`, env lockfile, data hashes, one line in `PROGRESS.md`.

## Phase 1 — Build the validation harness FIRST
- **Goal:** a CV scheme you can trust, the metric implemented, folds frozen.
- **Actions:**
  1. Implement the competition metric in `src/metric.py` and **prove HR-3**: a trivial submission
     (all-mean / all-majority) must reproduce its public-LB score within tolerance.
  2. Choose the fold scheme by test structure (see `hard-rules.md` → "CV scheme selection").
  3. Run **adversarial validation** (`src/cv.py`); log the AUC and any shifted features.
  4. Freeze folds → `data/folds.parquet` (HR-2). Decide `n_folds`.
  5. Record the adversarial-AUC band (see `hard-rules.md` → interpretation table) in `COMPETITION.md`; it drives the encoding strategy (global vs strictly in-fold).
- **Gate:** metric reproduces LB to tolerance; folds persisted; adversarial-validation AUC logged;
  `COMPETITION.md` CV decision upgraded from hypothesis to confirmed.
- **Owner:** Validation Guardian (the role that polices HR-1/HR-2/HR-7 for the rest of the run).
- **Save:** `src/metric.py`, `data/folds.parquet`, adversarial-validation result, one line in
  `PROGRESS.md`.

## Phase 2 — Smart EDA (beyond table stakes)
- **Goal:** understand signal, shift, and traps before modeling.
- **Actions:** standard checks (missingness, ranges, correlations, target balance) **plus** the two
  highest-value, most-skipped checks:
  - **Train vs test distribution per feature** — flag shifted features for the Guardian.
  - **Target over time** — trend / seasonality, if any time column exists.
  - Categorical cardinalities (drives encoding choice); leakage suspects (IDs, timestamps, columns
    that are implausibly predictive alone).
- **Gate:** `EDA_FINDINGS.md` listing shifted features, leakage suspects, an encoding plan per
  categorical, and 3–5 concrete FE hypotheses for Phase 4.
- **Owner:** EDA agent (writes and runs its own exploration code; reports back).
- **Save:** `EDA_FINDINGS.md`, key plots, one line in `PROGRESS.md`.

## Phase 3 — Diverse baselines (in parallel, no FE yet)
- **Goal:** map the model landscape; set the bar; expose leakage via sanity.
- **Actions:** train a *spread* of model families on raw / minimally-processed data, each through
  `run_experiment(...)` so each emits OOF + preds + a ledger row under the frozen CV. See
  `model-menu.md` for choices. At minimum: LightGBM, XGBoost, CatBoost (raw categoricals), plus a
  linear model. Add a small MLP and — if rows ≤ ~10k, features ≤ ~500 — TabPFN-2.5 and/or AutoGluon
  for diversity.
- **Gate:** ≥4 baseline families in the ledger with trustworthy CV; the family leaderboard written;
  no family shows impossibly-good CV (leakage check by the Guardian).
- **Owner:** one baseline agent per family, **in parallel** (embarrassingly parallel, cheap).
- **Save:** ledger rows, `experiments/baselines.md`, one line per family in `PROGRESS.md`.

## Phase 4 — Feature engineering loop (the main ROI engine)
- **Goal:** find features that *reliably* lift CV; keep the survivors **per model family**.
- **Method:** hypothesis-driven first (a reason per feature), then scale where cheap. Run features in
  **groups**; accept a group only if it improves CV beyond fold-noise (compare ΔCV to the family's
  `cv_std`). Full recipes and leak-free encoding in `feature-engineering.md`.
- **Discipline:** plan first — write `experiments/<model>/specs/NNN_<slug>.md` (hypothesis, expected
  ΔCV) before implementing. Then implement in `src/feature_engineering/<model>/NNN_<slug>.py` (never
  inline `python -c`/heredoc code — not auditable, not reproducible), with tunable knobs in
  `configs/features/<model>/NNN_<slug>.yaml`. Run it via `run_experiment` for a new OOF + preds +
  ledger row, and log the outcome in `experiments/<model>/NOTES.md` (ΔCV vs fold-noise, kept/dropped,
  why). Full convention in `feature-engineering.md` → "Artifact discipline". Keep **per-family**
  surviving feature sets — they differ by model, and the diversity feeds Phase 6.
- **Gate:** CV plateaus for a family (recent groups within fold-noise) → stop FE for that family.
- **Owner:** parallel FE-explorer agents, **one per model family**, to keep feature sets diverse.
- **Save:** ledger rows, `experiments/<model>/NOTES.md`, cached feature frames, one line per kept/
  dropped idea in `PROGRESS.md`.

## Phase 5 — Model-specific tuning (deliberately light)
- **Goal:** capture easy tuning gains without overfitting CV or burning the budget.
- **Reality:** for GBDTs, sensible defaults + **early stopping** capture most of the gain; tuning is
  lower ROI than FE and over-tuning overfits the validation set.
- **Method:** (1) hand-tune a few high-leverage params by intuition; (2) then a *small* Optuna search
  via `src/optimize/optuna_search.py::tune(...)` — a reusable, config-driven harness (search space in
  `configs/optimize/<model>.yaml`) that runs trials on the frozen folds without writing per-trial
  ledger/OOF artifacts (30 throwaway trials shouldn't cost 30 ledger rows); only the winning config
  is then logged via `run_experiment`. Params worth searching:
  - LightGBM/XGBoost: low `learning_rate` + more rounds + early stopping; `num_leaves`/`max_depth`;
    `min_child_samples`/`min_child_weight`; `feature_fraction`/`colsample_bytree`;
    `bagging_fraction`/`subsample`; L1/L2 (`reg_alpha`/`reg_lambda`).
  - CatBoost: `depth`, `learning_rate`, `l2_leaf_reg`, native categorical handling.
- **Gate:** tuned CV beats the family's best untuned CV by > fold-noise; else keep the simpler model.
- **Owner:** tuning agent per family (low-volume, high-judgment — a paid-tier fit).
- **Save:** ledger row with the frozen tuned config, one line in `PROGRESS.md`.

## Phase 6 — Ensembling (where the competition is actually won)
- **Goal:** combine diverse strong models into something better than any single one.
- **Precondition:** several *diverse* strong single models exist, each with OOF + preds on the
  identical folds. Diversity (different families, different feature sets) matters more than each
  member's raw strength. Full method in `ensembling.md`.
- **Escalation:** hill climbing over OOF → stacking (Level-2 meta-model on the OOF matrix, same
  folds) → multi-level only if each level beats the prior on OOF → optional distillation into one
  model.
- **Overfitting guard:** the meta-model can overfit the OOF; keep it simple, regularize, prefer
  hill-climbing weights when stacking gains are within fold-noise, watch the CV↔LB relationship.
- **Gate:** ensemble OOF beats best single-model OOF by > fold-noise; ensemble spec (members +
  weights / meta-model) saved as a reproducible artifact.
- **Owner:** ensembler agent (reads the ledger; owns `src/ensemble.py`).
- **Save:** ensemble spec, ledger row, one line in `PROGRESS.md`.

## Phase 7 — Final-mile strengthening
- **Goal:** squeeze the last reliable gains and lock submissions.
- **Actions (each low-risk, additive):** seed-ensembling (retrain over many seeds, average);
  retrain on 100% of data at ~1.25× the average best_iteration across folds, averaged over ~20 seeds (see ensembling.md → full-data refit); optional pseudo-labeling per
  `ensembling.md` (respecting HR-1 across folds).
- **Gate:** final two submissions chosen **by CV** (see `orchestration.md` → CV–LB contract);
  reproducible end-to-end via `just submit`.
- **Owner:** ensembler + Validation Guardian jointly.
- **Save:** final submission files, the exact specs that produced them, and a `PROGRESS.md` line per
  submission (CV, public LB once checked, which hedge it represents).
