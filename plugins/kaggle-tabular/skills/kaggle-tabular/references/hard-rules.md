# Hard Rules, Leakage & the Validation Guardian

Read this at Phase 1 and **before accepting any experiment as "kept"**. Leakage produces CV that
looks great and collapses on the private leaderboard — it is the single most common way solo
competitors lose. These rules are the ML analogue of "never let the same agent own tests and
implementation": they exist to stop the reward (CV score) from being hacked.

## The seven Hard Rules (enforce via hooks / CI / the Guardian audit)

- **HR-1 — The validation fold is sacred.** No preprocessing that uses the target or cross-row
  statistics may be fit on data that includes the validation fold. This includes: target encoding,
  frequency/count encoding, mean/median imputation, standard/robust scaling, TF-IDF or count
  vectorizers, PCA / SVD, feature selection by target correlation or importance, SMOTE / resampling,
  and any "fit on train+test" trick. All of these are fit **inside each fold on training rows only**,
  then applied to the held-out fold and to test. Fit-on-full-data is the #1 cause of inflated CV.
- **HR-2 — Fix the folds once, reuse everywhere.** Generate one `fold` assignment per training row at
  Phase 1 and persist it (`data/folds.parquet`). Every model, experiment, and stacking level uses
  the identical split. Stacking across mismatched folds leaks the target.
- **HR-3 — The metric must match the competition exactly.** Implement it locally and assert it
  reproduces a known public-LB point (e.g. an all-mean or all-zero submission) before trusting any CV
  number. A CV computed with the wrong metric optimizes the wrong thing.
- **HR-4 — Every experiment saves OOF + test preds + a ledger row** (including failures). An
  experiment that didn't save artifacts didn't happen and cannot be ensembled or audited.
- **HR-5 — Never tune, select features, or pick the ensemble against the public LB.** The public LB
  is a small held-out probe with its own noise and a limited submission budget. All decisions are
  made on CV. The LB is used only to (a) confirm HR-3 once and (b) monitor the CV↔LB relationship.
- **HR-6 — Determinism.** Fix and log all seeds and library versions. A CV number that can't be
  reproduced is not evidence.
- **HR-7 — No leakage features.** Before accepting a feature, confirm it is computable at inference
  time using only information available *before* the prediction moment: no future information, no
  test-target echoes, no row-identity / index leakage, no aggregates that span the validation rows.

## The Validation Guardian audit (run before marking any experiment `kept`)

The Guardian is an adversarial role (mirrors your pre-PR critic). For experiment `<exp_id>`, verify:

1. **Folds match.** The experiment used `data/folds.parquet` unchanged (HR-2). Reject if it
   regenerated or reshuffled folds.
2. **No validation-fold contamination.** Every target-aware / cross-row transform was fit per-fold on
   training rows only (HR-1). Inspect the feature code, not just the result.
3. **Metric correctness.** The CV score uses the proven competition metric (HR-3).
4. **Inference-time availability.** Every feature passes HR-7. Flag suspiciously predictive columns
   (IDs, timestamps, anything with near-perfect single-feature CV).
5. **Reproducibility.** Seeds and versions are logged; a re-run reproduces the CV (HR-6).
6. **Artifacts present.** `oof/` and `preds/` arrays exist and are row-aligned; ledger row written
   (HR-4).
7. **Sanity vs baselines.** CV is not "too good to be true" relative to the Phase-3 baseline spread.
   A sudden large jump is a leakage suspect until proven otherwise.

Outcome: mark the ledger row `kept` or `rejected:<reason>`. Rejected experiments stay in the ledger
(they're evidence) but are excluded from ensembling.

## Adversarial validation (standard diagnostic)

Run at Phase 1, and re-run whenever CV and LB disagree. It answers: *can I trust random K-fold?*

1. Label train rows `0`, test rows `1`; drop the real target.
2. Train a classifier (LightGBM) under CV to distinguish train from test; report **ROC-AUC**.
3. Interpret:
   - **AUC ≈ 0.5** → train and test are drawn alike; random K-fold is trustworthy.
   - **AUC ≫ 0.5** → distribution shift. Inspect the classifier's feature importances to find the
     *shifted* features, then respond: drop/transform those features; switch to time/group folds;
     weight training rows by their predicted "test-ness"; or select a validation subset of the most
     test-like training rows. Expect a CV↔LB offset and lean even harder on CV over LB.

`src/cv.py` ships an `adversarial_validation()` implementation.

## CV scheme selection (Phase 1)

Choose by **test structure**, not habit:

- i.i.d. rows → `StratifiedKFold` (classification) / `KFold` (regression).
- Grouped entities (user / patient / store / session) → **`GroupKFold`** so an entity never spans
  folds. Entity leakage is brutal and invisible to random CV.
- Temporal → **`TimeSeriesSplit`** or purged/embargoed time folds; never shuffle across time.
- Rare target / small data → `StratifiedKFold` + repeated CV for a more stable estimate.

Default `n_folds = 5`. Use more for small/noisy data (better estimate), fewer for very large data
(preserve throughput).

## Anti-pattern checklist (the Guardian rejects these)

- Fitting target encoding / scaler / TF-IDF / imputer / feature-selector on full data (HR-1).
- Different fold splits across models or stacking levels (HR-2).
- A local metric that doesn't match the competition's, or was never checked against the LB (HR-3).
- Tuning or feature selection driven by public-LB movement (HR-5).
- A single baseline → straight to feature engineering (skips landscape mapping; hides leakage).
- "CV too good to be true" accepted without a leakage audit.
- Ensembling correlated near-duplicate models and expecting gains (diversity is the point).
- Multi-level stacks where a level doesn't beat the prior level on OOF.
- Experiments not saved to the ledger.
- Heavy GBDT tuning before feature engineering has plateaued (wrong ROI order).
