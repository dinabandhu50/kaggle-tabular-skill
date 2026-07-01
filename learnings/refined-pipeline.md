# Refined Kaggle Tabular Competition Pipeline

Built from three tiers of evidence (in priority order):
1. **Expert grandmaster framework** — phase gates, hard rules, OOF contract, validation-first discipline
2. **S6E2 top-10 competition writeups** — what actually won, with concrete CV/LB numbers
3. **Operational preferences** — project structure, tooling, experiment tracking

> **Competition evidence base:** S6E2 Heart Disease (630k rows × 13 features, binary, AUC metric).
> Top private LB: 0.95535. Sources: 1st–4th, 8th, 10th place writeups.

---

## The Two Foundations

1. **Trustworthy local validation.** A CV score you can trust is worth more than any model. Build and *verify* the validation harness **before** feature engineering. If CV isn't trustworthy, every downstream decision is noise.

2. **Fast experiment throughput.** The count of *high-quality* experiments is the biggest lever. Every experiment must be cheap to launch, cheap to evaluate, and self-logging.

The mechanism that enforces both: **the OOF contract**. Every experiment writes a CV score, an OOF array, and a test prediction array to disk, plus one ledger row. The CV score *is* the reward. The OOF array *is* the reusable state.

---

## ROI Order (Read This First — Most Common Mistake)

```
Trustworthy CV → diverse baselines → feature engineering → light tuning → serious ensembling
```

- **FE** is the highest ROI after CV (10x range of improvement)
- **Tuning** comes *after* FE plateaus — good defaults + early stopping get 80% of gains; over-tuning overfits CV
- **Ensembling** is NOT a 1–2% afterthought — it's where competitions are won (top solutions stack 100+ models)
- **Baseline diversity** before heavy FE — understanding the signal linearity shapes your entire FE strategy

> 8th place: "Feature engineering mattered more than hyperparameter tuning at this stage."
> Expert framework: "Do not heavily tune before FE has plateaued."

---

## Hard Rules (Non-Negotiable)

| Rule | Description |
|---|---|
| **HR-1** | The validation fold is sacred: no target-aware preprocessing (target/freq encoding, scalers, PCA, SMOTE) may be fit on data that includes the validation fold. Fit inside the fold on train rows only. |
| **HR-2** | Fix folds once (`data/folds.parquet`). Every model, experiment, and stacking level reuses the identical split. |
| **HR-3** | Implement the competition metric exactly; prove it reproduces a known LB point before trusting any CV number. |
| **HR-4** | Every experiment saves OOF + test preds + a ledger row — including failures. No artifacts = it didn't happen. |
| **HR-5** | Never tune, select features, or pick the ensemble against the public LB. Decide on CV. |
| **HR-6** | Fix and log all seeds and library versions. |
| **HR-7** | No leakage features: every feature must be computable at inference time using only information available before the prediction moment. |

---

## Phase 0 — Project Setup

**Goal:** Reproducible environment, raw data, project facts established.
**Gate:** Environment reproduces from scratch; `COMPETITION.md` exists.

### Actions
- Folder structure: `data/`, `configs/`, `src/`, `notebooks/`, `scripts/`, `submissions/`, `experiments/`, `localdev/`
- Environment: `uv` for package management; `justfile` for repeated commands
- Credentials: `.envrc` with `KAGGLE_KEY`, `WANDB_API_KEY` (never commit; gitignored)
- Download competition data via `kaggle competitions download`
- Write `COMPETITION.md`: task type, metric, train/test shape, target distribution, known LB points

### Key Infrastructure Files
```
src/
  cv.py         — fold generation + adversarial validation
  metric.py     — competition metric registry (HR-3)
  ledger.py     — append-only experiment ledger
data/
  folds.parquet — frozen folds (written once, HR-2)
```

---

## Phase 1 — Validation Harness

**Goal:** Trustworthy CV, metric proven, folds frozen.
**Gate:** Metric reproduces a known LB point; `data/folds.parquet` written; adversarial-validation AUC logged.

### Step 1: Implement the Metric Exactly (HR-3)
```python
from sklearn.metrics import roc_auc_score
# Prove: submit a baseline, verify local AUC ≈ LB score within 0.0003
```

### Step 2: Adversarial Validation — Do This Before Anything Else
```python
# Label train rows as 0, test rows as 1
# Train XGBoost to distinguish them
# Report AUC (mean ± std across folds)
```

| AUC Result | Meaning | Action |
|---|---|---|
| ≈ 0.500 | Perfect distribution match, no drift | Safe to use global statistics as features (carefully) |
| 0.510–0.550 | Minor drift | Use in-fold encoding only; investigate shifted features |
| > 0.550 | Significant drift | Feature-level analysis required; external data risky |

> All S6E2 top solutions ran adversarial validation. Result: **AUC = 0.5017 ± 0.0013** — perfect match.
> This single result enabled the key strategic decision: target statistics from the original dataset could be safely merged.

### Step 3: Fix and Freeze Folds (HR-2)
```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Save fold assignments to data/folds.parquet — never regenerate
```
- **5-fold default** — fast experiments, reasonable variance
- **10-fold** (10th place) — better OOF estimate quality, 2× training cost; use if variance is high
- **Fixed random_state everywhere** — experiments are comparable

### Step 4: Target Encoding — In-Fold vs Global
> 2nd place (Akiyoshi Kinoshita) tested both and found in-fold TE won marginally on private LB despite lower CV.

**Default rule:** Always build in-fold TE first (HR-1 compliant).
**Exception:** If adversarial AUC ≈ 0.500 exactly, global TE *may* be safe — but always test both and compare on private LB before committing.

---

## Phase 2 — Smart EDA

**Goal:** Understand signal, distributions, leakage traps. Produce 3–5 FE hypotheses.
**Gate:** `EDA_FINDINGS.md` with: shift list, leakage suspects, encoding plan, FE hypotheses.

### Actions

**1. Feature distributions and target correlation**
- Distribution, missing rate, cardinality, target mean per value for each feature
- Flag extreme target correlations (potential leakage)

**2. Run LogReg + OHE as signal linearity test**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
# OHE all features → fit LogReg → evaluate CV AUC
```
> 4th place: LogReg + OHE (449 dims) → **CV 0.95550, LB 0.95371**. This proved near-linear signal and shaped the entire strategy.

| LogReg CV | Interpretation | Strategy |
|---|---|---|
| Close to GBDT baseline | Near-linear signal | Stumps (depth 2–3), avoid heavy FE, diversity > depth |
| Much lower than GBDT | Non-linear signal dominates | Deep trees, aggressive FE, interactions |

**3. Original dataset identification (Playground Series)**
- All Playground Series competitions are synthetic — always find the source dataset
- Check Kaggle discussions for the source (for S6E2: Cleveland Heart Disease from UCI)
- Plan usage: external TE, aggregate statistics — but **never append raw rows as NN training data** (4th place finding)

**4. FE hypothesis generation** (domain-grounded, not random)
- Each hypothesis needs a mathematical or domain justification
- Write down 3–5 hypotheses before coding anything

---

## Phase 3 — Diverse Baselines

**Goal:** Map the model landscape with 4+ families. Establish the CV floor.
**Gate:** ≥ 4 model families in ledger with trustworthy CV; no impossibly-good CV.

### Recommended Model Zoo

| Model | Config | S6E2 OOF | Notes |
|---|---|---|---|
| XGBoost | `device='cuda'`, `max_depth=2–4`, early stopping | 0.9555–0.9557 | Start here; very stable |
| LightGBM | `device='cuda'`, `num_leaves=15–31`, early stopping | 0.9552–0.9556 | Fast, competitive |
| CatBoost | `iterations=5000`, native cats, Ordered boosting | 0.9556–0.9558 | Best GBDT for categoricals |
| Logistic Regression | OHE all features | **S6E2: 0.95550** | **Run this first to test signal linearity** |
| RealMLP | `pytabkit`, `n_ens=8`, metric=`1-auc_ovr` | 0.9554–0.9576 | Best neural model for tabular |
| HistGradientBoosting | sklearn defaults | 0.9550–0.9555 | Fast baseline; diversity |

### The Most Important Baseline Insight (S6E2)
> 4th place: "Run LogReg first. If competitive with GBDTs, the signal is near-linear."
> 3rd place: "Lower depths for GBDT algorithms (2, 3) worked better on this data."

When LogReg is competitive → use `max_depth=2` (stumps) as default for all GBDTs. Stumps cleanly isolate category levels without overfitting synthetic noise.

### OOF Contract — Every Baseline Must Produce
```
submissions/oof_<model>_baseline.npy     # shape (n_train,)
submissions/preds_<model>_baseline.npy  # shape (n_test,)
experiments/<model>/baseline.md         # CV score, config, notes
```

---

## Phase 4 — Feature Engineering

**Goal:** Features that reliably lift CV per model family. Build a diverse OOF library.
**Gate:** CV plateaus for the family (recent gains within fold-noise).

### FE Core Principle: Diversity Over Individual Accuracy
> 1st place: each feature set produces OOFs with *different error patterns*. The ensemble benefits from *different mistakes*, not just a better single model.

Different feature sets → different OOFs → better ensemble. A feature set that doesn't improve CV alone can still add ensemble value if it produces different errors.

---

### FE Tier 1 — Universal High ROI

#### 1. Target Encoding — In-Fold (HR-1 Compliant)
```python
# For each categorical feature, within each fold:
# compute mean target per category value on TRAIN rows only
# transform both train and validation rows with learned mapping
```
Apply to: all low-cardinality categoricals (Thallium, Chest_pain_type, Number_of_vessels_fluro, Slope_of_ST, EKG_results).

- XGBoost + TE: **0.955663 OOF** (1st place)
- Used by: 1st, 2nd, 3rd, 8th, 10th place

#### 2. ALL_CATS — Treat Everything as Categorical
```python
df_cats = df.copy()
for col in df_cats.columns:
    df_cats[col] = df_cats[col].astype(str)
# Pass to CatBoost/LGB with categorical feature specification
```
- Different split logic → diverse OOFs vs numeric encoding
- 3rd place used two versions: (1) ALL_CATS, (2) Hybrid (< 10 unique → categorical, rest numeric)
- CatBoost + ALL_CATS: ~0.955686 OOF

#### 3. Original Dataset Target Statistics (External TE)
```python
# From original source dataset (e.g., Cleveland Heart Disease):
orig_stats = orig_df.groupby(col)['target'].agg(['mean', 'std', 'count', 'median'])
df = df.merge(orig_stats, on=col, suffixes=('', f'_orig_{col}'))
```
- No fold leakage — labels from an independent dataset
- Used by: 1st, 3rd, 4th, 8th, 10th place

#### 4. Frequency Encoding
```python
for col in cat_cols:
    freq = df[col].value_counts()
    df[f'{col}_freq'] = df[col].map(freq)
```
- Captures rarity signal; complements TE
- Used by: 1st, 4th, 10th place

---

### FE Tier 2 — Diversity Generators (Ensemble Value)

#### 5. Quantile Binning
```python
df[f'{col}_qbin'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')
```
Percentile-based bins create different tree splits vs equal-width bins.

#### 6. Digit Feature Extraction
```python
df[f'{col}_units'] = df[col].astype(int) % 10
df[f'{col}_tens']  = (df[col].astype(int) // 10) % 10
```
- Apply to: Age, BP, Cholesterol, Max_HR
- Captures hidden patterns in synthetic data generation (generative model may encode digit-level structure)
- Used by: 1st, 10th place

#### 7. Bi-gram / Tri-gram Interaction Features
```python
df['Sex_Thallium'] = df['Sex'].astype(str) + '_' + df['Thallium'].astype(str)
# Then target-encode 'Sex_Thallium' inside fold
```
Selection: train LogReg on all combinations → keep top 20–32 by coefficient magnitude.
- 10th place: top 32 selected, consistently impactful for NN models
- GBDTs find these internally — add more value for linear models / NNs

#### 8. Sinusoidal / Periodic Encoding (NNs only)
```python
for col in num_cols:
    for p in [12, 14, 20]:
        df[f'{col}_sin_{p}'] = np.sin(2 * np.pi * df[col] / p)
        df[f'{col}_cos_{p}'] = np.cos(2 * np.pi * df[col] / p)
```
- Bypasses "spectral bias" — standard MLPs struggle with raw tabular numerics
- Used by: 4th, 10th place
- **GBDTs do not benefit from this** — NN only

#### 9. Categorical Duplication of Numerics (NNs / GBDTs)
```python
for col in num_cols:
    df[f'{col}_as_cat'] = df[col].astype(str)  # label-encode for trees
```
Allows models to find non-linear ordinal relationships in continuous features. Used by: 10th place.

#### 10. Domain-Specific Composite Features (Linear Models / NNs)
```python
df['cardiac_wrkload'] = df['BP'] * df['Max_HR'] / df['Age']
df['stress_severity'] = df['ST_depression'] * df['Slope_of_ST']
df['high_risk_count'] = (df['FBS_over_120']
                         + (df['BP'] > 140).astype(int)
                         + (df['Cholesterol'] > 240).astype(int))
```
> Note: GBDTs find these interactions internally — limited additional value for tree models. Useful for LogReg, MLP.

---

### FE Anti-Patterns

| Anti-Pattern | Why It Fails |
|---|---|
| 800+ polynomial interactions | Collinearity noise crashes local CV (4th place) |
| Appending original dataset as NN training rows | Confuses the deep network; domain shift (4th place) |
| Target encoding *outside* the fold as default | Falsely inflated CV; may not generalise (HR-1) |
| Blind FE without domain reasoning | Signal-to-noise degradation |
| FE before validating that CV is trustworthy | Building on sand |

### FE Experiment Discipline
1. One hypothesis at a time (3–5 features per batch)
2. Compare OOF before/after each batch; keep only what helps
3. Different features may help different model families — that is valuable for ensembling
4. Log every batch in `experiments/<model>/fe_log.md` with delta OOF and decision

---

## Phase 5 — Hyperparameter Tuning (Light)

**Goal:** Easy GBDT gains without overfitting CV.
**Gate:** Tuned CV beats best untuned by > fold-noise, else keep simpler.

### Philosophy
- Good defaults + early stopping get 80% of HPT gains
- Over-tuning **overfits CV** — the most common trap after strong FE
- Hand-tune first based on intuition, then Optuna on a constrained search space

### The Early Stopping Ceiling Pattern (Recommended)
```yaml
# Do NOT tune n_estimators — use a high ceiling with early stopping
n_estimators: 5000          # ceiling; never actually reached
early_stopping_rounds: 50   # stop when val AUC plateaus
# Tune: learning_rate, depth, regularization, subsampling
```
Without this: high n_estimators + low learning_rate = slow trials. With this: trials are fast and consistent.

### Key Parameters to Tune (GBDTs)

| Param | XGBoost | LightGBM | CatBoost | Notes |
|---|---|---|---|---|
| Depth | `max_depth` 2–6 | `num_leaves` 15–63 | `depth` 2–6 | S6E2: 2–3 often best |
| Learning rate | 0.01–0.3 | 0.01–0.3 | 0.01–0.3 | — |
| Row sampling | `subsample` 0.6–1.0 | `bagging_fraction` | `subsample` | — |
| Feature sampling | `colsample_bytree` | `feature_fraction` | `rsm` | — |
| Regularization | `reg_alpha`, `lambda` | `lambda_l1`, `l2` | `l2_leaf_reg` | — |
| Leaf size | `min_child_weight` | `min_data_in_leaf` | `min_data_in_leaf` | — |

### Optuna Search Pattern
```python
# Subsample 300k rows + 3-fold for fast trials (~24s each on GPU)
# 50–100 trials for GBDTs; diminishing returns beyond 100
# Retrain final model: full data + 5-fold + best params
```

---

## Phase 6 — Ensembling

**Goal:** Combine diverse strong models; beat best single model by > fold-noise.
**Gate:** Ensemble OOF beats best single OOF by > fold-noise; ensemble spec saved.

> Expert framework: "Ensembling is where the leaderboard is won. Winning solutions routinely stack 100+ models."
> 1st place: "Simply averaging all of them hurt performance." Selection is more important than size.

### Step 1 — Build a Diverse OOF Library

Target: 20–150 OOFs from combinations of model × feature set × seed.

| Model | Feature Set | Target OOF |
|---|---|---|
| XGBoost | BASE | ~0.9556 |
| XGBoost | BASE + TE | ~0.9557 |
| XGBoost | BASE + ALL_CATS | ~0.9556 |
| LightGBM | BASE + TE | ~0.9556 |
| LightGBM | BASE + ALL_CATS + FREQ | ~0.9556 |
| CatBoost | ALL_CATS (Plain) | ~0.9557 |
| CatBoost | ALL_CATS (Ordered) | ~0.9558 |
| RealMLP | BASE + TE | ~0.9557 |
| RealMLP | ALL_CATS | ~0.9556 |
| LogReg | OHE (all features) | ~0.9555 |
| XGB → LogReg | Leaf indices OHE | ~0.9549 |

Plus 5–20 seeds per model variant:
```python
# 5 seeds × 6 variants = 30 OOFs cheaply
# Each seed produces different predictions due to subsampling randomness
```

### Step 2 — OOF Selection and Deduplication

Before any ensemble, prune the OOF library:

**1. Remove near-duplicate OOFs**
```python
# Compute pairwise Pearson correlation between all OOFs
# Remove one from each pair with correlation > 0.9999
# (2nd place selection criterion)
```

**2. Prefer multi-seed averages**
```python
# Average OOFs across seeds before deduplication
# Single-seed OOFs carry seed-specific noise; averages are cleaner ensemble members
# (2nd place: reduced 50 → 15 models by requiring multi-seed averages)
```

**3. Optuna-based subset search**
```python
# Each trial: propose a subset of OOFs
# Objective: maximize OOF AUC of Ridge-combined subset
# 500–2500 trials; expect ~10% of OOFs consistently selected
```

**4. Forward Selection + Backward Elimination (optional)**
- After Optuna, run greedy forward selection then backward elimination
- More stable than pure Optuna at small counts (2nd place used this for final 4 → 15 → 4 selection)

### Step 3 — Meta-Model Choice

| Meta-Model | When to Use | Notes |
|---|---|---|
| **Ridge** | Default; 10+ selected OOFs | Simple, stable, handles correlation well |
| **Logistic Regression** | Binary classification | Similar to Ridge; add `C` tuning |
| **NN stacking** | ≤ 6 selected OOFs after aggressive deduplication | Small input → overfitting risk low (2nd place) |
| **LR stacking** | Fast alternative to NN | Less expressive but more stable |

> 1st place used Ridge. 2nd place used NN after aggressive selection (6 inputs). 8th place used NN with exponential decay + early stopping (patience=7) for the jump from 0.95538 to 0.95569.

### Step 4 — Rank Transformation (Recommended)
```python
from scipy.stats import rankdata

# Before ensembling, optionally rank-transform OOFs
oof_ranked = rankdata(oof_proba) / len(oof_proba)  # normalize to [0, 1]
```
- Removes calibration differences between model families (GBDTs vs NNs produce different probability scales)
- 2nd place: added rank-transformed versions of 2 models as additional ensemble members
- 3rd place: converted all OOFs to ranks before hill climbing
- 4th place: rank ensembling as the final step — most robust to public-private shake-up

### Step 5 — Hill Climbing in Logit Space (Advanced)
```python
from scipy.special import logit, expit
import numpy as np

# Convert to logit space before blending
oof_logits = logit(np.clip(oofs, 1e-7, 1 - 1e-7))

# Greedy hill climbing: add models one at a time, accept if ensemble AUC improves
# Support NEGATIVE weights — can subtract a harmful model
# Stopping: tolerance 1e-7 OR max 1000 iterations
# Final output: expit(weighted_sum_of_logits)
```
Why logit space: avoids boundary saturation effects near 0 or 1 (10th place finding).
Why negative weights: more expressive; effectively subtracts correlated noise from a model (10th place).

### Anti-Patterns in Ensembling

| Anti-Pattern | Why It Fails |
|---|---|
| Simple average of all OOFs | Dilutes signal with redundant predictions (1st place) |
| Giving dominant model 65%+ weight | Overfits OOF; collapses on LB (4th place) |
| Nonlinear stacking without aggressive OOF selection | Overfits on stacked OOFs (1st place) |
| Hill climbing without CV guard | CV climbs but LB doesn't follow (2nd place cautionary) |
| Selecting ensemble based on public LB | HR-5; misleading (private LB differs) |

---

## Phase 7 — Final Submission

**Goal:** Squeeze + lock final predictions. Choose submissions by CV, not LB.
**Gate:** Seed ensemble + full-data refit done; final 2 submissions chosen by CV–LB relation.

### Full-Data Retraining (GBDT Best Practice)
```python
# Step 1: Record avg best_iteration across CV folds
avg_best_iter = np.mean([model_fold_k.best_iteration for k in range(n_folds)])

# Step 2: Retrain on FULL train data with:
n_estimators = int(avg_best_iter * 1.25)  # 25% more — full data needs more rounds

# Step 3: Average predictions over 10–20 random seeds
final_preds = np.mean([
    train_and_predict(seed=k, n_estimators=n_estimators)
    for k in range(20)
], axis=0)
```
> 1st place: full-data retrain at 1.25× best_iteration + 20 seeds outperformed averaging K fold models.
> Why 1.25×: more training data → each example seen proportionally fewer times → can afford slightly more iterations.

### The CV–LB Relation — The Most Important Final Decision

> "Trust the CV–LB relation, not the best CV in isolation." — 1st place (Masaya Kawamata)

**The pattern:**
- Early experiments: CV improvements translate reliably to LB improvements ✅
- Later experiments: CV keeps improving but LB stops following — **split overfitting**
- Choosing the highest-CV submission risks private LB collapse

**How to monitor it:**
1. Submit intermediate ensembles throughout development (not just at the end)
2. Plot CV vs Public LB for every submitted model
3. Find the breakpoint where CV→LB slope flattens
4. Choose final submission from the range where slope was still positive

**OOF-to-LB Gap Tracking (4th place method)**
```
Healthy: CV - Public LB ≈ 0.00185  →  experiment accepted
Danger:  CV - Public LB > 0.00190  →  experiment rejected (likely overfitting noise)
```
Track this gap for every submitted model. Widen it = reject.

**S6E2 example of split overfitting:**
| Submission | CV | Public LB | Private LB |
|---|---|---|---|
| 1st place (chosen) | 0.955780 | 0.95396 | **0.95535** |
| 1st place (best CV, not chosen) | 0.955865 | 0.955393 | 0.95534 |

The higher-CV submission had *lower* private LB. The chosen submission was the right call.

### Final Submission Checklist
- [ ] Submit intermediate ensembles early; track CV–LB slope
- [ ] Full-data retrain at 1.25× best_iteration, 20 seeds
- [ ] Two final submissions: (1) highest-trustworthy-CV, (2) conservative fallback
- [ ] Never chose based on public LB alone (HR-5)

---

## Cross-Cutting Patterns (Seen in 3+ Top Solutions)

| Pattern | Solutions | Description |
|---|---|---|
| Adversarial validation as first step | 1st, 2nd, 4th | AUC ≈ 0.5 → no drift; shapes TE strategy |
| Trust CV strictly | All | Never deviate to chase public LB |
| Target encoding (in-fold) | 1st, 2nd, 3rd, 8th, 10th | Single most impactful FE |
| ALL_CATS feature set | 1st, 3rd, 4th | All features as categorical → diversity |
| Low GBDT depth (2–3) | 3rd, 4th, 8th, 10th | Near-linear signal; stumps generalize better |
| RealMLP as best single neural model | 1st, 2nd, 3rd, 4th, 8th | `n_ens=8–20`, metric `1-auc_ovr` |
| Original dataset statistics | 1st, 3rd, 4th, 8th, 10th | External TE; no leakage |
| Rank transformation before ensemble | 2nd, 3rd, 4th | Removes calibration differences |
| Multi-seed averaging | 1st, 2nd, 4th | Stabilizes predictions cheaply |
| OOF correlation deduplication | 1st, 2nd | Remove OOFs with corr > 0.9999 before stacking |
| Hill climbing (greedy) for ensemble weights | 3rd, 4th, 10th | Beats uniform averaging |
| Logit-space blending | 10th | Avoids boundary saturation |
| Gap tracking (CV - LB) | 4th | Hard reject experiments that widen gap |

---

## What Does NOT Work (S6E2 Evidence)

| Technique | Finding | Source |
|---|---|---|
| Pseudo-labeling | Did not improve CV | 1st place |
| Knowledge distillation (soft labels) | Did not improve CV | 1st place |
| Averaging all OOFs | Dilutes signal, hurts ensemble | 1st place |
| Nonlinear stacking without selection | Overfits stacked OOFs | 1st place |
| 800+ polynomial interaction features | Collinearity noise crashed CV | 4th place |
| Appending original data as NN training rows | Confused the deep network | 4th place |
| Unconstrained ensemble weight optimizer | Gave 65% weight to CatBoost → LB collapse | 4th place |
| Custom NN architectures as base models | GBDTs dominated; NNs consistently weaker | 8th place |
| LR stacking | Minimal improvement vs NN stacking | 8th place |
| Standard autoencoders (non-VAE) | Latent features didn't help | 1st place |
| Deep trees (depth > 4) on synthetic data | Overfits noise; depth 2–3 is better | 3rd, 4th place |
| Public LB selection | Misleading; private LB can differ significantly | Multiple |

---

## Experiment Tracking Structure

```
experiments/
  xgb/
    baseline.md      # config, OOF score, notes
    fe_v1.md         # features added, delta OOF, keep/discard decision
    hpt_v1.md        # optuna results, best params, trial count
  lgb/
    ...
  catboost/
    ...
  ensemble/
    oof_library.md   # all OOFs: name, CV, correlation matrix summary
    ensemble_v1.md   # which OOFs selected, meta-model, CV, result
```

W&B tracks: fold-level AUC, OOF AUC, hyperparameters per run.
Markdown files track: reasoning, hypothesis, what worked and why.

---

## Scores Reference Map (S6E2)

| Approach | OOF / CV | Private LB |
|---|---|---|
| XGBoost baseline (no FE) | 0.95522–0.95535 | — |
| LogReg + OHE | 0.95550 | — |
| XGBoost + TE | 0.95563–0.95566 | — |
| XGBoost + ALL_CATS | 0.95562 | — |
| CatBoost + ALL_CATS (Ordered) | 0.95569–0.95575 | — |
| RealMLP (best config) | 0.95574–0.95576 | — |
| AutoGluon | 0.95575 | — |
| 3rd place ensemble | 0.955803 | 0.95535 |
| 2nd place ensemble | 0.955759 | **0.95535** |
| 1st place (chosen) | 0.955780 | **0.95535** |
| 1st place (best CV, NOT chosen) | 0.955865 | 0.95534 |
| Our current best (lgb_hp_v1) | 0.95542 | — |

Gap to close: **~0.0003 OOF**. Target encoding alone closes most of it.

---

## Operational Checklist

### Before Every Experiment
- [ ] Folds loaded from `data/folds.parquet` (HR-2)
- [ ] All target-aware preprocessing inside the fold loop (HR-1)
- [ ] Seeds fixed and logged (HR-6)

### After Every Experiment
- [ ] OOF saved to `submissions/oof_<name>.npy`
- [ ] Test preds saved to `submissions/preds_<name>.npy`
- [ ] Ledger row appended (CV score, config hash, timestamp)
- [ ] Markdown note written in `experiments/<model>/`

### Before Final Submission
- [ ] CV–LB relation tracked across all submitted models
- [ ] Full-data retrain at 1.25× best_iteration, 20 seeds
- [ ] OOF-to-LB gap within healthy range (~0.00185 for S6E2)
- [ ] Two submissions: best-trustworthy-CV + conservative fallback
- [ ] Did not select based on public LB alone (HR-5)
