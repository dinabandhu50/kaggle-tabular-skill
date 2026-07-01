# 1st Place Solution Analysis — S6E2 Predicting Heart Disease

Source: [Kaggle Writeup](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t)  
Author: anonymous (196 votes, 52 comments)

---

## Final Scores

| Submission | CV | Public LB | Private LB |
|---|---|---|---|
| Final chosen | 0.9557801 | 0.95396 | **0.95535** |
| Best CV obtained | 0.955865 | 0.955393 | 0.95534 |

Key insight: **the highest-CV submission was NOT chosen as the final submission.**  
Reason: suspected split overfitting — the CV–LB relation had broken down beyond CV ~0.95578.

---

## Our Scores vs 1st Place

| | Our best (lgb_hp_v1) | 1st Place |
|---|---|---|
| OOF AUC | 0.95542 | 0.95578 |
| Public LB | TBD | 0.95396 |
| Private LB | TBD | 0.95535 |
| Gap | **~0.0004 OOF** | — |

The gap is small but the approach to close it is clear from the writeup.

---

## Overall Strategy

> "Create many slightly different models → select effective combinations → combine with a simple linear model."

The central idea is **diversity**, not dominance. Instead of finding one magic model:

1. Generate ~150 OOF predictions from multiple feature sets × multiple model types
2. Use Optuna (2500 trials) to search for the best *subset* of OOFs
3. Combine selected OOFs with **Ridge regression** (not nonlinear stacking)

Only ~1/10 of generated OOFs were consistently selected — most were redundant.

---

## Feature Engineering — Key Techniques

### What They Did (that we haven't)

| Technique | Description | Why It Helps |
|---|---|---|
| **ALL_CATS** | Convert all features to string → treat as categorical | CatBoost/LGB handle categoricals differently, creating diverse splits |
| **Target Encoding (TE)** | Mean target per category value | Directly encodes discriminative power; XGBoost BASE+TE scored 0.955663 |
| **Frequency Encoding** | Count of each value in train | Rare values carry signal; complements TE |
| **Quantile binning (qcut)** | Bin numerics by percentile | Different from equal-width; tree splits change |
| **Digit features** | Extract units/tens/hundreds digit | Captures hidden structure in synthetic data |
| **Rounding** | Age/5, BP/10, etc. | Coarser groupings → different model behaviour |
| **Genetic Programming (gplearn)** | Auto-generate nonlinear interactions | Not for individual model boost — for OOF diversity |
| **Original dataset statistics** | Merge stats from the source Cleveland dataset | Target mean, WoE, Entropy — external target encoding |
| **DVAE (Denoising VAE)** | Learn compressed latent representations | Nonlinear representations for diversity, not accuracy |

### What We Tried (and the limits)

| Technique | Our result | Reason it didn't help much |
|---|---|---|
| Manual interaction features | +0 OOF | XGBoost already finds these internally |
| Polynomial features | +0 OOF | Same; trees handle nonlinearity |
| Risk score composite | Captured by model | Useful for linear models, redundant for GBDT |
| Feature pruning | Marginal | GBDT is robust to extra features |

### Key Insight on FE
> **FE for diversity, not just accuracy.** Each feature set produces different OOF predictions. The ensemble benefits from *different mistakes*, not just better individual predictions.

---

## Model Zoo

Models trained by the 1st place solution:

| Model | Notes |
|---|---|
| XGBoost | Multiple feature sets |
| LightGBM | Multiple feature sets |
| CatBoost | Native categorical handling |
| **RealMLP** | Neural net for tabular; strong with target encoding |
| **RGF** (Regularized Greedy Forest) | Lower standalone CV but high ensemble contribution |
| **TabICL** | In-context learning for tabular; subsample 100k×5 |
| **AutoGluon** | Best single model: 0.955747 OOF |

### Key Insight on Model Selection
> A model with **lower standalone CV can still contribute to ensemble** if it makes *different* errors than other models. RGF (0.954980 CV) was selected frequently despite being weaker than XGBoost (0.955575 CV).

---

## Single Model Performance Table

| Feature Set | Model | CV OOF |
|---|---|---|
| BASE+BIN+DIGIT+ALL_CATS | AutoGluon | 0.955747 |
| BASE+BIN+DIGIT+ALL_CATS | RealMLP | 0.955739 |
| ORIG+TE+EMB | RealMLP | 0.955726 |
| BASE+GP_FEAT+ALL_CATS | RealMLP | 0.955720 |
| BASE+BIN+DIGIT+ALL_CATS | CatBoost | 0.955686 |
| **BASE+TE** | **XGBoost** | **0.955663** |
| BASE+BIN+DIGIT+ALL_CATS+FREQ | LGBM | 0.955652 |
| BASE+ALL_CATS | XGBoost | 0.955619 |
| BASE | XGBoost | 0.955575 |
| DVAE+ALL_CATS | XGBoost | 0.955426 |
| BASE+BIN+DIGIT+ALL_CATS | RGF | 0.954980 |
| BASE (subsample 100k×5) | TabICL | 0.954971 |

**Our current best: 0.95542.** The gap to the top single model is ~0.0003.  
Adding TE to XGBoost/LGB alone could close most of that gap.

---

## Ensemble: OOF Selection + Ridge

### Process
1. Generate ~150 OOF prediction arrays (one per model × feature set combination)
2. Run **Optuna with 2500 trials**: each trial proposes a subset of OOFs
3. Objective: maximize OOF AUC of the combined subset
4. ~15 OOFs consistently selected (~10% of total)
5. Final combination: **Ridge regression** on selected OOFs

### Why Ridge (not nonlinear stacking)?
- Simple and stable
- Handles correlated predictions well (OOFs from similar models are highly correlated)
- Nonlinear meta-models (neural nets, GBDTs on OOFs) tend to overfit
- More complexity ≠ better generalization

### Why NOT simple averaging?
> "Simply averaging all of them hurt performance."  
Dilutes the signal with redundant/noisy predictions. Selection is more important than size.

---

## Full-Data Retraining Strategy

For final test predictions on GBDT/RGF:
1. Retrain on **full train+validation data** (not just fold models)
2. Use **20 different random seeds** and average predictions
3. Set `n_estimators` = **1.25 × average best iteration from CV**

This outperformed simply averaging the K fold models.

**Why 1.25×?**  
Training on more data means the model hasn't seen as many examples per tree — it can afford slightly more iterations without overfitting.

---

## The Most Important Lesson: Trust the CV–LB Relation

### What happened
- Best CV obtained: **0.955865**
- But this was **NOT** chosen as final submission

### Why not?
After running many experiments and submitting intermediate ensembles, the author noticed:
- Up to CV ~0.95578, CV improvements translated reliably to LB improvements
- Beyond CV ~0.95578, the CV–LB relation **broke down**
- Higher CV no longer meant higher LB — it was **split overfitting**

### The principle
> **"Trust the CV–LB relation, not the best CV in isolation."**

- Monitor the *slope* of the CV→LB relationship across submissions
- When improvements in CV stop translating to improvements in LB, you have found the ceiling of genuine generalization
- Choose final submission from the last range where CV–LB is still consistent

### Split Overfitting
When you run hundreds of Optuna trials selecting subsets of 150 OOFs, you are effectively hill-climbing on the CV metric. Eventually you start exploiting fold-specific noise rather than finding genuine signal.

---

## What Did NOT Work

| Method | Notes |
|---|---|
| Pseudo labeling (soft + hard) | Did not improve CV |
| Knowledge distillation | Did not improve CV |
| Very deep GBDT models | Overfitting, no gain |
| High-order interaction expansion | Too much noise |
| Autoencoders (other than DVAE) | Standard AE latents didn't help |
| Nonlinear stacking | Overfits on stacked OOFs |
| Averaging all 150 OOFs | Dilutes signal |
| Public LB chasing | Misleading; private LB different |

---

## CV Design and Leakage — Advanced Notes

### The standard approach (what we all do)
- 5-fold StratifiedKFold, `shuffle=True`, `random_state=42`
- Early stopping on validation fold
- OOF predictions collected from held-out fold

### The subtle leakage from early stopping
Even in standard K-fold + early stopping:
- The validation fold **indirectly selects** the model iteration (best round)
- So the OOF prediction for fold A was produced by a model *selected* using fold A
- This is widely accepted in Kaggle but **not perfectly leakage-free**

### The correct solution: Nested K-Fold
```
Outer fold (k): generates truly out-of-fold predictions
  Inner fold (k-1): tunes hyperparameters / determines early stopping
    → retrain on full outer-training data with determined params
    → predict on outer-validation fold
```
**Cost: k × (k-1) model fits per OOF.** With 150 OOFs, this is prohibitively expensive.

### Practical safeguards instead
- Fixed CV splits (same random_state everywhere)
- Avoid overly flexible meta-models (use Ridge)
- Limit ensemble size
- Check CV–LB consistency with actual submissions
- Don't blindly chase public LB

---

## Key Numbers to Remember

| Metric | Value |
|---|---|
| 1st place private LB | 0.95535 |
| Best single model CV (AutoGluon) | 0.955747 |
| XGBoost BASE+TE CV | 0.955663 |
| XGBoost BASE (no FE) CV | 0.955575 |
| Our best OOF (lgb_hp_v1) | 0.95542 |
| Gap to 1st place (single model) | ~0.0001 |
| Gap to 1st place (ensemble) | ~0.0002–0.0003 |

The gap is genuinely small. **Target encoding + model diversity + Ridge ensemble** is the path to close it.

