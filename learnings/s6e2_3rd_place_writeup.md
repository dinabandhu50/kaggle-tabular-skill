# 3rd Place Solution

**Author:** Mert Bayraktar  
**Source:** https://www.kaggle.com/competitions/playground-series-s6e2/writeups/3rd-place-solution  
**Votes:** 14 | **Competition:** Playground Series S6E2 — Heart Disease Prediction

---

## Final Score

| Metric | Value |
|---|---|
| **CV (OOF AUC)** | **0.955803** |
| Private LB | 0.95535 (3rd place) |

---

## Strategy Overview

> "My strategy focused on building models with different feature sets and objectives, checking the correlations of the predictions, and strictly trusting the CV score."

Three pillars:
1. **Diversity** — multiple model families with multiple feature sets
2. **Correlation checking** — ensure OOFs are not near-duplicates before blending
3. **Trust CV** — never deviate from local CV in favour of public LB

---

## Feature Engineering

The only feature engineering technique that worked:

### Original Dataset Target Encoding
- Sourced the original Cleveland Heart Disease dataset (not synthetic)
- Computed mean of the target from original data **per categorical feature value**
- Merged these statistics into competition training folds as new columns
- This is **external target encoding** — no fold leakage since the labels come from an independent dataset

**Why only this FE worked:**
- The signal in this competition is near-linear and mostly captured by the raw features
- Additional engineered features tend to add noise rather than signal for tree models
- The original dataset provides clean, small-sample signal that complements the large synthetic dataset

---

## Models

Final ensemble was a blend of Gradient Boosting, Neural Networks, and Linear Models.

### Two Feature Set Versions Used Across All Models

| Version | Description |
|---|---|
| **Version 1 — ALL_CATS** | All features treated as categorical (string type) |
| **Version 2 — Hybrid** | Features with < 10 unique values → categorical; rest treated as continuous |

### Model Results

| Model | CV | Notes |
|---|---|---|
| **RealMLP** | **0.95576** | Best single model; `n_cv=2`, metric: `1-auc_ovr` |
| **CatBoost Ordered** | **0.95575** | 2nd best; Ordered boosting type more effective than Plain |
| XGBoost (depth=3) | — | Standard GBDT with low depth |
| XGBoost + Pseudo-labeling | — | Modest additional diversity |
| XGBoost + LR Residuals | — | XGBoost trained on Logistic Regression residuals |
| HistGradientBoosting | — | sklearn; fast; architectural diversity |
| LightGBM | — | Additional diversity |
| Logistic Regression | — | Linear model; diversity contributor |

### Key Finding: Lower GBDT Depth Works Better
> "I noticed that lower depths for GBDT algorithms (2, 3) worked better on this data."

**Why:** The data is synthetic with near-linear signal. Deep trees overfit the synthetic noise pattern. Shallow trees (stumps, depth 2–3) isolate cleaner categorical boundaries.

### RealMLP Configuration
- `n_cv=2` (internal cross-validation ensemble within RealMLP)
- Metric: `1-auc_ovr` (one-vs-rest AUC variant — better suited for this competition's dynamics)
- Reference: https://www.kaggle.com/competitions/playground-series-s6e2/discussion/674394

### CatBoost: Ordered vs Plain Boosting
- **Ordered boosting** was consistently better than Plain on this dataset
- Ordered boosting (CatBoost's default) uses permutation-based training that reduces overfitting on small datasets — here it helps with the synthetic noise

---

## Ensemble

### Step 1: Convert All OOFs to Ranks
```python
from scipy.stats import rankdata
oof_ranked = rankdata(oof_proba) / len(oof_proba)  # normalize to [0, 1]
```
Rank transformation before ensembling removes calibration differences between model families (GBDTs vs NNs tend to produce different probability scales).

### Step 2: GPU-Accelerated Hill Climbing
- Greedy hill climbing algorithm implemented on GPU
- Finds optimal blend weights over the ranked OOFs
- Solution code on GitHub (link in writeup)

---

## Key Takeaways

| Insight | Detail |
|---|---|
| **Original dataset TE is the only FE that reliably helps** | External target encoding adds clean signal without leakage |
| **ALL_CATS beats manual encoding for GBDTs** | CatBoost/LGB handle all-categorical input powerfully |
| **Ordered boosting > Plain boosting (CatBoost)** | Reduced overfitting on synthetic noise |
| **Low GBDT depth (2–3) beats deep trees here** | Near-linear signal; deep trees overfit noise |
| **RealMLP is the best neural model** | `n_cv=2` + `1-auc_ovr` metric is important config detail |
| **Rank transform before ensemble** | Removes calibration differences between model families |
| **GPU hill climbing for ensemble weights** | Faster and finds better weights than random search |
| **Trust CV strictly** | Never deviated from CV in favour of public LB |
