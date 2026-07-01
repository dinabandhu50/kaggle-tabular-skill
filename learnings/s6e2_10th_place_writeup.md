# 10th Place Solution

**Author:** (anonymous)  
**Source:** https://www.kaggle.com/competitions/playground-series-s6e2/writeups/10th-rank-solution-playground-series-s6e2  
**Votes:** 8 | **Competition:** Playground Series S6E2 — Heart Disease Prediction

---

## Final Score

| Metric | Value |
|---|---|
| Public LB | 0.95393 |
| **Private LB** | **0.95534** |
| Hill Climbing Ensemble OOF | ~0.95578+ |

---

## Overview

End-to-end strategy: diverse gradient boosting + neural network base models → GPU-accelerated hill climbing in logit space for ensemble weights.

Three differentiators from simpler approaches:
1. **Rich, systematic feature engineering** (6 distinct FE strategies)
2. **Nine diverse base models** across different architectures and FE combinations
3. **Hill climbing in logit space** with negative weight support

---

## Cross-Validation Strategy

Used **10-fold** Stratified K-Fold (most other solutions used 5-fold):

```python
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
```

**Why 10-fold?** More folds → lower variance OOF estimates → better ensemble signal. Trade-off: 2× training time vs 5-fold.

---

## Feature Engineering — 6 Strategies

### 1. Categorical Duplication
Every numeric column is duplicated as a categorical (label-encoded) column:
```python
for col in num_cols:
    df[f'{col}_cat'] = df[col].astype(str)  # or label encode
```
Allows tree models and neural networks to discover non-linear ordinal relationships in numeric features treated as discrete.

### 2. Bi-gram and Tri-gram Interaction Features
Concatenate pairs and triplets of categorical columns into composite string features, then **target-encode** them:
```python
df['Sex_Thallium'] = df['Sex'].astype(str) + '_' + df['Thallium'].astype(str)
# Then: apply target encoding to 'Sex_Thallium'
```
- Initial: all pairwise + triplet combinations generated
- Selection: trained LogisticRegression on all combinations, selected **top 20 by coefficient magnitude**, manually optimised to **32 features** based on CV improvement
- These were among the most impactful features

### 3. Target Encoding from Original Dataset
The original (non-synthetic) Cleveland Heart Disease dataset was used as an external TE source:
```python
# For each feature value, compute from original dataset:
# mean, median, std, skew, count
# Merge as new columns into training data
```
No fold leakage — labels come from an independent external dataset.

### 4. Sinusoidal + Digit Encoding
Two-part numeric encoding for neural networks:

**Sinusoidal (periodic features):**
```python
for col in num_cols:
    for p in [12, 14, 20]:
        df[f'{col}_sin_{p}'] = np.sin(2 * np.pi * df[col] / p)
        df[f'{col}_cos_{p}'] = np.cos(2 * np.pi * df[col] / p)
```

**Digit extraction:**
```python
df[f'{col}_units'] = df[col] % 10
df[f'{col}_tens'] = (df[col] // 10) % 10
```
Both transformations as categorical columns — helps NNs detect periodic/digit-structured patterns in clinical measurements.

### 5. Count Encoding (for NN with Embeddings only)
Every categorical column + all **pairwise combinations** (136 total) count-encoded:
```python
for col in all_cat_cols:  # includes pairwise combos
    df[f'{col}_count'] = df[col].map(df[col].value_counts())
```
Captures frequency information; used specifically for the MLP-with-embeddings model.

### 6. Domain-Specific Composite Features
Cardiology-grounded features:
```python
df['cardiac_wrkload_age'] = df['BP'] * df['Max_HR'] / df['Age']
df['stress_severity']     = df['ST_depression'] * df['Slope_of_ST']
df['high_risk']           = (df['FBS_over_120'] 
                             + (df['BP'] > 140).astype(int) 
                             + (df['Cholesterol'] > 240).astype(int))
```

---

## Nine Base Models

Each model produced OOF + test predictions saved as CSV files.

| Model | OOF AUC | Key Config |
|---|---|---|
| XGBoost (Base + Cats) | ~0.95565 | `max_depth=2`, GPU, up to 25k rounds, early stopping |
| CatBoost (Base + Cats) | ~0.95569 | Native categorical, `one_hot_max_size=10` |
| XGBoost (Target Encoding) | ~0.95556 | TE features |
| CatBoost (Orig Dataset TE) | ~0.95561 | External TE from original dataset |
| XGBoost → LogReg (Leaf Stacking) | ~0.95486 | Leaf indices OHE → LogReg meta |
| RealMLP (Base) | ~0.95570 | pytabkit, early stopping, AUC metric |
| RealMLP (TE) | ~0.95551 | TE features |
| RealMLP (More FE) | ~0.95569 | Full FE pipeline |
| MLP + Categorical Embeddings | ~0.95543 | PyTorch, pairwise cat combos, OneCycleLR |

### Notable Model: XGBoost → LogReg (Leaf Stacking)
```python
# Train XGBoost on data
# Extract leaf indices for each tree
# One-hot encode leaf indices → high-dimensional sparse feature matrix
# Train Logistic Regression on this matrix
```
XGBoost acts as a non-linear feature transformer. The leaf indices encode which decision path each sample fell into. Logistic Regression on top learns a linear combination of these paths.
OOF was ~0.95486 — weaker as standalone but contributed ensemble diversity.

### MLP with Categorical Embeddings
- Custom PyTorch MLP
- Learned embedding tables for **all categorical features** (including 136 pairwise combinations)
- Standardised numerics
- OneCycleLR learning rate scheduling
- Count encoding for all 136 pairwise cat combos as input

---

## Ensembling: GPU-Accelerated Hill Climbing in Logit Space

### Why Logit Space?
```python
from scipy.special import logit, expit

# Before blending, convert probabilities to logit space
oof_logits = logit(np.clip(oofs, 1e-7, 1 - 1e-7))

# After combining, convert back
ensemble_proba = expit(weighted_sum_of_logits)
```
**Reason:** Probability averaging near 0 or 1 compresses gradients (0.99 + 0.98 = 0.985, but logit(0.99) + logit(0.98) maps to a richer space). Logit-space blending avoids boundary saturation effects.

### Algorithm Details
- Greedy hill climbing: add models one at a time, accept if ensemble AUC improves
- **Supports negative weights** — can subtract a model if it hurts the ensemble
- GPU-accelerated for speed
- Stopping: tolerance 1e-7 improvement OR maximum 1,000 iterations

### Key Innovation: Negative Weights
Allowing negative weights means the hill climbing can effectively *subtract* the effect of a model that introduces correlated noise. This is more powerful than constrained positive-weight blending.

---

## Key Takeaways

| Insight | Detail |
|---|---|
| **10-fold CV reduces OOF variance** | Better ensemble signal than 5-fold; worth the 2× training cost |
| **Bi/tri-gram interactions + TE are the most impactful FE** | Selected top 32 via LogReg coefficients |
| **External dataset TE (no leakage)** | Original dataset statistics merged as features; clean and powerful |
| **Categorical duplication of numerics** | Allows models to treat ordinal numerics as discrete categories |
| **Sinusoidal + digit encoding for NNs** | Periodic features help NNs find frequency/digit patterns |
| **Count encoding for pairwise combos** | 136 pairwise cat combos count-encoded for embedding model |
| **XGBoost → LogReg leaf stacking** | Weak standalone but unique error patterns → ensemble value |
| **MLP with embeddings for all cat combos** | 136 pairwise combos as learned embeddings |
| **Hill climbing in logit space** | Avoids boundary saturation vs probability space |
| **Negative weights in hill climbing** | More expressive than positive-only; can subtract harmful models |
| **Diverse models beat homogeneous ensembles** | GBDT + NN produced complementary signal |
