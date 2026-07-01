# 2nd Place Solution — Avoid Leaks and Overfitting

**Author:** Akiyoshi Kinoshita  
**Source:** https://www.kaggle.com/competitions/playground-series-s6e2/writeups/2nd-place-solution-avoid-leaks-and-overfitting  
**Votes:** 12 | **Competition:** Playground Series S6E2 — Heart Disease Prediction

---

## Final Scores

| Model | CV | Public LB | Private LB |
|---|---|---|---|
| **In-fold TE (chosen, 2nd place)** | 0.955759 | 0.95394 | **0.95535** |
| Global TE (not chosen) | 0.955774 | 0.95394 | 0.95534 |
| Hill climb (in-fold, with CV loop) | 0.9557888 | 0.95394 | 0.95535 |
| Hill climb (global TE, with CV loop) | 0.9558057 | 0.95394 | 0.95534 |

**Key observation:** The global TE model had *higher* CV (0.955774) but *lower* private LB (0.95534). The in-fold model won despite lower CV — confirming in-fold TE is safer.

---

## The Central Problem: Target Encoding Inside vs Outside CV Loop

This writeup is essentially an exploration of one critical question:

> Should target statistics (mean, count, etc.) be calculated **globally** (outside the CV loop) or **inside the CV loop** (fold-aware)?

### The Dilemma
- **Global TE** → higher CV score (more information leaks into features) → potentially falsely inflated CV
- **In-fold TE** → lower CV score, but leakage-free → more trustworthy

### The Evidence
- Adversarial validation: **Train vs Test AUC (XGB) = 0.5017 ± 0.0013** → near-perfect distribution match
- Because train/test distributions are almost identical, global TE was likely safe
- But to be certain, the author built **both versions** and submitted one of each

### The Verdict (private LB)
- In-fold model won by 0.00001 on private LB — the difference was negligible
- Conclusion: *Either approach works on this specific dataset*, but in-fold TE is the principled default

**Rule extracted:** Always build in-fold TE first. Only consider global TE when adversarial validation confirms near-zero distribution shift — and even then, test both.

---

## Model Selection Pipeline

105 total models examined → narrowed down to 6 for final stacking:

```
105 models
  (XGB, LGBM, CatBoost, RealMLP, TabM, Pairwise Ranking AUC NN)
↓
50 models  — keep only in-fold TE models
↓
15 models  — keep only multi-seed averaged results (remove single-seed)
↓
4 models   — remove results with correlation coefficient > 0.9999
↓
6 models   — add rank-transformed versions of 2 models (2 CatBoost + 4 RealMLP)
↓
NN stacking (StratifiedKFold 5 splits)
```

### Selection Logic — 3 Steps
1. **Remove highly correlated OOFs** (threshold = 0.9999) — eliminates near-duplicate predictions
2. **Forward Selection** — greedily add models that improve ensemble AUC
3. **Backward Elimination** — remove models that don't contribute after forward selection

### Why Multi-Seed Averages Only?
Single-seed OOFs carry seed-specific noise. Averaging over multiple seeds produces smoother, more stable OOFs that are better ensemble members. Requiring multi-seed averages also reduces the effective model count (50 → 15), making the correlation-based deduplication more meaningful.

### Why Rank Transformation?
Two of the final 6 models were rank-transformed versions of existing probability outputs:
```python
from scipy.stats import rankdata
oof_rank = rankdata(oof_proba) / len(oof_proba)
```
Rank transformation reduces the influence of extreme confidence predictions and can make the ensemble more robust to calibration differences between models.

---

## Ensemble Method: NN Stacking

The final meta-model was a **Neural Network**, not Ridge or simple averaging.

### Why NN Over Ridge?
- After aggressive deduplication (105 → 6), the remaining OOFs are already diverse and decorrelated
- With only 6 inputs, overfitting risk is low enough that a small NN is stable
- NN can capture small nonlinearities between the 6 diverse OOFs

### NN Stacking Configuration
- Input: 6 OOF predictions
- Cross-validation: StratifiedKFold 5 splits
- The NN was relatively simple (details in linked notebooks)

---

## Alternative: Hill Climbing with CV Loop

Also tested a custom hill climbing algorithm with a built-in anti-overfitting guard:

> "This algorithm tests the weights obtained from the training data on the validation data. Only if the AUC of **both** the training data **and** validation data improves will the results and weights of that model be adopted."

This is a conservative hill climbing variant: reject any step that degrades validation AUC even if it improves training AUC. Results were comparable to NN stacking but the CV appeared inflated, so NN stacking was chosen.

---

## Key Takeaways

| Insight | Detail |
|---|---|
| **In-fold TE is safer** | Won marginally vs global TE on private LB despite lower CV |
| **Multi-seed averages only** | Single-seed OOFs are too noisy for stacking selection |
| **Correlation deduplication** | Remove OOFs with correlation > 0.9999 before stacking |
| **Rank transformation adds value** | Stabilizes predictions from extreme confidence models |
| **NN stacking works on small, decorrelated input** | After aggressive selection (6 models), NN meta-model is stable |
| **CV–LB relation matters most** | Higher CV did not guarantee higher private LB across model families |
| **Adversarial validation guides TE strategy** | AUC = 0.5017 → in-fold and global TE both acceptable here |

---

## Notebook Links

- Final (rank transform + NN stacking): https://www.kaggle.com/code/satokin13m/s6e2-nn?scriptVersionId=300608701
- Model selection (4 OOFs): https://www.kaggle.com/code/satokin13m/s6e2-nn?scriptVersionId=300590084
- CatBoost (in-fold TE): https://www.kaggle.com/code/satokin13m/s6e2-ynong
- RealMLP v2: https://www.kaggle.com/code/satokin13m/s6e2-multi-seeds-realmlp?scriptVersionId=296465842
- RealMLP v3 (n_ens=20): https://www.kaggle.com/code/satokin13m/s6e2-multi-seeds-realmlp?scriptVersionId=299135216
- Hill climb dataset: https://www.kaggle.com/datasets/satokin13m/s6e2-myhillclimbdata
