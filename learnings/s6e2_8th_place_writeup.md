# 8th Place Solution — Ensemble and TrustCV

**Author:** (anonymous)  
**Source:** https://www.kaggle.com/competitions/playground-series-s6e2/writeups/8th-place-ensemble-and-trustcv  
**Votes:** 8 | **Competition:** Playground Series S6E2 — Heart Disease Prediction

---

## Final Score

| Submission | Public LB | Private LB |
|---|---|---|
| Best LR stacking | 0.95388 | 0.95530 |
| **Best NN stacking (chosen)** | **0.95394** | **0.95533** |

---

## Competition Journey — 4 Phases

### Phase 1 — XGBoost Exploration (Core Focus)

Goal: build a diverse set of XGBoost models using different feature engineering strategies.

| Experiment | CV | Description |
|---|---|---|
| xgb-5fold-3TRG_FE | 0.95533 | 3 target encodings per fold |
| xgb-5fold-2TRG | 0.95532 | 2 target encodings per fold |
| xgb-5fold-TRG | 0.95532 | 1 target encoding per fold |
| xgb-5fold-ORG | 0.95535 | Aggregate features from original dataset |
| xgb-dmatrix-5fold | 0.95522 | Baseline, no FE |

**Naming convention:**
- `3TRG / 2TRG / TRG` → number of target encoding variants per fold
- `ORG` → aggregate features (mean, std, etc.) engineered from the original Cleveland dataset
- `DMatrix` → plain XGBoost on competition data only

**Key finding:** Performance difference was marginal (~0.0001 spread across all variants).

> "Feature engineering mattered more than hyperparameter tuning at this stage."

Used and improved a target encoding framework from a previous competition month.

---

### Phase 2 — Tabular Deep Learning (TabM)

TabM (from `pytabkit`) chosen over heavier models due to time constraints.

| Experiment | CV |
|---|---|
| TabM-Org-5fold | 0.95532 |
| TabM-5fold | 0.95470 |

- TabM was competitive with XGBoost but did not clearly outperform it
- The `Org` variant (with original dataset features) scored better — consistent with XGB findings

---

### Phase 3 — Neural Networks (Failed)

Tried multiple NN architectures:

| Model | CV |
|---|---|
| NN_lr1e-3_pembd_es | 0.95473 |
| NN_lr1e-3_pembd | 0.95472 |
| NN_lr1e-3_NoExpDec | 0.95362 |
| NN_Improved | 0.95393 |
| NN_Pytorch_5 | 0.95313 |

> "For this dataset, gradient boosting was simply stronger and more stable than deep neural networks."

None of the custom NN architectures surpassed tree-based models as standalone models. This is consistent with findings from 4th, 8th, and 10th place — GBDTs dominated for base models; NNs contributed only as ensemble members.

---

### Phase 4 — Stacking & Ensembling

Initially tried **Logistic Regression stacking** → minimal improvement.

Switched to **Neural Network Stacking** in the final 2 days.

| NN Stacking Variant | CV |
|---|---|
| NN_stacking_1 | 0.95538 |
| **NN_Stacking_2** | **0.95569** |

### What Caused the Big Jump (0.95538 → 0.95569)?
Two changes made simultaneously:
1. **Introduced Exponential Decay** (learning rate schedule)
2. **Introduced Early Stopping** with **patience = 7**

These two regularization techniques dramatically improved the NN meta-model's generalization.

### Models Used in Stacking
Public kernel predictions were used as stacking inputs (no custom OOFs available for all):

- RealMLP + Temperature Scaling
- RealMLP (baseline)
- Single XGBoost
- ResNet (tabular)
- TabM (variant 1)
- TabM (variant 2)

**Why public kernels?** They provide diverse architectures and inductive biases that the author's own models (all XGB/TabM variants) could not provide alone.

---

## Final Blend

At the very end, incorporated an external ensemble (`@mikhailnaumov`'s solution) via simple blending (no OOF available for proper stacking):

```
Final Prediction = 0.5 × (Own NN Stacking) + 0.5 × (Mikhail's Ensemble)
```

This gave a slight boost and stabilized private LB performance.

---

## Key Takeaways

| Insight | Detail |
|---|---|
| **Feature engineering > hyperparameter tuning** | All XGB variants had similar CV; FE was the differentiator |
| **Target encoding diversity adds value** | Multiple TE strategies per fold produce usefully different OOFs |
| **Original dataset aggregate features help** | ORG variant consistently scored slightly better |
| **GBDTs dominate as base models** | Custom NNs failed; tree models were more stable |
| **TabM is worth trying** | Competitive with XGBoost, fast to train |
| **NN stacking >> LR stacking** | NN meta-model outperformed LR meta-model for ensemble |
| **Exponential decay + early stopping (patience=7) is critical for NN stacker** | CV jumped from 0.95538 to 0.95569 with these two changes |
| **Public kernels as ensemble diversity sources** | Using diverse public model predictions broadened the ensemble beyond own experiments |
| **Simple blending as last resort** | When OOF not available, 50/50 blend is a valid emergency strategy |
| **Trust CV** | "TrustCV" is literally in the title — never deviated to chase LB |
