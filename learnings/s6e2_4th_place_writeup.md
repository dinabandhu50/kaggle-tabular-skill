# 4th Place Solution — "Less is More" (Stumps & Rank Ensembling)

**Author:** BlamerX  
**Source:** https://www.kaggle.com/competitions/playground-series-s6e2/writeups/4th-place-solution  
**Votes:** 13 | **Competition:** Playground Series S6E2 — Heart Disease Prediction

---

## Final Score

| Metric | Value |
|---|---|
| **Private LB** | **0.95534** |
| Teacher Ensemble (best CV component) | 0.95580 CV / 0.95535 private |

---

## Core Philosophy: "Raw is Law"

> "The signal was nearly linear, and the associated synthetic data generation mechanisms heavily penalized over-engineering."

The central insight that shaped everything: **avoid aggressive non-linear transformations**. The data is synthetic but the generative model encodes near-linear patterns. Over-engineering creates collinearity noise that hurts more than it helps.

### Proof of Near-Linear Signal
- Basic Logistic Regression + One Hot Encoding (13 categorical features → **449 OHE dimensions**) → **CV 0.95550, LB 0.95371**
- This is competitive with deep GBDT models
- Conclusion: primary predictive boundaries are near-linear and clean

---

## 1. Adversarial Validation

Before any modeling:
- Train XGBoost to distinguish train vs test rows
- Result: **AUC = 0.501** — perfect distribution match, zero drift
- Impact: enables use of original dataset statistics without distribution mismatch concern

---

## 2. Original Dataset — Bifurcated Strategy

Because the competition data is synthetic, the real Cleveland dataset was a critical lever. But different architectures need it used differently:

| Architecture | How to Use Original Dataset | What Fails |
|---|---|---|
| **Tree-Based (GBDT)** | OOF target probability + frequency encoding from original data (+0.00028 LB lift) | 800+ polynomial interactions → collinearity crashes CV |
| **Neural Networks** | Use original data **only** as anchor for aggregated statistics (mean, std, skew per group) | Appending original data as actual training rows → completely confuses the deep network |

**Critical rule for NNs:** Never append original dataset rows as training rows. The domain shift between the small, clean original dataset and the large synthetic dataset confuses neural nets. Use only aggregated statistics.

---

## 3. Model Architecture

### A. Gradient Boosted Stumps (max_depth=2)
- **Why stumps?** Cleanly isolates category levels without overfitting synthetic noise
- Combined with One Hot Encoding (not target encoding — OHE preserves clean category separation)
- Consistent local CV: **~0.95574**
- Key: lower depth was explicitly better than deeper trees

### B. Periodic Embedding MLPs
Neural networks with sinusoidal input encoding:
```
Numerical features → Periodic Embeddings (sin/cos, dim=8 per feature)
Categorical features → One Hot Encoding
↓ Concatenate
4-layer MLP · 384 units per layer · Mish activation
↓
Output Averaging (8 internal sub-models trained simultaneously)
```
**Why periodic embeddings?** Bypass "spectral bias" — standard MLPs struggle to learn high-frequency patterns from raw numerical inputs. Sinusoidal encoding gives the network pre-computed frequency representations.

### C. RealMLP (pytabkit)
- Internal 8-model averaging for stability (`n_ens=8`, 256 batch size, 100 epochs)
- Each of the 8 sub-models trains on the same data with different initialization
- Final output: average of all 8 predictions

---

## 4. Gap Tracking — Anti-Overfitting Discipline

Standard CV trust is not enough on synthetic data. Added an extra layer:

> "I religiously tracked the OOF-to-Public LB Gap."

### Healthy Gap Definition (S6E2 specific)
```
CV - Public LB ≈ 0.00185   →   healthy, model generalizes
CV - Public LB >> 0.00190  →   experiment rejected (overfitting noise)
```

### What triggers rejection
- CatBoost models often had high CV but gap > 0.00190 → discarded
- Any experiment that widens the gap beyond threshold is suspect even if CV improves

**Implementation:** Track `cv_score - public_lb_score` for every submitted model. Build a table. Trust only models where gap stays within the empirical range.

---

## 5. Optimization Journey (Ensemble Construction)

| Phase | Challenge | Solution | Result |
|---|---|---|---|
| **Phase 1: OOF Trap** | Unconstrained optimizer gave ~65% weight to CatBoost → LB collapse | Manually cap tree models at 35% weight; blend only elite models (solo LB > 0.9538) | Proved: Purity > Diversity on this dataset |
| **Phase 2: Distillation** | Extract signal from complex Teacher blend into stable Student model | Extract hard labels (confidence > 99% or < 1%, ~48k test rows); train Student NN on Train + Hard Test rows | Student NN had highest standalone CV |
| **Phase 3: Rank Ensembling** | Raw probability averaging skewed by calibration differences | Convert 4 optimized blends to ranks → average ranks → normalize to [0,1] | **0.95534 private LB, robust to shake-up** |

### Rank Ensembling Flow

```
4 top-tier probability blends
↓
Convert each to rank vector (1 to N)
↓
Average the 4 rank vectors row-by-row
↓
Normalize to [0, 1]
↓
Final submission
```

**Why rank ensembling beats probability averaging:**
- Rank is invariant to calibration differences between model families
- Extreme confidence predictions (outliers) have bounded influence
- Targets relative order, ignoring absolute probability scale
- "Mathematical stabilization" — insulates against public-to-private distribution variance

### Component Blend Results

| Component | CV | Public LB | Private LB |
|---|---|---|---|
| Teacher Ensemble (RealMLP 60% + CatBoost 40%) | 0.95580 | 0.95396 | 0.95535 |
| Distillation Student (pseudo-labeled NN) | 0.95567 | 0.95397 | 0.95531 |
| Power-Averaged Blend (sharpened) | 0.95579 | 0.95397 | 0.95535 |
| Apex Blend | 0.95572 | 0.95397 | 0.95535 |
| High-Purity Blend (CatBoost capped 35%) | 0.95578 | 0.95398 | 0.95535 |
| **Selected Rank Blend (final submission)** | 0.9557X | **0.95398** | **0.95534** |

---

## Key Takeaways

| Insight | Detail |
|---|---|
| **Run LogReg first to test signal linearity** | LogReg + OHE → CV 0.95550; if competitive with GBDTs, signal is near-linear |
| **Adversarial validation before external data** | AUC=0.501 → safe to use original dataset |
| **Never append original data as NN training rows** | Use only as statistics anchor (mean, std, skew) for injecting context |
| **max_depth=2 stumps + OHE beats deeper trees** | Near-linear signal; stumps isolate category levels cleanly |
| **Periodic embeddings for neural nets** | Bypass spectral bias; sin/cos of numerics, dim=8 per feature |
| **RealMLP n_ens=8 for stability** | Internal averaging of 8 sub-models; 256 batch, 100 epochs |
| **Track OOF-to-LB gap per experiment** | Healthy gap ~0.00185; reject experiments that widen gap beyond 0.00190 |
| **Cap dominant models in unconstrained ensemble** | CatBoost at 65% weight overfits; cap trees at 35% |
| **Rank ensembling > probability averaging** | Removes calibration bias; robust to shake-up |
| **Purity > Diversity on near-linear signal datasets** | Fewer, cleaner models can beat many diverse models |
