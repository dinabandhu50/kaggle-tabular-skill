# Action Items — Based on 1st Place Analysis

Prioritized implementation roadmap to close the gap from our current best (0.95542 OOF) to 1st place (0.95578 OOF).

---

## Phase 1 — Target Encoding + Categorical Features (High ROI: +0.0002–0.0003)

**Estimated impact:** Single biggest win. XGBoost BASE+TE alone is 0.955663 vs our 0.95542.

### [ ] 1.1 Implement fold-aware Target Encoding in `src/features.py`
- **What:** For each categorical feature, compute mean target per category within each fold
- **Why:** Directly encodes discriminative power without leakage
- **Affected features:** `Thallium`, `Chest_pain_type`, `Number_of_vessels_fluro`, `Slope_of_ST`, `EKG_results`
- **Code location:** Add `get_features_v3_with_te()` function
- **Validation:** Train XGBoost on BASE+TE, expect ~0.955650+ OOF
- **Effort:** ~2 hours

### [ ] 1.2 Add ALL_CATS feature set to `src/features.py`
- **What:** Convert all features to string type, let CatBoost/LGB handle as categoricals
- **Why:** Different split logic from numeric features; creates diverse OOFs
- **Code location:** Add `get_features_all_cats()` function returning df with dtype='category'
- **Config:** Create `configs/lgb_fe_all_cats.yaml` + `configs/catboost_baseline.yaml`
- **Validation:** LGB+ALL_CATS should score ~0.95561+
- **Effort:** ~1 hour

### [ ] 1.3 Add Frequency Encoding to feature set
- **What:** Value counts for categorical features (rare values carry signal)
- **Code location:** `get_features_v3_with_te()` or separate `get_features_freq()`
- **Implementation:** `df['feature_freq'] = df['feature'].map(df['feature'].value_counts())`
- **Effort:** ~0.5 hours

---

## Phase 2 — Model Diversity (ROI: +0.0001–0.0002)

**Estimated impact:** Generates diverse OOFs for ensemble. Each model+feature combo adds new error patterns.

### [ ] 2.1 CatBoost baseline (already queued)
- **What:** Run baseline training with CatBoost on BASE features
- **Config:** `configs/catboost_baseline.yaml` (standard hyperparams)
- **Validation:** Expect ~0.95540+
- **Effort:** ~1 hour

### [ ] 2.2 CatBoost + Target Encoding
- **What:** CatBoost with TE features + native categorical handling
- **Config:** `configs/catboost_te.yaml`
- **Validation:** Expect ~0.95560+
- **Effort:** ~0.5 hours

### [ ] 2.3 Multiple random seeds (5–10 per model)
- **What:** Retrain XGBoost/LGB/CatBoost with different random seeds, collect all OOFs
- **Why:** Cheap way to generate diverse predictions; 5 seeds × 3 models = 15 OOFs without new feature engineering
- **Code:** Modify `src/train.py` to accept `--seed` parameter, run 5 times per config
- **Storage:** Save each OOF as `submissions/oof_<model>_<seed>.npy`
- **Validation:** Verify OOF correlation across seeds (should be ~0.998–0.9995, not identical)
- **Effort:** ~1.5 hours

### [ ] 2.4 Neural tabular model (RealMLP or TabNet equivalent)
- **What:** Add a neural network for diversity; 1st place used RealMLP to 0.955739
- **Why:** Completely different error patterns than GBDT
- **Implementation options:**
  - Use `skorch` (scikit-learn compatible wrapper around PyTorch)
  - Or `TabNet` (PyTorch, built-in CV support)
  - Or hand-roll with PyTorch + early stopping on validation fold
- **Features:** BASE+TE should get best results
- **Validation:** Target 0.95560+
- **Effort:** ~3 hours (depends on implementation choice)

---

## Phase 3 — Ensemble Design (ROI: +0.0001–0.0002)

**Estimated impact:** Selecting best OOF subset + Ridge meta-model vs simple averaging.

### [ ] 3.1 Collect all OOFs into a matrix
- **What:** Load all saved OOFs (from Phase 1–2) into a single `(n_samples, n_models)` array
- **Code location:** New script `src/collect_oofs.py`
- **Output:** `submissions/all_oofs.pkl` or `.npy` + metadata (model names, CV scores)
- **Effort:** ~1 hour

### [ ] 3.2 Implement Optuna-based OOF selection
- **What:** Run Optuna with 500–1000 trials to find best *subset* of OOFs
- **Objective:** Maximize OOF AUC of weighted combination
- **Search space:** Binary indicator per OOF (select / don't select)
- **Weight constraint:** Ridge regression on selected subset (sklearn)
- **Code location:** `src/ensemble_optuna.py`
- **Validation:** Track selected OOFs, verify ~10% of total selected (sparse)
- **Effort:** ~2 hours

### [ ] 3.3 Train Ridge meta-model on selected OOFs
- **What:** Use sklearn Ridge (or LogisticRegression) to weight selected OOFs
- **Why:** Simple, stable, less prone to overfit than neural stacking
- **Code location:** Integrate into `src/ensemble_optuna.py`
- **Validation:** Verify meta-model weights are reasonable (no single OOF dominates)
- **Effort:** ~0.5 hours

### [ ] 3.4 Generate test predictions from ensemble
- **What:** Load all trained models, generate OOFs on test set, apply Ridge weights
- **Code location:** `src/predict_ensemble.py`
- **Output:** `submissions/ensemble_<timestamp>.csv`
- **Effort:** ~1 hour

---

## Phase 4 — Advanced Features (Lower priority, nice-to-have)

**Estimated impact:** +0.00005–0.0001 each (diminishing returns)

### [ ] 4.1 Quantile binning (qcut) features
- **What:** Complement equal-width bins with percentile-based bins
- **Code:** `pd.qcut(df['feature'], q=10)` alongside existing `pd.cut()`
- **Effort:** ~0.5 hours

### [ ] 4.2 Digit extraction (units/tens)
- **What:** Extract `Age_units = Age % 10`, `Age_tens = Age // 10`, etc.
- **Why:** Captures hidden structure in synthetic data
- **Code location:** `get_features_v4()` or separate function
- **Effort:** ~0.5 hours

### [ ] 4.3 External dataset statistics (Cleveland Heart Disease)
- **What:** Merge target mean / WoE / entropy from source Cleveland dataset
- **Why:** External target encoding adds signal
- **Source:** UCI ML Repository
- **Effort:** ~2 hours (data sourcing + merging)

### [ ] 4.4 DVAE (Denoising VAE) latent features
- **What:** Train a VAE on train data, use compressed latents as features
- **Why:** Adds nonlinear diversity
- **Implementation:** PyTorch or `pymc` probabilistic programming
- **Effort:** ~4 hours

---

## Implementation Order (Recommended)

1. **Start with Phase 1.1** (Target Encoding) — highest ROI, lowest effort
2. **Phase 1.2 + 1.3** (ALL_CATS + Frequency) — quick wins
3. **Phase 2.1 + 2.2** (CatBoost baseline + TE) — completes model trio (XGB, LGB, CatBoost)
4. **Phase 2.3** (Multiple seeds) — cheap diversity
5. **Phase 3.1 + 3.2 + 3.3** (OOF collection + Optuna selection + Ridge) — assemble ensemble
6. **Phase 2.4** (Neural model) — higher effort, do if time permits
7. **Phase 4.x** (Advanced features) — only if not hitting target

---

## Success Criteria

| Milestone | Target OOF | Target Public LB | Status |
|---|---|---|---|
| After Phase 1.1 (TE) | 0.95560+ | TBD | Pending |
| After Phase 1 (TE + ALL_CATS + Freq) | 0.95565+ | TBD | Pending |
| After Phase 2 (CatBoost + seeds) | 0.95570+ | TBD | Pending |
| After Phase 3 (Ensemble) | 0.95575+ | 0.9541+ | Pending |
| Stretch goal | 0.95578+ | 0.9543+ | Pending |

---

## Notes

- **Early stopping ceiling:** With search space fix (fixed n_estimators=5000, early_stopping=50), each trial runs in ~24s. Optuna 500 trials = ~3.5 hours.
- **GPU utilization:** All models use CUDA. Verify LGB rebuilt with `-DUSE_CUDA=1`.
- **CV–LB correlation:** Submit intermediate ensembles to verify the CV→LB slope remains positive. Stop if it breaks (split overfitting).
- **Fold consistency:** Use fixed `random_state=42` everywhere to ensure reproducibility.

