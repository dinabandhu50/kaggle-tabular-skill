# Feature Engineering (Phase 4)

The main ROI engine. Generate features with a reason (hypothesis-driven), but exploit volume where
it's cheap — combining many simple features often reveals signal models can't find alone. Run
features in **groups** and keep a group only if ΔCV exceeds the family's fold-noise (`cv_std`).
Everything here must obey HR-1 (fit per-fold) and HR-7 (inference-time available).

## High-yield families (roughly ordered by typical ROI on tabular)

1. **Aggregations / group statistics (highest ROI).**
   `df.groupby(cat)[num].agg(['mean','std','min','max','count','nunique', q25, q75])` merged back
   onto rows. Then **combine the aggregates** to manufacture second-order signal, e.g.
   `std/count`, `count/nunique`, `value - group_mean`, `value / group_mean`. On synthetic/Playground
   data, the group mean of the target-correlated column often acts like a hidden "true" value.

2. **Categorical interactions.** Concatenate pairs (and sometimes triples) of categoricals into new
   categorical columns so trees see interactions directly:
   ```python
   for i, c1 in enumerate(CATS[:-1]):
       for c2 in CATS[i+1:]:
           df[f"{c1}_{c2}"] = df[c1].astype(str) + "_" + df[c2].astype(str)
   ```
   k categoricals → up to C(k,2) new ones. Prune by CV; keep the survivors.

3. **Leak-free target / frequency encoding** (for high-cardinality categoricals).
   - **Out-of-fold target encoding** with smoothing: within each CV fold, compute category target
     means from the *other* folds only, then apply to the held-out fold (HR-1). Smooth toward the
     global mean so rare categories don't overfit:
     `enc = (count * cat_mean + m * global_mean) / (count + m)` with smoothing `m` ~ 10–100.
   - Or simply **let CatBoost handle categoricals natively** (Ordered Target Statistics) and skip
     manual target encoding for the CatBoost family.
   - **Frequency / count encoding** (category → its count) is cheap and leak-light but still fit on
     train rows only.

4. **Numeric transforms.** Binning / quantile bucketing; rank or quantile transforms; ratios and
   differences between related numerics; log/Box-Cox for skew; NaN-indicator columns (missingness is
   often signal); "digit extraction" from suspiciously quantized floats; rounding to expose
   discretization.

5. **Domain features.** Encode real relationships between columns (what they physically mean). These
   are usually the highest-signal features and the hardest for a model to discover alone — e.g.
   `price_per_sqft = price / area`, `total_area = basement + ground_floor`.

6. **Original-data merges.** For synthetic competitions (Kaggle Playground), the original dataset the
   synthetic data was generated from is frequently a strong merge source and a known leaderboard
   booster. Check the competition's data description and forums.

## Encoding cheatsheet by model family

- **Trees (LGBM/XGB/CatBoost):** prefer native categorical handling (CatBoost) or label/ordinal +
  let the tree split; add OOF target encoding for high-cardinality columns. One-hot only for very
  low cardinality.
- **Linear / NN members:** require explicit numeric encoding — one-hot for low cardinality, OOF
  target encoding for high cardinality, and **in-fold scaling/standardization** (HR-1).
- **High-cardinality (e.g. 50k cities):** one-hot is impractical; use OOF target encoding with
  smoothing, or CatBoost native handling.
- Avoid plain label-encoding for nominal categories fed to linear/NN models (imposes a false order).

## Per-family discipline (why FE is run separately per model)

Different model families benefit from different feature sets — a feature that helps a linear model
may be redundant for a tree, and vice versa. Run a dedicated FE-explorer per family so each ends
Phase 4 with its own surviving feature set. This maximizes the **diversity** of the OOF predictions
that Phase 6 ensembles, which is where the diversity payoff is realized.

## Acceptance rule

Accept a feature group iff `ΔCV > cv_std` for that family (i.e. the gain exceeds fold-to-fold noise),
and the Guardian confirms HR-1/HR-7. Otherwise drop it and record why in
`experiments/<model>/NOTES.md`. Stop the loop when recent groups stop clearing the noise bar.
