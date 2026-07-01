# Learnings

A curated collection of insights, techniques, and lessons from analyzing top solutions and building our own Kaggle competition models.

## Contents

### Master Pipeline
- **[refined-pipeline.md](./refined-pipeline.md)** — ⭐ Master pipeline: grandmaster framework + top-10 writeup evidence + operational preferences
  - Two foundations, ROI order, Hard Rules HR-1 to HR-7
  - All 7 phases: Setup → Validation → EDA → Baselines → FE → Tuning → Ensembling → Final submission
  - Cross-cutting patterns seen in 3+ top solutions
  - What does NOT work (evidence-backed)
  - Scores reference map and operational checklist

### S6E2 Individual Writeup Analyses
- **[s6e2_1st_place_writeup.md](./s6e2_1st_place_writeup.md)** — 1st place (CV 0.955780, Private 0.95535): Diversity, Selection, Trusting CV–LB Relation
- **[s6e2_2nd_place_writeup.md](./s6e2_2nd_place_writeup.md)** — 2nd place (CV 0.955759, Private 0.95535): In-fold vs global TE, 105→6 model selection, NN stacking
- **[s6e2_3rd_place_writeup.md](./s6e2_3rd_place_writeup.md)** — 3rd place (CV 0.955803, Private 0.95535): ALL_CATS + Hybrid, Ordered CatBoost, rank hill climbing
- **[s6e2_4th_place_writeup.md](./s6e2_4th_place_writeup.md)** — 4th place (Private 0.95534): "Less is More" — stumps, periodic embeddings, gap tracking, rank ensembling
- **[s6e2_8th_place_writeup.md](./s6e2_8th_place_writeup.md)** — 8th place (Private 0.95533): XGB + TE diversity, TabM, NN stacking with exp decay + early stopping
- **[s6e2_10th_place_writeup.md](./s6e2_10th_place_writeup.md)** — 10th place (Private 0.95534): 6 FE strategies, 9 models, GPU hill climbing in logit space with negative weights

### Implementation Roadmap
- **[action_items.md](./action_items.md)** — Prioritized implementation roadmap (Phase 1: TE, Phase 2: model diversity, Phase 3: ensemble)

---

## How to Use This

When implementing new techniques or features:
1. Check if there's a relevant learnings file
2. Review the action items
3. After implementing, update the corresponding file with results/notes
4. Add new learnings files as we discover more insights

## References

1. https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t
2. https://www.kaggle.com/competitions/playground-series-s6e2/writeups/2nd-place-solution-avoid-leaks-and-overfitting
3. https://www.kaggle.com/competitions/playground-series-s6e2/writeups/3rd-place-solution
4. https://www.kaggle.com/competitions/playground-series-s6e2/writeups/4th-place-solution
5. https://www.kaggle.com/competitions/playground-series-s6e2/writeups/8th-place-ensemble-and-trustcv
6. https://www.kaggle.com/competitions/playground-series-s6e2/writeups/10th-rank-solution-playground-series-s6e2