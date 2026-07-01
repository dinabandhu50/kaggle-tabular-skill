# Model Menu (2026) — what to reach for and when

GBDTs are the workhorse and the backbone of every ensemble. Tabular foundation models (TabPFN-class)
are now a real baseline for small data and a strong diversity source — **not** a GBDT replacement at
scale. Pick for both *strength* and *decorrelation* (Phase 6 rewards diversity).

| Situation | First choices | Notes |
|---|---|---|
| Default tabular, > ~10k rows | **LightGBM, XGBoost, CatBoost** | Win the vast majority of tabular competitions; the backbone of every ensemble. Start here always. |
| Many / high-cardinality categoricals | **CatBoost** (native Ordered Target Statistics) | Avoids manual leak-prone encoding; excellent default; symmetric trees → fast inference. |
| Small data (≤ ~10k rows, ≤ ~500 features) | **TabPFN-2.5** + GBDTs | TFM often beats tuned GBDTs in a single forward pass and needs no FE/preprocessing; superb ensemble diversity. Memory-bound past ~10k rows — sample or cluster-subsample if larger. |
| Strong automated baseline / extra diversity | **AutoGluon** | A good "free" ensemble member and a sanity bar; expensive in extreme/long modes. |
| Ensemble decorrelation | add **linear (Ridge/Logistic), MLP/NN, KNN, RandomForest, SVR** | Individually weaker but decorrelate the blend; cheap members that lift the ensemble. |
| Temporal / sequence-flavored tabular | GBDTs + optionally a small **GRU/1D-CNN** member | Some structured comps benefit from a sequence member alongside GBDTs; keep it as one ensemble voice, not the whole solution. |

## Selection heuristics

- **Always** run the three GBDTs as core members; they rarely lose and they anchor the stack.
- Use **CatBoost** as the categorical specialist and **LightGBM** as the fast iterator for FE loops
  (speed = more experiments).
- Add **TabPFN-2.5 / AutoGluon** when data is small enough — cheap, strong, and decorrelated.
- Add **linear + NN** members primarily for diversity once the GBDTs plateau, even if individually
  behind.
- Prefer members that are **strong AND decorrelated** (check OOF correlation) over a third near-clone
  GBDT.

## Practical notes

- GBDT defaults + early stopping are a strong starting point; don't over-tune before FE plateaus
  (see `workflow-phases.md` Phase 5).
- TabPFN-2.5 hard limits: roughly ≤10k rows and ≤500 features per forward pass; for larger sets,
  subsample (random or K-Means cluster-representative) or skip it as a member.
- Keep GPU vs CPU consistent with how the repo was scaffolded (`--gpu` toggles RAPIDS/cuDF/cuML and
  GPU GBDT backends). GPU mainly buys throughput, which buys more experiments.
