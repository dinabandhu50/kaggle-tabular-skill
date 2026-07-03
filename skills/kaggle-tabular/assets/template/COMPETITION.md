# COMPETITION.md — {{COMP_NAME}}

Single source of truth for competition facts. Fill in during Phase 0; upgrade the CV decision from
"hypothesis" to "confirmed" at the end of Phase 1.

## Task
- **Type:** <binary / multiclass / regression / ranking>
- **Target column:** <name> — format: <0/1 | probability | continuous | class id>
- **Exact metric:** <AUC / RMSE / LogLoss / MAP@k / ...>  — implemented in `src/metric.py`? [ ]
- **GREATER_IS_BETTER:** <true|false>

## Data
- train rows × cols: <...>
- test rows × cols: <...>
- **Group structure:** <none | column that groups rows, e.g. user_id>
- **Time structure:** <none | time column>
- Missing data: <where / how much>
- High-cardinality categoricals: <list>
- Leakage suspects (IDs, timestamps, too-predictive columns): <list>

## CV decision (HR-2, HR-3)
- **Scheme:** <stratified | kfold | group | time> — **why:** <matches test structure because ...>
- **n_folds:** <5>
- **Metric reproduces a known LB point?** [ ]  (trivial submission LB = <...>, local = <...>)
- **Adversarial validation AUC:** <...>  → shift? <no (~0.5) | yes, shifted features: ...>
- **Status:** <hypothesis | CONFIRMED>

## Logistics
- Daily submission limit: <...>
- Final submissions to select: 2 (by CV — see references/orchestration.md)
- Deadline: <...>
- External data allowed? <...>   Pseudo-labeling on test allowed? <...>

## External links (for periodic intel gathering — see references/orchestration.md)
- Overview: <url>
- Discussion: <url>
- Code (public notebooks): <url>
