# What Works / What Doesn't (evidence-backed)

Distilled from top-10 tabular competition solutions. Patterns seen in 3+ independent winners are the safest bets; the "does not work" table lists techniques that repeatedly disappointed. Load this for a fast prior before committing effort in any phase.

## Cross-cutting patterns (seen in 3+ top solutions)

| Pattern | Why it wins |
|---|---|
| Adversarial validation as step 1 | confirms K-fold trust; shapes the encoding strategy |
| Trust CV strictly, never chase public LB | private LB differs; LB is a noisy probe (HR-5) |
| In-fold target encoding | single most impactful FE |
| ALL_CATS feature set | different split geometry → diverse OOFs |
| Low GBDT depth (2–3) on near-linear signal | stumps generalize better on synthetic noise |
| RealMLP as the neural member | strongest single NN; decorrelates the blend |
| External-dataset target statistics | leak-free extra signal |
| Rank transform before ensembling | removes cross-family calibration differences |
| Multi-seed averaging | cheap variance reduction |
| OOF dedup (corr > 0.9999) | removes redundant members before stacking |
| Greedy hill climbing (logit space) | beats uniform averaging; negative weights subtract noise |
| CV−LB gap tracking | hard-reject experiments that widen the gap |

## What does NOT work (evidence)

| Technique | Finding |
|---|---|
| Pseudo-labeling / soft-label distillation | did not improve CV (1st place) |
| Averaging all OOFs | dilutes signal (1st place) |
| Nonlinear stacking without selection | overfits stacked OOFs (1st place) |
| 800+ polynomial interactions | collinearity noise crashed CV (4th place) |
| Appending original data as NN rows | domain shift confused the net (4th place) |
| Unconstrained ensemble weight optimizer | 65% weight to one model → LB collapse (4th place) |
| Custom NN base models | GBDTs dominated (8th place) |
| Deep trees (>4) on synthetic data | overfits noise; depth 2–3 better (3rd/4th place) |
| Public-LB selection | misleading; private LB can differ (multiple) |
