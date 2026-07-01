# Ensembling & Final-Mile (Phase 6–7)

Where the leaderboard is actually won. Winning tabular solutions routinely combine 100+ models in
multi-level stacks. The precondition is **diverse** strong single models, each with OOF + test preds
on the *identical* folds (HR-2). Diversity between members matters more than any single member's
strength. All of this operates over the OOF ledger — the ensembler never needs to know how a model
was built.

## Escalate in this order

### 1. Hill climbing (the workhorse blend)
Start from the best single-model OOF. Greedily search for the model + weight to add that most
improves the OOF metric; keep only additions that improve; repeat to convergence. Robust, cheap, and
hard to overfit. Members may be added with replacement (allowing integer-like weights). `src/
ensemble.py` ships a `hill_climb()` over the OOF arrays.

### 2. Stacking (when models capture different structure)
Train a Level-2 **meta-model** on the OOF predictions as features (the OOF matrix → target). Respect
HR-2: the meta-model is CV'd on the *same* folds, and its inputs are the base models' OOF (never
in-fold-fit predictions). Two equivalent framings:
- **OOF features:** stack base OOF columns and learn the best combination.
- **Residuals:** train Stage-2 on what Stage-1 got wrong.
Try meta-models in increasing complexity: Ridge / Logistic first (regularized, hard to overfit),
then GBDT / MLP stackers. Go **multi-level only if each level beats the prior level on OOF** — depth
for its own sake overfits.

### 3. Distillation (optional simplification)
Train a new single GBDT/NN on the ensemble's OOF/test predictions as **soft targets** (knowledge
distillation). Sometimes yields a simpler model that matches or beats the full stack and is easier to
retrain on full data.

## Overfitting guards for the meta-layer

- The meta-model can overfit the OOF matrix. Keep it simple and regularized.
- If a stacker's OOF gain over hill-climbing is within fold-noise, prefer the hill-climbing weights.
- Watch the CV↔LB relationship (see `orchestration.md`): if stacking improves CV but not LB, the
  meta-model is overfitting — simplify.
- Correlated near-duplicate members add cost, not signal. Inspect the OOF correlation matrix and
  prune redundant members before stacking.

## Pseudo-labeling (Phase 7, if rules permit using test/unlabeled data)

Use the strong ensemble to label unlabeled/test data, fold those labels back into training, retrain.
- Prefer **soft labels** (probabilities) — they add signal, regularize, and let you filter
  low-confidence rows.
- A stronger teacher → better pseudo-labels; ensembles and multi-round pseudo-labeling beat a single
  pass.
- **Avoid leakage (HR-1 extends here):** with k-fold, compute *k separate* pseudo-label sets so a
  validation fold never sees labels from a model trained on itself.
- Optionally fine-tune on the original labeled data as the last step to shed pseudo-label noise.

## Final-mile strengthening (Phase 7)

- **Seed ensembling.** Retrain the final model(s) across many random seeds and average predictions.
  Different initializations/training paths decorrelate errors; averaging reliably reduces variance.
- **Retrain on 100% of the data.** Once features and hyperparameters are frozen, the final fit uses
  all training rows (no validation fold needed at the very end). Squeezes out extra accuracy.
- **Lock submissions by CV.** Choose the final pair by CV, not public LB (see `orchestration.md` →
  CV–LB contract): typically (1) the best-CV ensemble and (2) a lower-variance / more conservative
  blend to hedge a shake-up.

## Ensembling ladder (in order)

1. **Build a diverse OOF library** — 20–150 OOFs from model × feature-set × seed. 5 seeds × N variants is cheap diversity. (`src/ensemble.py::average_seeds`.)
2. **Deduplicate** — drop one of any pair with OOF Pearson correlation > 0.9999. (`src/ensemble.py::dedup_oofs`.)
3. **Prefer multi-seed averages** as ensemble members over single-seed OOFs (cleaner signal).
4. **Subset selection** — Optuna over which OOFs to include, maximizing the CV metric of a Ridge/Logistic combo; ~10% of OOFs typically survive. Optional forward-selection + backward-elimination for stability at small counts.
5. **Rank transform** (recommended) — replace probabilities with `rankdata/n` before blending to erase calibration differences between GBDTs and NNs. (`src/ensemble.py::rank_transform`.)
6. **Hill climbing in logit space with negative weights** — blend `logit(clip(oof))`, greedily add members (negative weights allowed to subtract correlated noise), stop on tolerance 1e-7 or max iters, output `expit(sum)`. (`src/ensemble.py::hill_climb_logit`.)

### Meta-model choice

| Meta-model | When |
|---|---|
| Ridge | default; 10+ selected OOFs |
| Logistic | binary classification; add `C` tuning |
| Small NN | only after aggressive dedup (≤ ~6 inputs) |

## Full-data refit + final decision (Phase 7)

- Record avg `best_iteration` across folds; retrain on 100% of train at `int(avg_best_iter * 1.25)` rounds, averaged over ~20 seeds. More data → afford slightly more rounds. (`src/finalize.py::full_data_refit`.)
- **Decide by the CV–LB relation, not the single best CV** (see hard-rules.md → CV–LB gap). Submit intermediate ensembles throughout to map the slope; pick from the still-positive-slope range.
- Lock **two** submissions: the best-trustworthy-CV blend and a conservative/lower-variance fallback to hedge a shake-up.

## Ensembling anti-patterns

| Anti-pattern | Why it fails |
|---|---|
| Simple average of all OOFs | dilutes signal with redundant members |
| One model given 65%+ weight | overfits OOF, collapses on LB |
| Nonlinear stacking without aggressive OOF selection | overfits the stacked OOFs |
| Hill climbing without a CV guard | CV climbs, LB doesn't follow |
| Selecting the ensemble on public LB | HR-5 violation |
