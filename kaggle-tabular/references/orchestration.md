# Orchestration: Multi-Agent Roles, Prompts & the CV–LB Contract

How to run this with parallel agents across Claude Code / Codex / OpenCode, and how to select final
submissions. Coordination happens **through the filesystem** (the OOF ledger), not through chat —
this is what makes parallelism safe and the ensembler decoupled.

## Role map (roles, not just models — and one adversarial role)

| Role | Responsibility | Volume | Suggested tier |
|---|---|---|---|
| **Setup agent** | env, ingestion, `COMPETITION.md` | one-shot | free / local |
| **Validation Guardian** | metric proof, fold freezing, adversarial validation, **and auditing every experiment against the Hard Rules before it is `kept`** | continuous, high-judgment | paid / standing critic |
| **EDA agent** | distribution & shift checks, leakage suspects, FE hypotheses | medium | free |
| **Baseline agents** (×N families) | parallel diverse baselines via `run_experiment` | high, parallel | free / high-throughput |
| **FE-explorer agents** (×N families) | per-family feature search, ledger + NOTES discipline | very high, parallel | free / high-throughput |
| **Tuning agents** | small, careful Optuna per family | low, high-judgment | paid |
| **Ensembler** | hill climbing, stacking, distillation over the ledger | medium | paid |
| **Summarizer** | "what's working" reports from the ledger | medium | free |

This mirrors a Generator/Evaluator split with an adversarial critic: the **Validation Guardian** is
the critic whose whole job is to stop the CV score from being hacked via leakage. Never let the agent
that produced an experiment also be the one that certifies it `kept`.

## Filesystem coordination (why parallelism is safe)

- Agents communicate by **appending** to `experiments/ledger.parquet` and writing `oof/` + `preds/`
  arrays. Append-only → no write collisions between parallel agents.
- The ensembler reads the ledger and loads OOF arrays; it never needs to know how a model was built.
- The summarizer answers "what's working?" by querying the ledger.
- Every model is built through `src/models/base.py::run_experiment(...)`, which enforces HR-2
  (frozen folds) and HR-4 (artifacts + ledger row) by construction.

## Drop-in agent prompt patterns

- **EDA:** "Write and run EDA on `train.csv`/`test.csv`. Report row/col counts, target format and
  balance, per-feature train-vs-test distribution shift, categorical cardinalities, and leakage
  suspects. Output `EDA_FINDINGS.md` with 3–5 concrete FE hypotheses. Do not model yet."
- **Baseline:** "Build a k-fold `<MODEL>` baseline via `run_experiment` using the frozen folds and
  `src/metric.py`. Save OOF/preds and append a ledger row. Print per-fold and overall CV. No feature
  engineering."
- **FE:** "Take the current best `<MODEL>` config. Add feature group `<hypothesis>` as a fold-safe
  transform (fit on each fold's train rows only). Re-run via `run_experiment`, save new OOF/preds and
  a ledger row, and add a line to `experiments/<MODEL>/NOTES.md` with ΔCV vs cv_std and keep/drop."
- **Combine into one stronger model:** "Read the top `kept` experiments' configs/feature groups and
  write one new `<MODEL>` combining the surviving ideas; run via `run_experiment`."
- **Distill:** "Train a new single GBDT/NN via knowledge distillation from all `kept` OOF/test preds
  as soft targets; run via `run_experiment`."
- **Ensemble:** "Read `experiments/ledger.parquet`, load OOF arrays for all `kept` experiments, and
  find the best blend via hill climbing, then Ridge/Logistic and GBDT stackers on the same folds.
  Report OOF for each; save the winning ensemble spec."
- **Guardian audit:** "Audit experiment `<exp_id>` against the Hard Rules (folds match
  `data/folds.parquet`; no preprocessing fit on validation rows; all features inference-time
  available; seeds/versions logged and reproducible; artifacts present; CV not implausibly good).
  Mark `kept` or `rejected:<reason>` in the ledger."

## The CV–LB contract (final submission selection)

- Log `lb_public` next to `cv_score` for every submitted experiment (the ledger has the column).
- Track CV vs public-LB. A stable monotone relationship → trust CV fully. A weak/no relationship →
  your CV or the public split is unrepresentative: re-examine folds and adversarial validation, and
  trust CV *even harder* while treating the public LB as noise.
- **Spend submissions as a budget** (respect the daily limit): confirm the metric once (HR-3), then
  periodically check that CV gains track LB. Do **not** probe the LB to choose features/blends
  (HR-5).
- **Final picks (choose by CV):** a common safe pair is (1) the best-CV ensemble and (2) a more
  conservative / lower-variance blend, to hedge a private-LB shake-up.

## Throughput note

The biggest lever is the number of high-quality experiments. Keep FE-loop models fast (LightGBM,
GPU backends if scaffolded with `--gpu`), run baselines and FE-explorers in parallel, and let the
append-only ledger absorb the concurrency.
