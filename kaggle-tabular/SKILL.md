---
name: kaggle-tabular
description: >-
  Grandmaster-grade workflow for solo Kaggle / structured-data (tabular) competitions, designed to
  be executed by AI coding agents. Use this skill whenever the user is working on a
  tabular/structured-data prediction task, a Kaggle (or DrivenData / Zindi / Numerai) competition,
  mentions a leaderboard, train/test CSVs, cross-validation,
  CV-LB gap, feature engineering, target encoding, GBDTs (XGBoost/LightGBM/CatBoost), stacking,
  hill-climbing, OOF predictions, or wants to scaffold/structure a competition repo — even if they
  don't say the word "Kaggle". It enforces validation-first discipline, a strict out-of-fold (OOF)
  artifact contract, leakage hard-rules, and a gated phase workflow that parallel subagents can
  follow. Trigger it for setting up a new competition, planning the modeling approach, building
  baselines, engineering features, or ensembling. Do NOT use for NLP/CV/audio competitions yet
  (tabular only) or for non-competition production ML where deployment concerns dominate.
---

# Kaggle Tabular Competition Workflow

Operating manual for solo structured-data competitions run with AI coding agents (Claude Code /
Codex / OpenCode). This file is the **router**: read it fully, then load the reference file for the
phase you're in. Phases are **gated** — do not advance until the current gate is met and logged.

Grounding: NVIDIA *Kaggle Grandmasters Playbook* (Deotte/Onodera/Titericz/Viel), NVIDIA
*GenAI-assisted Kaggle* case study (4-level stack, 150/850 models), Kaggle winning write-ups, and
2025–26 tabular benchmarks (TabArena / TabPFN-2.5).

## The two foundations (everything rests on these)

1. **Trustworthy local validation.** A CV score you can trust is worth more than any model. Build
   and *verify* the validation harness **before** feature engineering. If CV isn't trustworthy,
   every downstream decision is noise.
2. **Fast experiment throughput.** The count of *high-quality* experiments is the biggest lever.
   Make every experiment cheap to launch, cheap to evaluate, and self-logging.

The mechanism that makes both work with agents is **the OOF contract**: every experiment writes a CV
score, an out-of-fold prediction array, and a test prediction array to disk, plus one ledger row.
These artifacts are the verifiable reward signal and the reusable state that let many agents compose.
The CV score *is* the reward; the OOF array *is* the state.

## How to use this skill

- **New competition?** Run the scaffolder, then start the gated workflow at Phase 0:
  ```bash
  python scripts/scaffold_competition.py <comp-name> --dest <path>
  # add --gpu to default the templates to RAPIDS/cuDF/cuML instead of CPU
  ```
  It creates the repo structure and drops in working template code that already enforces the OOF
  contract and the leakage hard-rules (see "What the scaffold gives you" below).
- **Existing repo / mid-competition?** Skip scaffolding. Identify the current phase from
  `references/workflow-phases.md` and continue from its gate.
- **Always:** before accepting any experiment as "kept", run the Validation Guardian audit in
  `references/hard-rules.md`. Leakage is the silent CV killer.

## The phased loop (overview — full detail in references/workflow-phases.md)

| Phase | Goal | Gate to advance |
|---|---|---|
| 0 Setup | reproducible env + raw data + competition facts | env reproduces; `COMPETITION.md` states the CV hypothesis |
| 1 **Validation harness** | trustworthy CV, metric proven, folds frozen | metric reproduces a known LB point; `data/folds.parquet` written; adversarial-validation AUC logged |
| 2 Smart EDA | signal, shift, leakage traps | `EDA_FINDINGS.md` with shift list, leakage suspects, encoding plan, 3–5 FE hypotheses |
| 3 Diverse baselines | map the model landscape | ≥4 model families in the ledger with trustworthy CV; no impossibly-good CV |
| 4 **Feature engineering** | features that reliably lift CV (per family) | CV plateaus for the family (recent gains within fold-noise) |
| 5 Tuning (light) | easy GBDT gains without overfitting CV | tuned CV beats best untuned by > fold-noise, else keep simpler |
| 6 **Ensembling** | combine diverse strong models | ensemble OOF beats best single OOF by > fold-noise; spec saved |
| 7 Final-mile | squeeze + lock | seed-ensemble + full-data refit; final subs chosen **by CV** |

Bold phases are where competitions are won or lost. Most effort goes to **1, 4, and 6** — not tuning.

## ROI order (corrected — read this, it's the most common mistake)

Trustworthy CV → diverse baselines → **feature engineering** → *light* tuning → **serious
ensembling**. Hyperparameter tuning is *lower* ROI than people assume for GBDTs (good defaults +
early stopping capture most of it, and over-tuning overfits CV). Ensembling is *not* a 1–2%
afterthought — it's where the leaderboard is won (winning solutions routinely stack 100+ models).
Do not heavily tune before FE has plateaued.

## Hard Rules (non-negotiable — full text + audit in references/hard-rules.md)

- **HR-1** The validation fold is sacred: no target-aware or cross-row preprocessing (target/freq
  encoding, scalers, imputers, TF-IDF vocab, PCA, SMOTE, target-corr feature selection) may be fit
  on data that includes the validation fold. Fit inside the fold on train rows only.
- **HR-2** Fix the folds once (`data/folds.parquet`); every model, experiment, and stacking level
  reuses the identical split.
- **HR-3** Implement the competition metric exactly; prove it reproduces a known LB point before
  trusting any CV number.
- **HR-4** Every experiment saves OOF + test preds + a ledger row — including failures. No artifacts
  = it didn't happen.
- **HR-5** Never tune, select features, or pick the ensemble against the public LB. Decide on CV.
- **HR-6** Determinism: fix and log all seeds and library versions.
- **HR-7** No leakage features: every feature must be computable at inference time using only
  information available before the prediction moment.

## What the scaffold gives you

After `scaffold_competition.py`, the repo contains working template code (in `src/`) that *encodes*
the rules so agents can't easily violate them:

- `src/cv.py` — fold generation (stratified / group / time) + **adversarial validation**.
- `src/metric.py` — competition-metric registry + CV scorer (fill in the exact metric, HR-3).
- `src/ledger.py` — append-only experiment ledger (the agent-coordination layer).
- `src/models/base.py` — `run_experiment(...)` that loads the frozen folds, runs CV, saves
  `oof/` + `preds/`, and appends a ledger row. **Using this is how HR-2/HR-4 are enforced.**
- `src/ensemble.py` — hill climbing + stacking over the OOF ledger.
- `justfile`, `configs/`, `AGENTS.md`, `COMPETITION.md`.

Agents should build every model through `run_experiment(...)` rather than hand-rolling CV loops.

## Reference files — load on demand

- `references/hard-rules.md` — the leakage constitution, the Validation Guardian audit checklist,
  adversarial validation, and the anti-pattern list. **Read at Phase 1 and before accepting any
  experiment.**
- `references/workflow-phases.md` — detailed Phase 0–7: actions, gates, owner agent, what to save.
  **Read to execute any phase.**
- `references/feature-engineering.md` — high-yield FE families, recipes, and leak-free encoding.
  **Read in Phase 4.**
- `references/ensembling.md` — hill climbing, stacking depth discipline, distillation, pseudo-
  labeling, seed-ensembling, full-data refit. **Read in Phase 6–7.**
- `references/model-menu.md` — 2026 model selection (GBDTs vs TabPFN-2.5 vs AutoGluon vs diversity
  members) and when each wins. **Read in Phase 3.**
- `references/orchestration.md` — multi-agent role map (incl. the adversarial Validation Guardian),
  drop-in agent prompt patterns, filesystem-based coordination, and the CV–LB contract for final
  submission selection. **Read when planning parallel agents or selecting final submissions.**

## One-line summary

Trustworthy CV → diverse baselines → hypothesis-driven feature engineering (per family, OOF-logged)
→ light tuning → hill-climb + stack the diverse OOFs → seed-ensemble + full-data refit → submit by
CV. Every step emits OOF + test preds + a ledger row, audited by an adversarial Validation Guardian.
