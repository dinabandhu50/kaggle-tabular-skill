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
| **Kaggle Intel agent** | periodic web search of the competition's Overview/Discussion/Code pages; writes `localdev/external/*.md` | low, recurring | free |

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

## Daily experiment planning cadence

Don't front-load a full competition plan — each batch's results tell you what the next batch should
test. Work in small, verifiable batches, one per session/day: plan the batch, spec each experiment,
execute (in parallel where possible), verify, then let the results pick the next batch — the same
plan-before-code discipline the rest of this skill applies to code, applied here to experiments.
(If you're running in Claude Code with the `superpowers` plugin installed, this cadence maps
directly onto its `brainstorming` → `writing-plans` → `subagent-driven-development` skills — use
them if available, but nothing here depends on it.)

1. **Review state.** Read `PROGRESS.md`, the ledger's recent rows, and any new notes in
   `localdev/external/` (see "External intel gathering" below) to see what's already been tried and
   what's still open.
2. **Pick a small batch of hypotheses.** 2–4 ideas that fit the *current* phase (see
   `workflow-phases.md`) and are cheap enough to evaluate within the session — not a backlog for the
   whole competition. Prefer ideas with a stated reason to fail fast on bad ones.
3. **Spec each one before touching code.** FE ideas get `experiments/<model>/specs/NNN_<slug>.md`
   (see `feature-engineering.md` → "Artifact discipline"); tuning/ensembling ideas get an equivalent
   short spec (what's being tried, expected effect, how it'll be judged) saved next to their output.
4. **Dispatch.** Where experiments are independent (different model families, different FE groups),
   run one subagent per experiment in parallel rather than serially — see the role map above.
5. **Verify.** Each experiment must go through `run_experiment` (OOF + preds + ledger row, HR-4) and
   the Validation Guardian audit (`hard-rules.md`) before being marked `kept`. Compare ΔCV to
   `cv_std`, not to zero.
6. **Decide the next batch from evidence.** Update `PROGRESS.md`, commit (see `SKILL.md` →
   "Commit at meaningful steps"), and let what worked/failed — plus anything new in
   `localdev/external/` — drive the next batch's hypotheses. Don't plan batch N+2 before batch N's
   results are in.

## External intel gathering

Other competitors' write-ups routinely shortcut hours of independent discovery — treat the
competition's own Discussion and Code (public notebooks) tabs as a data source, not just the raw
train/test files.

- **At Phase 0**, ask the user for the competition's **Overview**, **Discussion**, and **Code**
  URLs and record them in `COMPETITION.md`. If the user doesn't have them handy, ask again before
  the first FE batch — it's worth the pause.
- **Periodically** (a reasonable default is once per session/day, or whenever a batch's results are
  surprising and external context might explain why) search the web for and read those pages for
  new ideas, reported pitfalls, and public LB chatter — use whatever browsing/search capability the
  harness provides, or ask the user to paste key threads if it has none. Synthesize findings into a
  dated or slug-named file under `localdev/external/` (e.g.
  `localdev/external/2026-07-03-discussion-notes.md`) — don't just dump raw search results; extract
  what's actionable (a feature idea, a CV pitfall, a model that's working for others) and note
  whether it's been tried yet.
- **Feed it into planning**, not straight into code: a promising idea from `localdev/external/`
  becomes a hypothesis in the next batch (step 2 above) with its own spec, same as an internally
  generated idea — it still has to earn its keep against `cv_std` and pass the Guardian audit. Public
  notebooks' reported CV/LB numbers are not a substitute for your own.
- `localdev/external/` is scratch research for *this* competition's repo — it has no access to and
  makes no reference to the kaggle-tabular skill's own source repo. If a technique proves broadly
  useful beyond this one competition, that's a call for the human running the competition to carry
  forward manually (e.g. into their own cross-competition notes), not something this repo's agents
  can or should do on their own.

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

The biggest lever is the number of high-quality experiments. Keep FE-loop models fast (LightGBM;
GPU is used automatically when present via `src/device.py::has_gpu()` — no flag needed), run
baselines and FE-explorers in parallel across subagents (one per model family), and let the
append-only ledger absorb the concurrency.

## Progress transparency for parallel agents

With several agents training concurrently, make progress legible without reading full logs:

- `run_experiment` already shows a per-fold `tqdm` bar with running score — don't silence it.
- Each agent prints one line when an experiment starts (`exp_id`, model family, fold count) and one
  when it finishes (final CV, elapsed time), so a human tailing several panes can tell what's running
  and roughly how long it'll take without opening each log.
- Prefer this over a spinner with no information — the useful signal is *which* experiment, *how far
  along*, and *what score so far*.
