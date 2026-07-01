# kaggle-tabular Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repackage the existing `kaggle-tabular/` skill + `learnings/refined-pipeline.md` knowledge into a reusable, cross-harness, installable plugin that mirrors the `superpowers` marketplace layout.

**Architecture:** One router skill (`kaggle-tabular`) with on-demand references, plus a scaffolder + template `src/` that *encodes* the leakage hard rules. The repo root becomes a Claude plugin marketplace; a cross-harness `install.sh` wires Codex/OpenCode via AGENTS.md pointers. Reference docs are upgraded with generalized recipes distilled from `learnings/refined-pipeline.md`; competition-specific numbers are kept only as cited evidence.

**Tech Stack:** Markdown skill files, Python 3 (stdlib + numpy/pandas/scikit-learn/lightgbm patterns already in the template), Bash installer, Claude plugin manifests (JSON).

## Global Constraints

- Plugin layout mirrors superpowers: marketplace manifest at repo root `.claude-plugin/marketplace.json`; plugin nested at `plugins/kaggle-tabular/` with `.claude-plugin/plugin.json`; skill at `plugins/kaggle-tabular/skills/kaggle-tabular/`.
- **No hardcoded competition constants in shipped files.** S6E2 feature names (e.g. `Thallium`, `Chest_pain_type`) and exact scores may appear ONLY as cited evidence inside `references/*.md`, never in `assets/template/src/` code or as defaults.
- Hard rules are invariant — no code/doc change may weaken HR-1 (no target-aware preprocessing touching the validation fold), HR-2 (frozen folds), HR-4 (OOF+preds+ledger artifacts), HR-5 (decide on CV).
- Template files use `{{COMP_NAME}}` placeholder; never hardcode a competition slug.
- `--gpu` works by *uncommenting* exact marker lines (`# device_type="gpu",  # uncomment if scaffolded with --gpu`) — preserve that exact commented form so `enable_gpu()` can find and flip it.
- Keep skill prose and the template `src/` that encodes it in sync.
- `learnings/` is research and is NOT part of the shipped plugin (lives outside `plugins/`).
- Every task ends with a `git commit`.

---

### Task 1: Restructure repo into plugin + marketplace skeleton

**Files:**
- Create: `.claude-plugin/marketplace.json`
- Create: `plugins/kaggle-tabular/.claude-plugin/plugin.json`
- Move: `kaggle-tabular/` → `plugins/kaggle-tabular/skills/kaggle-tabular/` (git mv, preserves history)

**Interfaces:**
- Produces: the installable marketplace + plugin manifests and the final skill path `plugins/kaggle-tabular/skills/kaggle-tabular/SKILL.md` that every later task edits.

- [ ] **Step 1: Move the skill into the plugin skills dir**

```bash
mkdir -p plugins/kaggle-tabular/skills
git mv kaggle-tabular plugins/kaggle-tabular/skills/kaggle-tabular
```

- [ ] **Step 2: Create the plugin manifest**

Create `plugins/kaggle-tabular/.claude-plugin/plugin.json`:

```json
{
  "name": "kaggle-tabular",
  "version": "0.1.0",
  "description": "Grandmaster-grade, validation-first workflow for solo tabular Kaggle competitions: leakage hard-rules, a gated 8-phase loop, an OOF artifact contract, a scaffolder + rule-encoding template, and multi-agent orchestration. Cross-harness (Claude Code / Codex / OpenCode).",
  "author": { "name": "Dinabandhu Behera" },
  "keywords": ["kaggle", "tabular", "cross-validation", "feature-engineering", "ensembling", "gbdt", "competition"]
}
```

- [ ] **Step 3: Create the marketplace manifest**

Create `.claude-plugin/marketplace.json`:

```json
{
  "name": "kaggle-tabular-marketplace",
  "owner": { "name": "Dinabandhu Behera" },
  "plugins": [
    {
      "name": "kaggle-tabular",
      "source": "./plugins/kaggle-tabular",
      "description": "Grandmaster-grade validation-first workflow for solo tabular Kaggle competitions."
    }
  ]
}
```

- [ ] **Step 4: Validate the manifests**

Run: `claude plugin validate plugins/kaggle-tabular`
Expected: validation passes (plugin.json well-formed, at least one skill discovered).
Fallback if `claude` CLI unavailable in the exec env: `python -c "import json; json.load(open('.claude-plugin/marketplace.json')); json.load(open('plugins/kaggle-tabular/.claude-plugin/plugin.json')); print('json OK')"`
Expected: `json OK`

- [ ] **Step 5: Verify the skill path resolves**

Run: `test -f plugins/kaggle-tabular/skills/kaggle-tabular/SKILL.md && echo OK`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(plugin): restructure repo into kaggle-tabular plugin + marketplace"
```

---

### Task 2: Upgrade hard-rules.md and workflow-phases.md

**Files:**
- Modify: `plugins/kaggle-tabular/skills/kaggle-tabular/references/hard-rules.md`
- Modify: `plugins/kaggle-tabular/skills/kaggle-tabular/references/workflow-phases.md`
- Source: `learnings/refined-pipeline.md` (lines 88–118 adversarial table; 505–519 CV–LB gap)

**Interfaces:**
- Produces: an adversarial-AUC interpretation table and CV–LB gap rule other references cite; enriched Phase-1/Phase-7 gates.

- [ ] **Step 1: Add the adversarial-AUC interpretation table to hard-rules.md**

In `hard-rules.md`, under the "## Adversarial validation" section (after the existing interpret list), insert this table verbatim:

```markdown
### Adversarial-AUC interpretation (action table)

| AUC | Meaning | Action |
|---|---|---|
| ≈ 0.500 | train/test drawn alike, no drift | random K-fold trustworthy; global statistics *may* be usable as features (carefully) |
| 0.510–0.550 | minor drift | in-fold encoding only; inspect the top shifted features |
| > 0.550 | significant shift | feature-level analysis required; prefer time/group folds; weight rows by test-likeness; external data risky |

> Evidence: all S6E2 top solutions ran adversarial validation (AUC ≈ 0.5017 ± 0.0013 — a near-perfect match), which is what licensed safely merging external target statistics.
```

- [ ] **Step 2: Add the CV–LB gap tracking rule to hard-rules.md**

Append a new section to `hard-rules.md`:

```markdown
## CV–LB gap tracking (reject overfitting experiments)

The CV↔public-LB relationship is a leakage/overfitting sensor, used within HR-5 (never *select* on the LB):

- Log `lb_public` next to `cv_score` for every submitted experiment (the ledger has the column).
- Track the gap `cv_score − lb_public`. A **stable** gap = healthy; a **widening** gap = the experiment is fitting split-specific noise → reject it even if CV improved.
- Choose final submissions from the range where the CV→LB slope was still positive, not the single highest CV.

> Evidence (S6E2, 4th place): experiments were rejected once `CV − Public LB` widened past its healthy band (~0.00190 vs a healthy ~0.00185). 1st place's chosen submission had *lower* CV than their best-CV submission but higher private LB — the CV–LB relation, not raw CV, made the call.
```

- [ ] **Step 3: Enrich the Phase-1 and Phase-7 gates in workflow-phases.md**

In `workflow-phases.md`, Phase 1 "Gate" line — confirm it already requires "metric reproduces LB to tolerance; folds persisted; adversarial-validation AUC logged" (it does). Add one bullet to Phase 1 Actions after the adversarial-validation step:

```markdown
  5. Record the adversarial-AUC band (see `hard-rules.md` → interpretation table) in `COMPETITION.md`; it drives the encoding strategy (global vs strictly in-fold).
```

In `workflow-phases.md`, Phase 7 "Actions" — replace the seed-ensembling bullet's parenthetical with the concrete refit recipe:

Find: `retrain on 100% of data once features/params are frozen;`
Replace with: `retrain on 100% of data at ~1.25× the average best_iteration across folds, averaged over ~20 seeds (see ensembling.md → full-data refit);`

- [ ] **Step 4: Verify no competition constants leaked into code (guard check)**

Run: `grep -rniE "thallium|chest_pain|cleveland" plugins/kaggle-tabular/skills/kaggle-tabular/assets || echo "clean: no comp constants in template"`
Expected: `clean: no comp constants in template`

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs(refs): add adversarial-AUC table + CV-LB gap rule; enrich phase gates"
```

---

### Task 3: Upgrade feature-engineering.md and model-menu.md

**Files:**
- Modify: `plugins/kaggle-tabular/skills/kaggle-tabular/references/feature-engineering.md`
- Modify: `plugins/kaggle-tabular/skills/kaggle-tabular/references/model-menu.md`
- Source: `learnings/refined-pipeline.md` (lines 188–316 FE tiers/anti-patterns; 141–177 linearity probe / low-depth / RealMLP)

**Interfaces:**
- Produces: the Tier-1/Tier-2 FE recipe catalog and the "LogReg-first linearity probe" heuristic referenced by Phase 3–4.

- [ ] **Step 1: Append the Tier-1 / Tier-2 recipe catalog to feature-engineering.md**

Port the FE recipes from `learnings/refined-pipeline.md` lines 200–298 into a new section of `feature-engineering.md`, **generalizing all feature names** (use `cat_col`/`num_col`/`CATS`/`NUMS`, never `Thallium`/`Age`). Structure:

```markdown
## Recipe catalog (generalized; obey HR-1 + HR-7)

### Tier 1 — universal high ROI
1. **In-fold target encoding** — per fold, mean target per category on TRAIN rows only, smoothed toward the global mean; apply to val + test. (Encoded in `src/features.py::OOFTargetEncoder`.)
2. **ALL_CATS** — cast every column to string/category so trees split on levels; a different split geometry → diverse OOFs. (`src/features.py::all_cats`.)
3. **External-data target statistics** — for synthetic/Playground comps, merge `groupby(cat)[target].agg([...])` computed on the *independent* original dataset (no fold leakage — labels are external).
4. **Frequency / count encoding** — `df[col].map(df[col].value_counts())`, fit on train rows only. (`src/features.py::frequency_encode`.)

### Tier 2 — diversity generators (ensemble value even if flat on single-model CV)
5. **Quantile binning** — `pd.qcut(num, q=10, labels=False, duplicates='drop')`. (`src/features.py::quantile_bins`.)
6. **Digit extraction** — units/tens digits of quantized numerics; exposes generator structure in synthetic data. (`src/features.py::digit_features`.)
7. **Categorical n-gram interactions** — concat pairs/triples of categoricals, then in-fold target-encode. Select by linear-model coefficient magnitude. (`src/features.py::categorical_interactions`.)
8. **Periodic / sinusoidal encoding (NN only)** — `sin/cos(2π·num/p)`; helps MLPs beat spectral bias; GBDTs do NOT benefit.
9. **Domain composites (linear/NN)** — physically-meaningful ratios/products; GBDTs find these internally.

> Principle (1st place): the ensemble rewards *different mistakes*. A feature set that is flat on single-model CV can still earn its place by decorrelating OOFs.
```

- [ ] **Step 2: Append the FE anti-pattern table to feature-engineering.md**

Port `learnings/refined-pipeline.md` lines 300–309 as:

```markdown
## FE anti-patterns (the Guardian rejects these)

| Anti-pattern | Why it fails |
|---|---|
| 800+ polynomial interactions | collinearity noise crashes CV |
| Appending original-dataset rows as NN training data | domain shift confuses the network |
| Target encoding fit outside the fold | inflated CV that doesn't generalize (HR-1) |
| Blind FE with no hypothesis | signal-to-noise degradation |
| FE before CV is trustworthy | building on sand |
```

- [ ] **Step 3: Add the linearity-probe + low-depth heuristics to model-menu.md**

Append to `model-menu.md`:

```markdown
## Signal-shape heuristics (run before heavy FE)

- **Run Logistic/Linear + one-hot FIRST as a linearity probe.** If it is competitive with the GBDT baseline, the signal is near-linear → prefer **shallow GBDTs (depth 2–3 / few leaves)**, favor diversity over depth, and go easy on aggressive FE. If it is far behind → non-linear signal dominates → deeper trees + aggressive interactions/FE. (`src/models/logreg.py` doubles as this probe.)
- **Best neural member:** RealMLP (`pytabkit`, `n_ens≈8`, metric `1-auc_ovr`) is the strongest single neural model on tabular and decorrelates the GBDT blend; add it for diversity once GBDTs plateau.

> Evidence (S6E2): LogReg+OHE reached 0.95550 CV — near the GBDT baseline — which correctly signaled a near-linear problem and made depth-2 stumps the right default.
```

- [ ] **Step 4: Verify recipes reference the code helpers that Task 6 will create (name consistency)**

Run: `grep -oE "src/features\.py::[a-zA-Z_]+" plugins/kaggle-tabular/skills/kaggle-tabular/references/feature-engineering.md | sort -u`
Expected: lists `OOFTargetEncoder`, `all_cats`, `categorical_interactions`, `digit_features`, `frequency_encode`, `quantile_bins` — the exact names Task 6 implements.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs(refs): add FE recipe catalog + anti-patterns + linearity-probe heuristic"
```

---

### Task 4: Upgrade ensembling.md and orchestration.md

**Files:**
- Modify: `plugins/kaggle-tabular/skills/kaggle-tabular/references/ensembling.md`
- Modify: `plugins/kaggle-tabular/skills/kaggle-tabular/references/orchestration.md`
- Source: `learnings/refined-pipeline.md` (lines 357–520)

**Interfaces:**
- Produces: the ensembling escalation ladder (dedup → multi-seed avg → subset select → rank → logit hill-climb) and the full-data refit recipe that `src/ensemble.py` (Task 7) and `src/finalize.py` (Task 8) encode.

- [ ] **Step 1: Append the OOF-library + selection ladder to ensembling.md**

Port `learnings/refined-pipeline.md` lines 365–455 (generalized) as:

```markdown
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
```

- [ ] **Step 2: Append the full-data refit + final-decision recipe to ensembling.md**

Port `learnings/refined-pipeline.md` lines 474–519 as:

```markdown
## Full-data refit + final decision (Phase 7)

- Record avg `best_iteration` across folds; retrain on 100% of train at `int(avg_best_iter * 1.25)` rounds, averaged over ~20 seeds. More data → afford slightly more rounds. (`src/finalize.py::full_data_refit`.)
- **Decide by the CV–LB relation, not the single best CV** (see hard-rules.md → CV–LB gap). Submit intermediate ensembles throughout to map the slope; pick from the still-positive-slope range.
- Lock **two** submissions: the best-trustworthy-CV blend and a conservative/lower-variance fallback to hedge a shake-up.
```

- [ ] **Step 3: Add the ensembling anti-patterns table**

Port `learnings/refined-pipeline.md` lines 457–465:

```markdown
## Ensembling anti-patterns

| Anti-pattern | Why it fails |
|---|---|
| Simple average of all OOFs | dilutes signal with redundant members |
| One model given 65%+ weight | overfits OOF, collapses on LB |
| Nonlinear stacking without aggressive OOF selection | overfits the stacked OOFs |
| Hill climbing without a CV guard | CV climbs, LB doesn't follow |
| Selecting the ensemble on public LB | HR-5 violation |
```

- [ ] **Step 4: Cross-check code-helper names against Tasks 7–8**

Run: `grep -oE "src/(ensemble|finalize)\.py::[a-zA-Z_]+" plugins/kaggle-tabular/skills/kaggle-tabular/references/ensembling.md | sort -u`
Expected: `average_seeds`, `dedup_oofs`, `hill_climb_logit`, `rank_transform` (ensemble) and `full_data_refit` (finalize) — the exact names Tasks 7–8 implement.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs(refs): add ensembling ladder, logit hill-climb, full-data refit recipes"
```

---

### Task 5: Add references/what-works.md

**Files:**
- Create: `plugins/kaggle-tabular/skills/kaggle-tabular/references/what-works.md`
- Modify: `plugins/kaggle-tabular/skills/kaggle-tabular/SKILL.md` (add to reference index)
- Source: `learnings/refined-pipeline.md` (lines 529–564)

**Interfaces:**
- Produces: a single loadable "distilled memory" reference; SKILL.md router lists it.

- [ ] **Step 1: Create what-works.md**

Create the file with two tables ported from `learnings/refined-pipeline.md` lines 531–545 (cross-cutting patterns) and 551–564 (what does NOT work), generalized, each row keeping its solution-count evidence. Header:

```markdown
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
```

- [ ] **Step 2: Register what-works.md in the SKILL.md reference index**

In `SKILL.md`, under "## Reference files — load on demand", add this bullet after the `orchestration.md` entry:

```markdown
- `references/what-works.md` — evidence-backed cross-cutting patterns and the "what does NOT work"
  table. **Load for a fast prior before committing effort in any phase.**
```

- [ ] **Step 3: Verify the router lists every reference file**

Run: `for f in $(ls plugins/kaggle-tabular/skills/kaggle-tabular/references/*.md | xargs -n1 basename); do grep -q "$f" plugins/kaggle-tabular/skills/kaggle-tabular/SKILL.md && echo "listed: $f" || echo "MISSING: $f"; done`
Expected: every reference prints `listed:` (no `MISSING:`).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs(refs): add what-works.md distilled-evidence reference + index it"
```

---

### Task 6: Add template src/features.py (leak-safe FE toolkit)

**Files:**
- Create: `plugins/kaggle-tabular/skills/kaggle-tabular/assets/template/src/features.py`

**Interfaces:**
- Produces: `OOFTargetEncoder`, `frequency_encode`, `all_cats`, `quantile_bins`, `digit_features`, `categorical_interactions` — all called *inside* `fit_fold` (HR-1). Consumed by model wrappers (Task 9) and cited by feature-engineering.md (Task 3).

- [ ] **Step 1: Write features.py**

Create the file exactly:

```python
"""Leak-safe feature helpers (HR-1).

Every function here is designed to be called INSIDE a model's `fit_fold`, fit on the fold's
TRAINING rows only, then applied to validation and test. Fitting any of these on full data (or on
data that includes the validation fold) is an HR-1 violation and inflates CV.

Competition-agnostic: pass the column lists explicitly; nothing here hardcodes a dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class OOFTargetEncoder:
    """Smoothed target encoding, fit on training rows only.

    Usage inside fit_fold:
        te = OOFTargetEncoder(cols, smoothing=20.0).fit(X_tr, y_tr)
        X_tr = te.transform(X_tr); X_val = te.transform(X_val); X_test = te.transform(X_test)
    """

    def __init__(self, cols: list[str], smoothing: float = 20.0, suffix: str = "_te"):
        self.cols = cols
        self.smoothing = smoothing
        self.suffix = suffix
        self.global_mean_: float = 0.0
        self.maps_: dict[str, pd.Series] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "OOFTargetEncoder":
        y = np.asarray(y, dtype=float)
        self.global_mean_ = float(y.mean())
        df = X[self.cols].copy()
        df["__y__"] = y
        for c in self.cols:
            stats = df.groupby(c)["__y__"].agg(["mean", "count"])
            smooth = (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_) / (
                stats["count"] + self.smoothing
            )
            self.maps_[c] = smooth
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.cols:
            X[c + self.suffix] = X[c].map(self.maps_[c]).astype(float).fillna(self.global_mean_)
        return X


def frequency_encode(X_tr: pd.DataFrame, X_others: list[pd.DataFrame], cols: list[str],
                     suffix: str = "_freq") -> list[pd.DataFrame]:
    """Map each category to its TRAIN-row frequency. Returns transformed [X_tr, *X_others]."""
    frames = [X_tr] + list(X_others)
    frames = [f.copy() for f in frames]
    for c in cols:
        freq = X_tr[c].value_counts()
        for f in frames:
            f[c + suffix] = f[c].map(freq).fillna(0).astype(float)
    return frames


def all_cats(X: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Cast columns (default: all) to pandas 'category' so trees split on levels."""
    X = X.copy()
    cols = cols or list(X.columns)
    for c in cols:
        X[c] = X[c].astype(str).astype("category")
    return X


def quantile_bins(X_tr: pd.DataFrame, X_others: list[pd.DataFrame], cols: list[str], q: int = 10,
                  suffix: str = "_qbin") -> list[pd.DataFrame]:
    """Percentile bins with edges learned on TRAIN rows only. Returns [X_tr, *X_others]."""
    frames = [f.copy() for f in ([X_tr] + list(X_others))]
    for c in cols:
        _, edges = pd.qcut(X_tr[c], q=q, retbins=True, duplicates="drop", labels=False)
        for f in frames:
            f[c + suffix] = pd.cut(f[c], bins=edges, labels=False, include_lowest=True)
            f[c + suffix] = f[c + suffix].fillna(-1).astype(int)
    return frames


def digit_features(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Extract units/tens digits of integer-valued numerics (exposes generator structure)."""
    X = X.copy()
    for c in cols:
        v = X[c].fillna(0).astype(int)
        X[c + "_units"] = v % 10
        X[c + "_tens"] = (v // 10) % 10
    return X


def categorical_interactions(X: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Concatenate categorical pairs into new categorical columns (then target-encode in-fold)."""
    X = X.copy()
    for a, b in pairs:
        X[f"{a}_{b}"] = X[a].astype(str) + "_" + X[b].astype(str)
    return X
```

- [ ] **Step 2: Verify it imports and round-trips without leakage**

Run:
```bash
cd plugins/kaggle-tabular/skills/kaggle-tabular/assets/template && \
python -c "
import pandas as pd, numpy as np
import src.features as F
X = pd.DataFrame({'a':['x','y','x','y','x'],'n':[10,23,45,67,89]})
y = np.array([1,0,1,0,1])
te = F.OOFTargetEncoder(['a']).fit(X, y)
out = te.transform(X)
assert 'a_te' in out and out['a_te'].notna().all()
assert F.all_cats(X)['a'].dtype.name == 'category'
assert 'n_units' in F.digit_features(X, ['n'])
tr, = F.frequency_encode(X, [], ['a']); assert 'a_freq' in tr
print('features.py OK')
"
```
Expected: `features.py OK`

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat(template): add leak-safe features.py (in-fold TE/freq/cats/bins/digits/interactions)"
```

---

### Task 7: Upgrade template src/ensemble.py

**Files:**
- Modify: `plugins/kaggle-tabular/skills/kaggle-tabular/assets/template/src/ensemble.py`

**Interfaces:**
- Consumes: existing `competition_metric`, `is_improvement`, `GREATER_IS_BETTER` from `metric.py`.
- Produces: `rank_transform`, `dedup_oofs`, `average_seeds`, `hill_climb_logit` (added alongside existing `hill_climb`, `stack`, `load_oof_matrix`, `save_spec`).

- [ ] **Step 1: Append the new functions to ensemble.py**

Add these functions to the end of `src/ensemble.py` (keep all existing functions unchanged):

```python
from scipy.special import expit, logit  # add near the top imports
from scipy.stats import rankdata


def rank_transform(a: np.ndarray) -> np.ndarray:
    """Column-wise rank normalization to [0,1]; erases cross-family calibration differences."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return rankdata(a) / len(a)
    return np.column_stack([rankdata(a[:, j]) / a.shape[0] for j in range(a.shape[1])])


def dedup_oofs(ids: list[str], oof: np.ndarray, test: np.ndarray, *, thresh: float = 0.9999):
    """Drop one member from each pair with Pearson correlation > thresh. Returns (ids, oof, test)."""
    keep, corr = [], np.corrcoef(oof.T)
    for j in range(oof.shape[1]):
        if all(abs(corr[j, k]) <= thresh for k in keep):
            keep.append(j)
    return [ids[j] for j in keep], oof[:, keep], test[:, keep]


def average_seeds(oofs: list[np.ndarray], tests: list[np.ndarray]):
    """Average a group of same-config, different-seed OOF/test arrays into one cleaner member."""
    return np.mean(oofs, axis=0), np.mean(tests, axis=0)


def hill_climb_logit(oof: np.ndarray, test: np.ndarray, y: np.ndarray, *,
                     n_iter: int = 1000, tol: float = 1e-7, allow_negative: bool = True):
    """Greedy blend in logit space; supports negative weights (subtract correlated noise).

    Returns (weights[n_models], oof_blend_prob, test_blend_prob). Blends logits, outputs probs.
    """
    eps = 1e-7
    zо = logit(np.clip(oof, eps, 1 - eps))
    zt = logit(np.clip(test, eps, 1 - eps))
    n = oof.shape[1]
    w = np.zeros(n)
    steps = [0.05, -0.05] if allow_negative else [0.05]
    best = -np.inf if GREATER_IS_BETTER else np.inf
    for _ in range(n_iter):
        improved = False
        for j in range(n):
            for s in steps:
                w2 = w.copy(); w2[j] += s
                score = competition_metric(y, expit(zо @ w2))
                if is_improvement(score, best) and abs(score - best) > tol:
                    w, best, improved = w2, score, True
        if not improved:
            break
    return w, expit(zо @ w), expit(zt @ w)
```

Note: the two `zо` identifiers above use a lookalike char by mistake — write them as plain `z_oof`. Use `z_oof` and `z_test` as the variable names throughout this function.

- [ ] **Step 2: Rewrite the function with clean identifiers**

Ensure the committed `hill_climb_logit` uses `z_oof` / `z_test` (not any non-ASCII names). Final body:

```python
def hill_climb_logit(oof: np.ndarray, test: np.ndarray, y: np.ndarray, *,
                     n_iter: int = 1000, tol: float = 1e-7, allow_negative: bool = True):
    eps = 1e-7
    z_oof = logit(np.clip(oof, eps, 1 - eps))
    z_test = logit(np.clip(test, eps, 1 - eps))
    n = oof.shape[1]
    w = np.zeros(n)
    steps = [0.05, -0.05] if allow_negative else [0.05]
    best = -np.inf if GREATER_IS_BETTER else np.inf
    for _ in range(n_iter):
        improved = False
        for j in range(n):
            for s in steps:
                w2 = w.copy(); w2[j] += s
                score = competition_metric(y, expit(z_oof @ w2))
                if is_improvement(score, best) and abs(score - best) > tol:
                    w, best, improved = w2, score, True
        if not improved:
            break
    return w, expit(z_oof @ w), expit(z_test @ w)
```

- [ ] **Step 3: Verify imports resolve and functions run on synthetic OOFs**

Run:
```bash
cd plugins/kaggle-tabular/skills/kaggle-tabular/assets/template && \
python -c "
import numpy as np
# metric.py must define competition_metric to run; stub it for the smoke test:
import src.metric as M
from sklearn.metrics import roc_auc_score
M.competition_metric = lambda yt, yp: roc_auc_score(yt, yp)
import importlib, src.ensemble as E; importlib.reload(E)
rng = np.random.default_rng(0)
y = rng.integers(0,2,500)
oof = np.clip(np.column_stack([y*0.6+rng.random(500)*0.4, y*0.5+rng.random(500)*0.5, rng.random(500)]),1e-6,1-1e-6)
test = np.clip(rng.random((200,3)),1e-6,1-1e-6)
ids,o,t = E.dedup_oofs(['a','b','c'], oof, test)
assert E.rank_transform(oof).shape == oof.shape
w,ob,tb = E.hill_climb_logit(oof, test, y, n_iter=50)
assert ob.shape==(500,) and tb.shape==(200,)
print('ensemble.py OK')
"
```
Expected: `ensemble.py OK` (grep first to confirm no non-ASCII identifier survived: `grep -nP "[^\x00-\x7F]" src/ensemble.py || echo "ascii clean"` → `ascii clean`).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(template): add rank/dedup/seed-avg/logit-hill-climb to ensemble.py"
```

---

### Task 8: Add template src/finalize.py (full-data refit)

**Files:**
- Create: `plugins/kaggle-tabular/skills/kaggle-tabular/assets/template/src/finalize.py`

**Interfaces:**
- Produces: `full_data_refit(fit_full, best_iters, X, y, X_test, *, mult=1.25, seeds=range(20))`.
- Consumes: a `fit_full(X, y, X_test, n_rounds, seed) -> test_pred` callable supplied per model family.

- [ ] **Step 1: Write finalize.py**

Create the file exactly:

```python
"""Phase-7 full-data refit (no validation fold at the very end).

Once features and params are frozen, retrain on 100% of train at ~1.25x the average per-fold
best_iteration, averaged over many seeds. More data -> examples seen proportionally fewer times ->
afford slightly more rounds. Evidence: 1st-place S6E2 beat K-fold averaging with 1.25x + 20 seeds.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

# fit_full(X, y, X_test, n_rounds, seed) -> test_pred (np.ndarray, shape [n_test])
FitFull = Callable[..., np.ndarray]


def full_data_refit(fit_full: FitFull, best_iters: Iterable[int], X, y, X_test, *,
                    mult: float = 1.25, seeds: Iterable[int] = range(20)) -> np.ndarray:
    """Average test predictions over seeds from a full-data refit at mult x avg best_iteration."""
    best_iters = list(best_iters)
    n_rounds = int(round(float(np.mean(best_iters)) * mult)) if best_iters else 0
    seeds = list(seeds)
    preds = np.zeros(len(X_test), dtype=float)
    for s in seeds:
        preds += np.asarray(fit_full(X, y, X_test, n_rounds, s), dtype=float) / len(seeds)
    return preds
```

- [ ] **Step 2: Verify import + averaging math**

Run:
```bash
cd plugins/kaggle-tabular/skills/kaggle-tabular/assets/template && \
python -c "
import numpy as np, src.finalize as Fz
calls = []
def fit_full(X,y,Xt,n,seed):
    calls.append((n,seed)); return np.full(len(Xt), float(seed))
Xt = list(range(4))
out = Fz.full_data_refit(fit_full, [100,120,140], None, None, Xt, seeds=range(4))
assert calls[0][0] == int(round(120*1.25))   # 150
assert np.allclose(out, np.mean(range(4)))    # avg over seeds 0..3 = 1.5
print('finalize.py OK')
"
```
Expected: `finalize.py OK`

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat(template): add finalize.py full-data refit (1.25x best_iter, N seeds)"
```

---

### Task 9: Add model wrappers xgb.py, cat.py, logreg.py

**Files:**
- Create: `plugins/kaggle-tabular/skills/kaggle-tabular/assets/template/src/models/xgb.py`
- Create: `plugins/kaggle-tabular/skills/kaggle-tabular/assets/template/src/models/cat.py`
- Create: `plugins/kaggle-tabular/skills/kaggle-tabular/assets/template/src/models/logreg.py`

**Interfaces:**
- Produces: `make_fit_fold(...) -> fit_fold` for each family, matching `lgbm.py`'s contract `fit_fold(X_tr, y_tr, X_val, X_test, fold, seed) -> (val_pred, test_pred, model)` consumed by `run_experiment`.

- [ ] **Step 1: Write xgb.py**

```python
"""XGBoost wrapper. Same fit_fold contract as lgbm.py. HR-1: any target-aware transform is fit
inside fit_fold on X_tr only (see src/features.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_fit_fold(params: dict | None = None, num_boost_round: int = 2000,
                  early_stopping: int = 100, task: str = "classification"):
    import xgboost as xgb

    default = dict(
        max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.0, reg_lambda=1.0,
        objective="binary:logistic" if task == "classification" else "reg:squarederror",
        eval_metric="auc" if task == "classification" else "rmse",
        tree_method="hist",
        # device="cuda",  # uncomment if scaffolded with --gpu
    )
    params = {**default, **(params or {})}

    def fit_fold(X_tr: pd.DataFrame, y_tr: np.ndarray, X_val: pd.DataFrame,
                 X_test: pd.DataFrame, fold: int, seed: int):
        dtr = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dval = xgb.DMatrix(X_val, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, enable_categorical=True)
        model = xgb.train({**params, "seed": seed}, dtr, num_boost_round=num_boost_round)
        return model.predict(dval), model.predict(dtest), model

    return fit_fold
```

- [ ] **Step 2: Write cat.py**

```python
"""CatBoost wrapper. Native categorical handling (Ordered TS) — the categorical specialist.
Same fit_fold contract as lgbm.py."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_fit_fold(params: dict | None = None, iterations: int = 3000,
                  task: str = "classification"):
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    default = dict(
        depth=4, learning_rate=0.03, l2_leaf_reg=3.0, iterations=iterations,
        verbose=0, allow_writing_files=False,
        # task_type="GPU",  # uncomment if scaffolded with --gpu
    )
    params = {**default, **(params or {})}

    def fit_fold(X_tr: pd.DataFrame, y_tr: np.ndarray, X_val: pd.DataFrame,
                 X_test: pd.DataFrame, fold: int, seed: int):
        cat_cols = [c for c in X_tr.columns if str(X_tr[c].dtype) in ("object", "category")]
        Model = CatBoostClassifier if task == "classification" else CatBoostRegressor
        model = Model(**params, random_seed=seed)
        model.fit(Pool(X_tr, y_tr, cat_features=cat_cols))
        pv = model.predict_proba(X_val)[:, 1] if task == "classification" else model.predict(X_val)
        pt = model.predict_proba(X_test)[:, 1] if task == "classification" else model.predict(X_test)
        return pv, pt, model

    return fit_fold
```

- [ ] **Step 3: Write logreg.py (also the Phase-2 linearity probe)**

```python
"""Logistic/Linear + one-hot wrapper. Doubles as the Phase-2 LINEARITY PROBE: if competitive with
the GBDT baseline, the signal is near-linear -> prefer shallow trees + diversity. HR-1: the encoder
and scaler are fit on X_tr only, inside fit_fold."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_fit_fold(params: dict | None = None, task: str = "classification"):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    params = params or {}

    def fit_fold(X_tr: pd.DataFrame, y_tr: np.ndarray, X_val: pd.DataFrame,
                 X_test: pd.DataFrame, fold: int, seed: int):
        cat_cols = [c for c in X_tr.columns if str(X_tr[c].dtype) in ("object", "category")]
        num_cols = [c for c in X_tr.columns if c not in cat_cols]
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ])
        est = (LogisticRegression(max_iter=2000, C=params.get("C", 1.0))
               if task == "classification" else Ridge(alpha=params.get("alpha", 1.0)))
        pipe = Pipeline([("pre", pre), ("est", est)])
        pipe.fit(X_tr, y_tr)
        if task == "classification":
            return pipe.predict_proba(X_val)[:, 1], pipe.predict_proba(X_test)[:, 1], pipe
        return pipe.predict(X_val), pipe.predict(X_test), pipe

    return fit_fold
```

- [ ] **Step 4: Verify all three import and expose make_fit_fold**

Run:
```bash
cd plugins/kaggle-tabular/skills/kaggle-tabular/assets/template && \
python -c "
import importlib
for m in ['src.models.xgb','src.models.cat','src.models.logreg']:
    try:
        mod = importlib.import_module(m); assert hasattr(mod,'make_fit_fold'); print('ok', m)
    except ImportError as e:
        print('ok (lib not installed, syntax fine):', m, '->', e)
"
```
Expected: each of the three prints `ok ...` (either fully imported, or "lib not installed, syntax fine" if xgboost/catboost aren't in the exec env — a SyntaxError would instead raise and fail).

- [ ] **Step 5: Confirm GPU marker lines match the exact enable_gpu pattern**

Run: `grep -c 'uncomment if scaffolded with --gpu' plugins/kaggle-tabular/skills/kaggle-tabular/assets/template/src/models/xgb.py plugins/kaggle-tabular/skills/kaggle-tabular/assets/template/src/models/cat.py`
Expected: each file reports `1`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(template): add xgb/cat/logreg model wrappers (logreg doubles as linearity probe)"
```

---

### Task 10: Extend scaffolder enable_gpu + refresh SKILL.md scaffold notes

**Files:**
- Modify: `plugins/kaggle-tabular/skills/kaggle-tabular/scripts/scaffold_competition.py`
- Modify: `plugins/kaggle-tabular/skills/kaggle-tabular/SKILL.md`

**Interfaces:**
- Consumes: the new template files (auto-copied by the existing `shutil.copytree`, so no copy-list change is needed).
- Produces: `--gpu` now flips markers in `xgb.py` and `cat.py` too; SKILL.md's "what the scaffold gives you" mentions the new modules.

- [ ] **Step 1: Extend enable_gpu() to the new wrappers**

In `scaffold_competition.py`, replace the body of `enable_gpu(dest)` with a loop over marker/replacement pairs:

```python
def enable_gpu(dest: Path) -> None:
    """Uncomment GPU toggles in the model wrappers (exact commented markers -> live lines)."""
    swaps = {
        "src/models/lgbm.py": (
            '        # device_type="gpu",  # uncomment if scaffolded with --gpu',
            '        device_type="gpu",',
        ),
        "src/models/xgb.py": (
            '        # device="cuda",  # uncomment if scaffolded with --gpu',
            '        device="cuda",',
        ),
        "src/models/cat.py": (
            '        # task_type="GPU",  # uncomment if scaffolded with --gpu',
            '        task_type="GPU",',
        ),
    }
    for rel, (old, new) in swaps.items():
        p = dest / rel
        if p.exists():
            p.write_text(p.read_text().replace(old, new))
```

- [ ] **Step 2: Mention the new modules in SKILL.md**

In `SKILL.md` under "## What the scaffold gives you", add these bullets after the `src/ensemble.py` bullet:

```markdown
- `src/features.py` — leak-safe FE helpers (in-fold target/frequency encoding, ALL_CATS, quantile
  bins, digit features, categorical interactions) — call these INSIDE `fit_fold` (HR-1).
- `src/finalize.py` — Phase-7 full-data refit at 1.25× best_iteration over many seeds.
- `src/models/{lgbm,xgb,cat,logreg}.py` — four ready `fit_fold` wrappers; `logreg.py` is also the
  Phase-2 linearity probe.
```

- [ ] **Step 3: Verify scaffolder still runs end-to-end and GPU flip works**

Run:
```bash
cd plugins/kaggle-tabular/skills/kaggle-tabular && \
rm -rf /tmp/scaffold-test && python scripts/scaffold_competition.py demo-slug --dest /tmp/scaffold-test --gpu && \
grep -q 'device="cuda",' /tmp/scaffold-test/comp-demo-slug/src/models/xgb.py && \
grep -q 'task_type="GPU",' /tmp/scaffold-test/comp-demo-slug/src/models/cat.py && \
grep -q 'device_type="gpu",' /tmp/scaffold-test/comp-demo-slug/src/models/lgbm.py && \
test -f /tmp/scaffold-test/comp-demo-slug/src/features.py && \
test -f /tmp/scaffold-test/comp-demo-slug/src/finalize.py && \
grep -rn '{{COMP_NAME}}' /tmp/scaffold-test/comp-demo-slug && echo "LEFTOVER PLACEHOLDER" || echo "scaffold OK"
```
Expected: `scaffold OK` (all GPU markers flipped, new files present, no `{{COMP_NAME}}` left).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(scaffold): flip GPU markers in xgb/cat; document new src modules in SKILL.md"
```

---

### Task 11: Cross-harness install.sh

**Files:**
- Create: `plugins/kaggle-tabular/install.sh`

**Interfaces:**
- Produces: an idempotent installer wiring Claude Code (marketplace+plugin), Codex (`~/.codex` skill link + AGENTS.md pointer), and OpenCode (config pointer). Supports `--dry-run`.

- [ ] **Step 1: Write install.sh**

```bash
#!/usr/bin/env bash
# Cross-harness installer for the kaggle-tabular skill.
# Claude Code -> real plugin install. Codex / OpenCode -> AGENTS.md + config pointer to the skill.
set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PLUGIN_DIR/../.." && pwd)"
SKILL_DIR="$PLUGIN_DIR/skills/kaggle-tabular"
SKILL_MD="$SKILL_DIR/SKILL.md"

run() { if [[ $DRY_RUN == 1 ]]; then echo "DRY: $*"; else eval "$*"; fi; }

POINTER="$(cat <<EOF

# >>> kaggle-tabular skill >>>
# Grandmaster tabular-competition workflow. Read this before tabular/Kaggle work:
#   $SKILL_MD
# and load its references/ on demand. Enforce the hard rules (HR-1..HR-7).
# <<< kaggle-tabular skill <<<
EOF
)"

echo "== Claude Code =="
if command -v claude >/dev/null 2>&1; then
  run "claude plugin marketplace add '$REPO_ROOT'"
  run "claude plugin install kaggle-tabular@kaggle-tabular-marketplace"
else
  echo "  claude CLI not found — skipping (install manually: claude plugin marketplace add '$REPO_ROOT')"
fi

echo "== Codex =="
CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
if [[ -d "$CODEX_HOME" ]]; then
  run "mkdir -p '$CODEX_HOME/skills'"
  run "ln -sfn '$SKILL_DIR' '$CODEX_HOME/skills/kaggle-tabular'"
  if [[ $DRY_RUN == 1 ]]; then
    echo "DRY: append pointer to $CODEX_HOME/AGENTS.md"
  elif ! grep -q "kaggle-tabular skill" "$CODEX_HOME/AGENTS.md" 2>/dev/null; then
    printf '%s\n' "$POINTER" >> "$CODEX_HOME/AGENTS.md"
  fi
else
  echo "  ~/.codex not found — skipping"
fi

echo "== OpenCode =="
OC_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/opencode"
if [[ -d "$OC_DIR" ]]; then
  if [[ $DRY_RUN == 1 ]]; then
    echo "DRY: append pointer to $OC_DIR/AGENTS.md"
  elif ! grep -q "kaggle-tabular skill" "$OC_DIR/AGENTS.md" 2>/dev/null; then
    printf '%s\n' "$POINTER" >> "$OC_DIR/AGENTS.md"
  fi
else
  echo "  opencode config dir not found — skipping"
fi

echo "done."
```

- [ ] **Step 2: Make it executable and dry-run it**

Run:
```bash
chmod +x plugins/kaggle-tabular/install.sh && \
bash plugins/kaggle-tabular/install.sh --dry-run
```
Expected: prints `== Claude Code ==`, `== Codex ==`, `== OpenCode ==` sections with `DRY:` lines or "not found — skipping", and ends with `done.` — no error, nothing actually written.

- [ ] **Step 3: Lint with shellcheck if available (non-blocking)**

Run: `command -v shellcheck >/dev/null && shellcheck plugins/kaggle-tabular/install.sh || echo "shellcheck not present, skip"`
Expected: no errors, or `shellcheck not present, skip`.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(plugin): add cross-harness install.sh (Claude/Codex/OpenCode)"
```

---

### Task 12: USAGE.md, CLAUDE.md refresh, and final validation

**Files:**
- Create: `plugins/kaggle-tabular/USAGE.md`
- Modify: `CLAUDE.md` (repo-root: describe the new plugin layout)

**Interfaces:**
- Consumes: everything above. Produces: the human-facing step-by-step and an accurate repo guide.

- [ ] **Step 1: Write USAGE.md**

Create `plugins/kaggle-tabular/USAGE.md`:

```markdown
# Using the kaggle-tabular skill across Claude Code, Codex, and OpenCode

## One-time install (all harnesses)

```bash
bash plugins/kaggle-tabular/install.sh          # or: --dry-run to preview
```

Or manually, per harness:

- **Claude Code:** `claude plugin marketplace add <path-or-git-url-of-this-repo>` then
  `claude plugin install kaggle-tabular@kaggle-tabular-marketplace`. Restart; the skill is then
  available via the Skill tool and auto-triggers on tabular/Kaggle work.
- **Codex:** symlink `plugins/kaggle-tabular/skills/kaggle-tabular` into `~/.codex/skills/` and add a
  pointer to `~/.codex/AGENTS.md` (install.sh does both). Codex reads AGENTS.md at session start.
- **OpenCode:** add a pointer to `~/.config/opencode/AGENTS.md` referencing the skill's SKILL.md
  (install.sh does this). OpenCode reads AGENTS.md as project/global instructions.

## Per-competition flow (identical across harnesses)

```bash
# 1. Scaffold a competition repo (encodes the hard rules)
python plugins/kaggle-tabular/skills/kaggle-tabular/scripts/scaffold_competition.py \
  <competition-slug> --dest ~/kaggle [--gpu]
cd ~/kaggle/comp-<competition-slug>

# 2. Phase 0 — environment + data
uv sync
# add KAGGLE_KEY / WANDB_API_KEY to .envrc (gitignored), then:
just download

# 3. Phase 1 — validation harness FIRST (implement src/metric.py, then:)
just folds        # freezes data/folds.parquet (HR-2)
just adval        # adversarial validation AUC

# 4. Phases 2–7 — follow the gated workflow
just eda
just baseline model=lgbm     # + xgb, cat, logreg for diversity
just fe model=lgbm group=<hypothesis>
just tune model=lgbm
just ensemble
just submit spec=best
just audit exp=<id>          # Guardian: kept | rejected:<reason>
```

## How each harness discovers the workflow

- **Claude Code:** the installed plugin's skill auto-loads; open with the Skill tool.
- **Codex / OpenCode:** the AGENTS.md pointer (global) and the scaffolder's per-repo `AGENTS.md`
  route the agent into the skill. Inside a scaffolded `comp-<name>/`, the local `AGENTS.md` is enough
  even without the global install.

## The rules the skill enforces (never weaken)

HR-1 no target-aware preprocessing touching the validation fold · HR-2 frozen folds ·
HR-3 exact metric proven vs a known LB point · HR-4 every experiment saves OOF+preds+ledger row ·
HR-5 decide on CV, never public LB · HR-6 log seeds/versions · HR-7 inference-time-available features.
```

- [ ] **Step 2: Update repo-root CLAUDE.md to describe the new layout**

In `CLAUDE.md`, update the "What this repository is" section: replace the sentence that says the skill lives in `kaggle-tabular/` with the new path, and add a short "Plugin packaging" note.

Find: `- **`kaggle-tabular/`** — a self-contained, cross-harness`
Replace the leading path token `kaggle-tabular/` in that bullet with `plugins/kaggle-tabular/` and append this paragraph after the two-products list:

```markdown
### Plugin packaging

The repo root is a Claude plugin **marketplace** (`.claude-plugin/marketplace.json`) exposing one
plugin at `plugins/kaggle-tabular/` (`.claude-plugin/plugin.json`), whose skill lives at
`plugins/kaggle-tabular/skills/kaggle-tabular/`. Install with
`claude plugin marketplace add <repo>` → `claude plugin install kaggle-tabular`, or run
`plugins/kaggle-tabular/install.sh` to also wire Codex and OpenCode. See
`plugins/kaggle-tabular/USAGE.md`.
```

- [ ] **Step 3: Full-plugin validation sweep**

Run:
```bash
# manifests parse
python -c "import json; json.load(open('.claude-plugin/marketplace.json')); json.load(open('plugins/kaggle-tabular/.claude-plugin/plugin.json')); print('manifests OK')"
# router lists every reference
for f in $(ls plugins/kaggle-tabular/skills/kaggle-tabular/references/*.md | xargs -n1 basename); do grep -q "$f" plugins/kaggle-tabular/skills/kaggle-tabular/SKILL.md || echo "MISSING ref: $f"; done; echo "ref-index checked"
# NO competition constants in shipped code (references may cite them; assets/src must not)
grep -rniE "thallium|chest_pain|cleveland|0\.9555" plugins/kaggle-tabular/skills/kaggle-tabular/assets && echo "LEAKED CONSTANT" || echo "template clean"
# scaffold smoke (no gpu)
rm -rf /tmp/scaffold-final && python plugins/kaggle-tabular/skills/kaggle-tabular/scripts/scaffold_competition.py fin-demo --dest /tmp/scaffold-final >/dev/null && echo "scaffold OK"
```
Expected: `manifests OK`, `ref-index checked` (no `MISSING ref:`), `template clean`, `scaffold OK`.

- [ ] **Step 4: If `claude` CLI is available, validate the real plugin**

Run: `command -v claude >/dev/null && claude plugin validate plugins/kaggle-tabular || echo "claude CLI absent — manifest JSON already checked above"`
Expected: validation passes, or the fallback message.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs: add USAGE.md cross-harness guide + update CLAUDE.md for plugin layout"
```

---

## Self-Review

**Spec coverage:**
- Plugin+marketplace restructure (mirror superpowers) → Task 1. ✅
- Reference upgrades folding in refined-pipeline (abstracted) → Tasks 2–4. ✅
- New what-works.md → Task 5. ✅
- Full encoded template toolkit (features/ensemble/finalize/wrappers) → Tasks 6–9. ✅
- Scaffolder + GPU markers + SKILL.md sync → Task 10. ✅
- Cross-harness install.sh (Claude/Codex/OpenCode) → Task 11. ✅
- USAGE.md step-by-step + CLAUDE.md refresh → Task 12. ✅
- "No hardcoded competition constants" constraint → guard greps in Tasks 2, 10, 12. ✅
- Keystone code (run_experiment/ledger/cv/metric) unchanged → not touched by any task. ✅

**Placeholder scan:** all code steps contain complete code; doc-porting steps give exact source line ranges + the abstraction rule + the exact structure to emit (not "fill in details"). The one non-ASCII-identifier hazard in Task 7 Step 1 is explicitly corrected in Step 2 with the clean `z_oof`/`z_test` version.

**Type/name consistency:** helper names cited in references (Tasks 3–4) — `OOFTargetEncoder`, `all_cats`, `frequency_encode`, `quantile_bins`, `digit_features`, `categorical_interactions`, `rank_transform`, `dedup_oofs`, `average_seeds`, `hill_climb_logit`, `full_data_refit` — exactly match the implementations in Tasks 6–8, and Tasks 3/4 include grep checks that assert this. Model wrappers all expose `make_fit_fold` matching `lgbm.py` and `run_experiment`'s `fit_fold` contract.
