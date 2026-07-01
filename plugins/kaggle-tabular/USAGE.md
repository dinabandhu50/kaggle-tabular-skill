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
