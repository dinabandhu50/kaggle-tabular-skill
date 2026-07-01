# Using the kaggle-tabular skill across Claude Code, Codex, and OpenCode

This repo is **root-is-plugin** (same layout as superpowers): the plugin manifests live at the repo
root and the skill lives at `skills/kaggle-tabular/`. Each harness loads it through its own native
mechanism.

## One-time install (all harnesses)

```bash
bash install.sh          # or: --dry-run to preview
```

Or manually, per harness:

- **Claude Code** (plugin marketplace): `claude plugin marketplace add <path-or-git-url-of-this-repo>`
  then `claude plugin install kaggle-tabular@kaggle-tabular-marketplace`. Restart; the skill is then
  available via the Skill tool and auto-triggers on tabular/Kaggle work.
- **Codex** (native skills dir): symlink `skills/kaggle-tabular` into `~/.codex/skills/`
  (install.sh does this). Codex discovers it alongside its built-in skills; `.codex-plugin/plugin.json`
  is also present if you use Codex's plugin manager.
- **OpenCode** (plugin array): add this repo's path to the `plugin` array in `opencode.jsonc`
  (install.sh inserts it, with a `.bak` backup). The bundled `.opencode/plugins/kaggle-tabular.js`
  registers `skills/` into `config.skills.paths`, so OpenCode's native `skill` tool discovers it —
  no symlinks or manual `skills.paths` edits. See `.opencode/INSTALL.md`.

## Per-competition flow (identical across harnesses)

```bash
# 1. Scaffold a competition repo (encodes the hard rules)
python skills/kaggle-tabular/scripts/scaffold_competition.py \
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
- **Codex:** the skill sits in `~/.codex/skills/kaggle-tabular/` — list/load it with Codex's native
  skill tool.
- **OpenCode:** the plugin registers `skills/` into `config.skills.paths`; list/load `kaggle-tabular`
  with OpenCode's native `skill` tool.
- **Any harness, inside a scaffolded repo:** the scaffolder also writes an `AGENTS.md` into each
  `comp-<name>/`, so even a harness without the global install gets the workflow rules locally.

## The rules the skill enforces (never weaken)

HR-1 no target-aware preprocessing touching the validation fold · HR-2 frozen folds ·
HR-3 exact metric proven vs a known LB point · HR-4 every experiment saves OOF+preds+ledger row ·
HR-5 decide on CV, never public LB · HR-6 log seeds/versions · HR-7 inference-time-available features.
