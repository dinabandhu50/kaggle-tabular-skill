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
