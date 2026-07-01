#!/usr/bin/env bash
# Cross-harness installer for the kaggle-tabular skill (root-is-plugin layout).
#   Claude Code -> plugin marketplace install
#   Codex       -> native skills dir  (~/.codex/skills/kaggle-tabular)
#   OpenCode    -> register this repo in opencode.jsonc plugin[]  (config hook adds skills.paths)
# Re-runnable. Pass --dry-run to preview without changing anything.
set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$REPO_ROOT/skills/kaggle-tabular"

run() { if [[ $DRY_RUN == 1 ]]; then echo "DRY: $*"; else eval "$*"; fi; }

echo "== Claude Code =="
if command -v claude >/dev/null 2>&1; then
  run "claude plugin marketplace add '$REPO_ROOT' || echo '  (marketplace add returned non-zero — continuing)'"
  run "claude plugin install kaggle-tabular@kaggle-tabular-marketplace || echo '  (plugin install returned non-zero — continuing)'"
else
  echo "  claude CLI not found — skip (manual: claude plugin marketplace add '$REPO_ROOT')"
fi

echo "== Codex =="
CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
if [[ -d "$CODEX_HOME/skills" ]]; then
  run "ln -sfn '$SKILL_DIR' '$CODEX_HOME/skills/kaggle-tabular'"
  echo "  skill -> $CODEX_HOME/skills/kaggle-tabular (native Codex skill discovery)"
else
  echo "  $CODEX_HOME/skills not found — skip"
fi

echo "== OpenCode =="
OC_BASE="${XDG_CONFIG_HOME:-$HOME/.config}/opencode"
OC_CFG="$OC_BASE/opencode.jsonc"
[[ -f "$OC_CFG" ]] || OC_CFG="$OC_BASE/opencode.json"
if [[ -f "$OC_CFG" ]]; then
  if grep -qF "$REPO_ROOT" "$OC_CFG"; then
    echo "  already registered in $OC_CFG"
  elif grep -qE '"plugin"[[:space:]]*:[[:space:]]*\[' "$OC_CFG"; then
    if [[ $DRY_RUN == 1 ]]; then
      echo "DRY: insert \"$REPO_ROOT\" into plugin[] of $OC_CFG (backup .bak)"
    else
      cp "$OC_CFG" "$OC_CFG.bak"
      awk -v p="    \"$REPO_ROOT\"," '
        { print }
        !done && $0 ~ /"plugin"[[:space:]]*:[[:space:]]*\[/ { print p; done=1 }
      ' "$OC_CFG.bak" > "$OC_CFG"
      echo "  registered in $OC_CFG (backup: $OC_CFG.bak)"
    fi
  else
    echo "  no plugin[] array in $OC_CFG — add \"$REPO_ROOT\" manually (see .opencode/INSTALL.md)"
  fi
else
  echo "  opencode config not found under $OC_BASE — skip (see .opencode/INSTALL.md)"
fi

echo "done."
