# Installing kaggle-tabular for OpenCode

OpenCode discovers skills from `config.skills.paths`. This plugin's `config` hook
adds its own `skills/` directory to that list, so all you do is register the plugin.

## Install

Add this repo's path (or a git spec) to the `plugin` array in your
`opencode.jsonc` / `opencode.json` (global or project-level):

```jsonc
{
  "plugin": [
    "/absolute/path/to/kaggle-tabular",        // <-- this repo
    "opencode-btw@latest"
  ]
}
```

Or from GitHub once pushed:

```jsonc
{ "plugin": ["kaggle-tabular@git+https://github.com/<you>/kaggle-tabular.git"] }
```

Restart OpenCode. Verify with the native skill tool:

```
use skill tool to list skills          # kaggle-tabular should appear
use skill tool to load kaggle-tabular
```

`install.sh` at the repo root does this registration for you (it inserts the repo
path into your `opencode.jsonc` plugin array, with a `.bak` backup).

## Alternative: skills.paths directly (no plugin)

If you prefer not to register a plugin, point OpenCode at the skills dir yourself:

```jsonc
{ "skills": { "paths": ["/absolute/path/to/kaggle-tabular/skills"] } }
```

## Tool mapping

Skills speak in actions; on OpenCode they resolve to: "invoke a skill" →
native `skill` tool · "read/edit/create a file" → `read` / `apply_patch` ·
"run a shell command" → `bash` · "search" → `grep` / `glob`.
