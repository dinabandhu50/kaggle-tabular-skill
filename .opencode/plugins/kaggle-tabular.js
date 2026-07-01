/**
 * kaggle-tabular plugin for OpenCode.ai
 *
 * Registers the kaggle-tabular skills directory with OpenCode's native skill
 * discovery via the `config` hook (no symlinks, no manual config edits). Unlike
 * superpowers this does NOT inject any bootstrap message — kaggle-tabular is an
 * on-demand skill the agent loads with OpenCode's native `skill` tool when a
 * tabular / Kaggle task appears.
 */
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// this file lives at <repo>/.opencode/plugins/kaggle-tabular.js -> ../../skills
const skillsDir = path.resolve(__dirname, '../../skills');

export const KaggleTabularPlugin = async () => ({
  config: async (config) => {
    config.skills = config.skills || {};
    config.skills.paths = config.skills.paths || [];
    if (!config.skills.paths.includes(skillsDir)) {
      config.skills.paths.push(skillsDir);
    }
  },
});

export default KaggleTabularPlugin;
