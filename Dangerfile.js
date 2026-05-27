// Dangerfile.js  — XMem PR Review Bot
// Docs: https://danger.systems/js/

// ── 1. Warn on big PRs ────────────────────────────────────────────────────────
const bigPRThreshold = 300;
const totalChanges = danger.github.pr.additions + danger.github.pr.deletions;

if (totalChanges > bigPRThreshold) {
  warn(
    `📦 This PR changes **${totalChanges} lines** (additions + deletions). ` +
    `Large PRs are harder to review thoroughly — consider splitting it.`
  );
}

// ── 2. Require tests alongside source changes ─────────────────────────────────
const hasSourceChanges = danger.git.modified_files
  .some(f => f.startsWith("src/"));
const hasTestChanges = danger.git.modified_files
  .concat(danger.git.created_files)
  .some(f => f.startsWith("tests/"));

if (hasSourceChanges && !hasTestChanges) {
  warn(
    "🧪 Source files in `src/` were modified but no test files changed. " +
    "Please add or update tests to cover your changes."
  );
}

// ── 3. Changelog reminder ────────────────────────────────────────────────────
const hasChangelog = danger.git.modified_files.includes("CHANGELOG.md");
if (!hasChangelog) {
  message("📝 No `CHANGELOG.md` update detected. If this PR introduces a user-visible change, please add an entry.");
}

// ── 4. Flag changes to sensitive files ───────────────────────────────────────
const sensitiveFiles = [
  "src/api/routes/auth.py",
  "src/api/routes/admin.py",
  "src/config/settings.py",
  "src/config/security.py",
  "Dockerfile",
  "docker-compose.yml",
  "docker-compose.prod.yml",
];

const touchedSensitive = danger.git.modified_files
  .concat(danger.git.created_files)
  .filter(f => sensitiveFiles.some(s => f.includes(s)));

if (touchedSensitive.length > 0) {
  fail(
    `🔐 This PR modifies sensitive files: **${touchedSensitive.join(", ")}**. ` +
    `These require review by a core maintainer (@ishaanxgupta or @ved015) before merging.`
  );
}

// ── 5. Dependency changes reminder ───────────────────────────────────────────
const depFiles = ["pyproject.toml", "uv.lock", "requirements.txt"];
const touchedDeps = danger.git.modified_files.filter(f => depFiles.includes(f));

if (touchedDeps.includes("pyproject.toml") || touchedDeps.includes("requirements.txt")) {
  warn(
    "📦 `pyproject.toml` or `requirements.txt` was modified. " +
    "Make sure `uv.lock` is updated (`uv lock`) and the security audit passes."
  );
}

// ── 6. No direct commits to main ─────────────────────────────────────────────
const targetBranch = danger.github.pr.base.ref;
if (targetBranch === "main" || targetBranch === "master") {
  // We're already in a PR — just remind about squash
  message(
    `✅ Targeting \`${targetBranch}\`. Please **squash commits** before merging ` +
    `to keep the git history clean.`
  );
}

// ── 7. PR description completeness ───────────────────────────────────────────
const prBody = danger.github.pr.body || "";
if (prBody.trim().length < 80) {
  fail(
    "📋 PR description is too short. Please describe:\n" +
    "- **What** changed and **Why**\n" +
    "- Any relevant issue links (`Closes #NNN`)\n" +
    "- Steps to test manually"
  );
}
