#!/usr/bin/env node
const { createClient, formatResults } = require("./lib/xmem-client.cjs");
const { projectName, readStdin, writeJson } = require("./lib/plugin-utils.cjs");

async function main() {
  const input = await readStdin();
  const cwd = input.cwd || process.cwd();
  const project = projectName(cwd);

  try {
    const client = createClient(cwd);
    const data = await client.search(`Antigravity project context, architecture, decisions, conventions for ${project}`, 6);
    const formatted = formatResults(data);

    writeJson({
      hookSpecificOutput: {
        hookEventName: "SessionStart",
        additionalContext: `<xmem-context>\n${formatted}\n</xmem-context>`,
      },
    });
  } catch (error) {
    writeJson({
      hookSpecificOutput: {
        hookEventName: "SessionStart",
        additionalContext: `<xmem-status>\nXMem memory unavailable: ${error.message}\nSet XMEM_API_KEY to enable Antigravity memory.\n</xmem-status>`,
      },
    });
  }
}

main().catch((error) => {
  console.error(`XMem fatal: ${error.message}`);
  process.exit(1);
});
