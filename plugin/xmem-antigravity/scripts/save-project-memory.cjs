#!/usr/bin/env node
const { createClient } = require("./lib/xmem-client.cjs");
const { projectName, redactSecrets, truncate } = require("./lib/plugin-utils.cjs");

async function main() {
  const content = process.argv.slice(2).join(" ").trim();
  if (!content) {
    console.log('Usage: node save-project-memory.cjs "project knowledge to save"');
    return;
  }

  try {
    const cwd = process.cwd();
    const client = createClient(cwd);
    await client.ingest(truncate(redactSecrets(content)), {
      source: "antigravity",
      type: "project-knowledge",
      project: projectName(cwd),
    });
    console.log(`Saved project memory to XMem for: ${projectName(cwd)}`);
  } catch (error) {
    console.log(`XMem project save failed: ${error.message}`);
  }
}

main().catch((error) => {
  console.error(`Fatal error: ${error.message}`);
  process.exit(1);
});
