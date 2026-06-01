#!/usr/bin/env node
const { createClient } = require("./lib/xmem-client.cjs");
const { projectName, redactSecrets, truncate } = require("./lib/plugin-utils.cjs");

async function main() {
  const content = process.argv.slice(2).join(" ").trim();
  if (!content) {
    console.log('Usage: node add-memory.cjs "content to save"');
    return;
  }

  try {
    const cwd = process.cwd();
    const client = createClient(cwd);
    await client.ingest(truncate(redactSecrets(content)), {
      source: "claude-code",
      type: "manual",
      project: projectName(cwd),
    });
    console.log(`Saved memory to XMem for project: ${projectName(cwd)}`);
  } catch (error) {
    console.log(`XMem save failed: ${error.message}`);
  }
}

main().catch((error) => {
  console.error(`Fatal error: ${error.message}`);
  process.exit(1);
});
