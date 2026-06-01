#!/usr/bin/env node
const path = require("node:path");
const { saveMemory } = require("./lib/xmem-client.cjs");

function projectName() {
  return path.basename(process.cwd());
}

async function main() {
  const content = process.argv.slice(2).join(" ").trim();

  if (!content) {
    console.log('Usage: node scripts/save-memory.cjs "content to save"');
    return;
  }

  try {
    await saveMemory(content, {
      source: "codex",
      type: "manual",
      project: projectName(),
    });
    console.log(`Saved memory to XMem for project: ${projectName()}`);
  } catch (error) {
    console.log(`XMem save failed: ${error.message}`);
  }
}

main().catch((error) => {
  console.error(`Fatal error: ${error.message}`);
  process.exit(1);
});
