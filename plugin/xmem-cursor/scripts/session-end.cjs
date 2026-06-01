#!/usr/bin/env node
const fs = require("node:fs");
const { addMemory, projectName, redactSecrets, truncate } = require("./lib/xmem-client.cjs");

function transcriptFromEnv() {
  const file = process.env.CURSOR_TRANSCRIPT_PATH || process.env.TRANSCRIPT_PATH || "";
  if (!file || !fs.existsSync(file)) return "";
  return truncate(redactSecrets(fs.readFileSync(file, "utf8")));
}

async function main() {
  const transcript = transcriptFromEnv();
  if (!transcript) return;

  try {
    await addMemory(`[Cursor session]\nProject: ${projectName()}\n\n${transcript}`, {
      source: "cursor",
      type: "session-summary",
      project: projectName(),
    });
  } catch (error) {
    console.error(`XMem: ${error.message}`);
  }
}

main().catch((error) => {
  console.error(`XMem fatal: ${error.message}`);
  process.exit(1);
});
