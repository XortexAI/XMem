#!/usr/bin/env node
const { createClient } = require("./lib/xmem-client.cjs");
const { projectName, readStdin, transcriptTail, writeJson } = require("./lib/plugin-utils.cjs");

async function main() {
  const input = await readStdin();
  const cwd = input.cwd || process.cwd();
  const content = transcriptTail(input.transcript_path, input.session_id, cwd);

  if (!content) {
    writeJson({ continue: true });
    return;
  }

  try {
    const client = createClient(cwd);
    await client.ingest(content, {
      source: "claude-code",
      type: "session-summary",
      project: projectName(cwd),
      session_id: input.session_id || "",
    });
  } catch (error) {
    console.error(`XMem: ${error.message}`);
  }

  writeJson({ continue: true });
}

main().catch((error) => {
  console.error(`XMem fatal: ${error.message}`);
  process.exit(1);
});
