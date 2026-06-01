#!/usr/bin/env node
const { createClient, formatResults } = require("./lib/xmem-client.cjs");
const { redactSecrets } = require("./lib/plugin-utils.cjs");

async function main() {
  const query = process.argv.slice(2).join(" ").trim();
  if (!query) {
    console.log('Usage: node search-memory.cjs "query"');
    return;
  }

  try {
    const client = createClient(process.cwd());
    const data = await client.search(redactSecrets(query), 10);
    console.log(formatResults(data));
  } catch (error) {
    console.log(`XMem search failed: ${error.message}`);
  }
}

main().catch((error) => {
  console.error(`Fatal error: ${error.message}`);
  process.exit(1);
});
