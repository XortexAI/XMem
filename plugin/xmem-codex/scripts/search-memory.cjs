#!/usr/bin/env node
const { formatResults, searchMemory } = require("./lib/xmem-client.cjs");

async function main() {
  const args = process.argv.slice(2);
  const query = args.join(" ").trim();

  if (!query) {
    console.log('Usage: node scripts/search-memory.cjs "query"');
    return;
  }

  try {
    const data = await searchMemory(query, 10);
    console.log(formatResults(data));
  } catch (error) {
    console.log(`XMem search failed: ${error.message}`);
  }
}

main().catch((error) => {
  console.error(`Fatal error: ${error.message}`);
  process.exit(1);
});
