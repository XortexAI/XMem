#!/usr/bin/env node
const { formatResults, projectName, searchMemory } = require("./lib/xmem-client.cjs");

async function main() {
  try {
    const data = await searchMemory(`Cursor project context architecture decisions conventions for ${projectName()}`, {
      limit: 6,
    });
    console.log(`<xmem-context>\n${formatResults(data)}\n</xmem-context>`);
  } catch (error) {
    console.log(`<xmem-status>\nXMem memory unavailable: ${error.message}\nSet XMEM_API_KEY to enable Cursor memory.\n</xmem-status>`);
  }
}

main().catch((error) => {
  console.error(`XMem fatal: ${error.message}`);
  process.exit(1);
});
