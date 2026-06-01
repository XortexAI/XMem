#!/usr/bin/env node
const { config, DEFAULT_API_URL } = require("./lib/xmem-client.cjs");

const cfg = config();

console.log(cfg.apiKey ? "XMEM_API_KEY is set" : "XMEM_API_KEY is not set");
console.log(`XMEM_API_URL=${cfg.apiUrl || DEFAULT_API_URL}`);
console.log(`XMEM_USER_ID=${cfg.userId || "codex"}`);
