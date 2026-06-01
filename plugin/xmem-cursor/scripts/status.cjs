#!/usr/bin/env node
const { config, projectName } = require("./lib/xmem-client.cjs");

const cfg = config();
console.log(cfg.apiKey ? "XMEM_API_KEY is set" : "XMEM_API_KEY is not set");
console.log(`XMEM_API_URL=${cfg.apiUrl}`);
console.log(`XMEM_USER_ID=${cfg.userId}`);
console.log(`Project=${projectName()}`);
