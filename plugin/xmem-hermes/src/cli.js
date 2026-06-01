#!/usr/bin/env node
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

const CONNECTOR = {
  dir: "xmem-hermes",
  bin: "xmem-hermes",
  id: "hermes",
  display: "Hermes",
  description: "Hermes Agent MCP connector for XMem persistent memory.",
  defaultUserId: "hermes_user",
  category: "hermes",
  installTarget: ".hermes/config.yaml and HERMES.md",
};
const SECRET_PLACEHOLDER = "${XMEM_API_KEY}";

function parseArgs(argv) {
  const command = argv[0] || "help";
  const options = {
    command,
    dryRun: false,
    force: false,
    apiUrl: process.env.XMEM_API_URL || "https://api.xmem.in",
  };
  for (let i = 1; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--dry-run") options.dryRun = true;
    else if (arg === "--force") options.force = true;
    else if (arg === "--config-root") {
      options.configRoot = readValue(argv, i, arg);
      i += 1;
    } else if (arg === "--api-url") {
      options.apiUrl = readValue(argv, i, arg);
      i += 1;
    } else if (arg === "--mcp-command") {
      options.mcpCommand = readValue(argv, i, arg);
      i += 1;
    }
    else if (arg === "--help" || arg === "-h") options.command = "help";
  }
  return options;
}

function readValue(argv, index, arg) {
  const value = argv[index + 1];
  if (!value || value.startsWith("--")) {
    console.error(arg + " requires a value.");
    process.exit(1);
  }
  return value;
}

function configRoot(options) {
  return options.configRoot || homedir();
}

function ensureParent(filePath) {
  mkdirSync(dirname(filePath), { recursive: true });
}

function write(filePath, content, dryRun) {
  if (dryRun) {
    console.log("[dry-run] would write " + filePath);
    return;
  }
  ensureParent(filePath);
  writeFileSync(filePath, content, "utf8");
  console.log("wrote " + filePath);
}

function mcpServer(options) {
  const parts = (options.mcpCommand || "uvx xmem-mcp").split(/\s+/).filter(Boolean);
  return {
    command: parts[0],
    args: parts.slice(1),
    env: {
      TRANSPORT: "stdio",
      XMEM_API_URL: options.apiUrl,
      XMEM_API_KEY: SECRET_PLACEHOLDER,
      XMEM_USERNAME: "${XMEM_USERNAME}",
      DEFAULT_USER_ID: CONNECTOR.defaultUserId,
    },
  };
}

function yamlConfig(options) {
  const server = mcpServer(options);
  return [
    "mcp_servers:",
    "  xmem:",
    "    command: " + server.command,
    "    args:",
    ...server.args.map((arg) => "      - " + arg),
    "    env:",
    "      TRANSPORT: stdio",
    "      XMEM_API_URL: " + options.apiUrl,
    "      XMEM_API_KEY: " + SECRET_PLACEHOLDER,
    "      XMEM_USERNAME: ${XMEM_USERNAME}",
    "      DEFAULT_USER_ID: " + CONNECTOR.defaultUserId,
    "",
  ].join("\n");
}

function memoryInstructions() {
  return [
    "# XMem memory",
    "",
    "Use the xmem MCP tools when the user asks you to remember, recall, search, or connect project context across sessions.",
    "",
    "- Save durable user preferences in user scope.",
    "- Save repository workflows, architecture, and decisions in project scope.",
    "- Search or retrieve before answering questions that depend on prior sessions.",
    "- Never store secrets, API keys, tokens, private keys, or credential material.",
    "",
  ].join("\n");
}

function install(options) {
  const root = configRoot(options);
  const targets = [
    [join(root, ".hermes", "config.yaml"), yamlConfig(options)],
    [join(root, "HERMES.md"), memoryInstructions()],
  ];
  if (!options.force && !options.dryRun) {
    const existing = targets.map(([filePath]) => filePath).filter((filePath) => existsSync(filePath));
    if (existing.length > 0) {
      console.error("Refusing to overwrite existing file(s): " + existing.join(", "));
      console.error("Re-run with --force to replace them.");
      process.exit(1);
    }
  }
  for (const [filePath, content] of targets) {
    write(filePath, content, options.dryRun);
  }
  console.log(CONNECTOR.display + " connector install complete.");
  console.log("Keep XMEM_API_KEY in your environment; it was not copied into generated files.");
}

function doctor(options) {
  const root = configRoot(options);
  const expected = [".hermes/config.yaml", "HERMES.md"];
  let ok = true;
  for (const file of expected) {
    const filePath = join(root, file);
    const present = existsSync(filePath);
    ok = ok && present;
    console.log((present ? "ok " : "missing ") + file);
    if (present) {
      const content = readFileSync(filePath, "utf8");
      if (process.env.XMEM_API_KEY && content.includes(process.env.XMEM_API_KEY)) {
        console.error("secret value found in " + file);
        process.exitCode = 1;
      }
    }
  }
  if (!ok) process.exitCode = 1;
}

async function smokeTest() {
  const apiKey = process.env.XMEM_API_KEY;
  const apiUrl = (process.env.XMEM_API_URL || "https://api.xmem.in").replace(/\/$/, "");
  const username = process.env.XMEM_USERNAME || CONNECTOR.defaultUserId;
  if (!apiKey) {
    console.error("XMEM_API_KEY is required for smoke-test.");
    process.exit(1);
  }
  const response = await fetch(apiUrl + "/v1/memory/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: "Bearer " + apiKey,
      "X-XMem-Username": username,
    },
    body: JSON.stringify({
      query: "xmem connector smoke test",
      user_id: CONNECTOR.defaultUserId,
      top_k: 1,
    }),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok || body.status === "error") {
    console.error("XMem smoke test failed: HTTP " + response.status);
    process.exit(1);
  }
  const data = body.data || body;
  const count = Array.isArray(data.results) ? data.results.length : 0;
  console.log("XMem smoke test ok for " + CONNECTOR.display + " (" + count + " result(s)).");
}

function help() {
  console.log(`
${CONNECTOR.bin} - XMem connector for ${CONNECTOR.display}

Commands:
  install      Write connector config files
  doctor       Check generated files
  smoke-test   Verify XMem API access from environment

Options:
  --config-root <path>    Write config under a custom root
  --api-url <url>         XMem API URL
  --mcp-command <cmd>     MCP launch command, default: uvx xmem-mcp
  --force                 Replace existing generated files
  --dry-run               Print intended writes
`);
}

const options = parseArgs(process.argv.slice(2));
if (options.command === "install") install(options);
else if (options.command === "doctor") doctor(options);
else if (options.command === "smoke-test") await smokeTest();
else help();
