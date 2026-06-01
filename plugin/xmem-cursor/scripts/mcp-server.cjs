#!/usr/bin/env node
const { addMemory, config, formatResults, projectName, searchMemory } = require("./lib/xmem-client.cjs");

let buffer = "";

function send(id, result, error) {
  const payload = error
    ? { jsonrpc: "2.0", id, error: { code: -32000, message: error.message || String(error) } }
    : { jsonrpc: "2.0", id, result };
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}

function tool(name, description, inputSchema) {
  return { name, description, inputSchema };
}

const tools = [
  tool("xmem_status", "Show XMem Cursor configuration without printing secrets.", {
    type: "object",
    properties: {},
  }),
  tool("xmem_search", "Search XMem memory for prior coding context, project decisions, and solved bugs.", {
    type: "object",
    properties: {
      query: { type: "string", description: "Focused search query" },
      limit: { type: "number", description: "Maximum results", default: 10 },
    },
    required: ["query"],
  }),
  tool("xmem_add", "Save important project knowledge to XMem memory.", {
    type: "object",
    properties: {
      content: { type: "string", description: "Memory content to save" },
      type: { type: "string", description: "Memory type, such as architecture or error-solution" },
    },
    required: ["content"],
  }),
];

async function handle(message) {
  const { id, method, params = {} } = message;

  try {
    if (method === "initialize") {
      send(id, {
        protocolVersion: "2024-11-05",
        capabilities: { tools: {} },
        serverInfo: { name: "xmem", version: "0.1.0" },
      });
      return;
    }

    if (method === "notifications/initialized") return;

    if (method === "tools/list") {
      send(id, { tools });
      return;
    }

    if (method === "tools/call") {
      const name = params.name;
      const args = params.arguments || {};

      if (name === "xmem_status") {
        const cfg = config();
        send(id, {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  apiKeyConfigured: Boolean(cfg.apiKey),
                  apiUrl: cfg.apiUrl,
                  userId: cfg.userId,
                  project: projectName(),
                },
                null,
                2,
              ),
            },
          ],
        });
        return;
      }

      if (name === "xmem_search") {
        const data = await searchMemory(args.query, { limit: args.limit || 10 });
        send(id, { content: [{ type: "text", text: formatResults(data) }] });
        return;
      }

      if (name === "xmem_add") {
        await addMemory(args.content, {
          source: "cursor",
          type: args.type || "manual",
          project: projectName(),
        });
        send(id, { content: [{ type: "text", text: `Saved memory to XMem for project: ${projectName()}` }] });
        return;
      }

      throw new Error(`Unknown tool: ${name}`);
    }

    send(id, {}, new Error(`Unsupported method: ${method}`));
  } catch (error) {
    send(id, null, error);
  }
}

process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  buffer += chunk;
  const lines = buffer.split(/\r?\n/);
  buffer = lines.pop() || "";
  for (const rawLine of lines) {
    const line = rawLine.replace(/^\uFEFF/, "");
    if (!line.trim()) continue;
    try {
      handle(JSON.parse(line));
    } catch (error) {
      send(null, null, error);
    }
  }
});
