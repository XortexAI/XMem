const os = require("node:os");
const { loadConfig } = require("./plugin-utils.cjs");

const DEFAULT_API_URL = "https://api.xmem.in";

class XMemClient {
  constructor(options = {}) {
    this.apiUrl = String(options.apiUrl || DEFAULT_API_URL).replace(/\/+$/, "");
    this.apiKey = options.apiKey;
    this.userId = options.userId;
  }

  async request(path, payload) {
    if (!this.apiKey) {
      throw new Error("XMEM_API_KEY is not configured.");
    }

    const response = await fetch(`${this.apiUrl}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(payload),
    });

    const text = await response.text();
    let body = null;
    try {
      body = JSON.parse(text);
    } catch {
      body = { error: text };
    }

    if (!response.ok || body?.status === "error") {
      throw new Error(body?.error || body?.detail || `XMem request failed with HTTP ${response.status}`);
    }

    return body?.data ?? body;
  }

  ingest(content, metadata = {}) {
    return this.request("/v1/memory/ingest", {
      user_query: content,
      agent_response: "",
      user_id: this.userId,
      session_datetime: new Date().toISOString(),
      effort_level: "low",
      metadata,
    });
  }

  search(query, topK = 8) {
    return this.request("/v1/memory/search", {
      query,
      user_id: this.userId,
      top_k: topK,
      domains: ["profile", "temporal", "summary"],
    });
  }
}

function createClient(cwd = process.cwd()) {
  const config = loadConfig(cwd);
  const userInfo = (() => {
    try {
      return os.userInfo().username;
    } catch {
      return "claude-code";
    }
  })();

  return new XMemClient({
    apiKey: process.env.XMEM_API_KEY || process.env.XMEM_CLAUDE_API_KEY || config.apiKey,
    apiUrl: process.env.XMEM_API_URL || process.env.XMEM_CLAUDE_API_URL || config.apiUrl || DEFAULT_API_URL,
    userId: process.env.XMEM_USER_ID || process.env.XMEM_CLAUDE_USER_ID || config.userId || userInfo || "claude-code",
  });
}

function formatResults(data) {
  const results = data?.results || [];
  if (!results.length) return "No XMem memories matched.";

  return results
    .slice(0, 8)
    .map((item, index) => {
      const score = typeof item.score === "number" ? ` score=${item.score.toFixed(3)}` : "";
      const domain = item.domain ? ` domain=${item.domain}` : "";
      return `${index + 1}.${domain}${score}\n${item.content || ""}`.trim();
    })
    .join("\n\n");
}

module.exports = {
  DEFAULT_API_URL,
  XMemClient,
  createClient,
  formatResults,
};
