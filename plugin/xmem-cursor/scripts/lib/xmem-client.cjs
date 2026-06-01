const os = require("node:os");
const path = require("node:path");

const DEFAULT_API_URL = "https://api.xmem.in";

function config() {
  const userId = (() => {
    try {
      return os.userInfo().username || "cursor";
    } catch {
      return "cursor";
    }
  })();

  return {
    apiKey: process.env.XMEM_API_KEY || process.env.XMEM_CURSOR_API_KEY || "",
    apiUrl: String(process.env.XMEM_API_URL || process.env.XMEM_CURSOR_API_URL || DEFAULT_API_URL).replace(/\/+$/, ""),
    userId: process.env.XMEM_USER_ID || process.env.XMEM_CURSOR_USER_ID || userId,
  };
}

function projectName(cwd = process.cwd()) {
  return path.basename(cwd || process.cwd()) || "cursor-project";
}

function redactSecrets(text) {
  return String(text || "")
    .replace(/xmem_[A-Za-z0-9_-]{12,}/g, "[redacted-xmem-key]")
    .replace(/sk-[A-Za-z0-9_-]{16,}/g, "[redacted-api-key]")
    .replace(/(authorization\s*[:=]\s*bearer\s+)[^\s"'`]+/gi, "$1[redacted]")
    .replace(/(bearer\s+)[^\s"'`]+/gi, "$1[redacted]")
    .replace(/((?:api[_-]?key|authorization|token)\s*[:=]\s*)[^\s"'`]+/gi, "$1[redacted]");
}

function truncate(text, limit = 12000) {
  const value = String(text || "").trim();
  if (value.length <= limit) return value;
  return `${value.slice(0, limit)}\n\n[truncated]`;
}

async function request(pathname, payload) {
  const cfg = config();
  if (!cfg.apiKey) {
    throw new Error("XMEM_API_KEY is not configured.");
  }

  const response = await fetch(`${cfg.apiUrl}${pathname}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${cfg.apiKey}`,
    },
    body: JSON.stringify(payload),
  });

  const text = await response.text();
  let body;
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

function searchMemory(query, options = {}) {
  const cfg = config();
  return request("/v1/memory/search", {
    query: redactSecrets(query),
    user_id: cfg.userId,
    top_k: options.limit || options.topK || 8,
    domains: ["profile", "temporal", "summary"],
  });
}

function addMemory(content, metadata = {}) {
  const cfg = config();
  return request("/v1/memory/ingest", {
    user_query: truncate(redactSecrets(content)),
    agent_response: "",
    user_id: cfg.userId,
    session_datetime: new Date().toISOString(),
    effort_level: "low",
    metadata,
  });
}

function formatResults(data) {
  const results = data?.results || [];
  if (!results.length) return "No XMem memories matched.";

  return results
    .slice(0, 10)
    .map((item, index) => {
      const score = typeof item.score === "number" ? ` score=${item.score.toFixed(3)}` : "";
      const domain = item.domain ? ` domain=${item.domain}` : "";
      return `${index + 1}.${domain}${score}\n${item.content || ""}`.trim();
    })
    .join("\n\n");
}

module.exports = {
  DEFAULT_API_URL,
  addMemory,
  config,
  formatResults,
  projectName,
  redactSecrets,
  searchMemory,
  truncate,
};
