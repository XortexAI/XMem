const os = require("node:os");

const DEFAULT_API_URL = "https://api.xmem.in";

function config() {
  return {
    apiKey: process.env.XMEM_API_KEY || process.env.XMEM_CODEX_API_KEY || "",
    apiUrl: (process.env.XMEM_API_URL || process.env.XMEM_CODEX_API_URL || DEFAULT_API_URL).replace(/\/+$/, ""),
    userId: process.env.XMEM_USER_ID || process.env.XMEM_CODEX_USER_ID || safeUsername(),
  };
}

function safeUsername() {
  try {
    return os.userInfo().username || "codex";
  } catch {
    return "codex";
  }
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

async function request(path, payload) {
  const cfg = config();
  if (!cfg.apiKey) {
    throw new Error("XMEM_API_KEY is not configured.");
  }

  const response = await fetch(`${cfg.apiUrl}${path}`, {
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

async function searchMemory(query, topK = 10) {
  const cfg = config();
  return request("/v1/memory/search", {
    query: redactSecrets(query),
    user_id: cfg.userId,
    top_k: topK,
    domains: ["profile", "temporal", "summary"],
  });
}

async function saveMemory(content, metadata = {}) {
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
  config,
  formatResults,
  redactSecrets,
  saveMemory,
  searchMemory,
  truncate,
};
