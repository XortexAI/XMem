const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const MAX_TEXT_LENGTH = 12000;

function readStdin() {
  return new Promise((resolve) => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => {
      data += chunk;
    });
    process.stdin.on("end", () => {
      if (!data.trim()) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(data));
      } catch {
        resolve({});
      }
    });
  });
}

function writeJson(value) {
  process.stdout.write(`${JSON.stringify(value)}\n`);
}

function projectName(cwd = process.cwd()) {
  return path.basename(path.resolve(cwd));
}

function projectConfigPath(cwd = process.cwd()) {
  return path.join(path.resolve(cwd), ".claude", ".xmem-claude", "config.json");
}

function globalConfigPath() {
  return path.join(os.homedir(), ".xmem-claude", "settings.json");
}

function readJsonFile(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return {};
  }
}

function loadConfig(cwd = process.cwd()) {
  return {
    ...readJsonFile(globalConfigPath()),
    ...readJsonFile(projectConfigPath(cwd)),
  };
}

function truncate(text, limit = MAX_TEXT_LENGTH) {
  const value = String(text || "").trim();
  if (value.length <= limit) return value;
  return `${value.slice(0, limit)}\n\n[truncated]`;
}

function redactSecrets(text) {
  return String(text || "")
    .replace(/xmem_[A-Za-z0-9_-]{12,}/g, "[redacted-xmem-key]")
    .replace(/sk-[A-Za-z0-9_-]{16,}/g, "[redacted-api-key]")
    .replace(/(authorization\s*[:=]\s*bearer\s+)[^\s"'`]+/gi, "$1[redacted]")
    .replace(/(bearer\s+)[^\s"'`]+/gi, "$1[redacted]")
    .replace(/((?:api[_-]?key|authorization|token)\s*[:=]\s*)[^\s"'`]+/gi, "$1[redacted]");
}

function extractText(value) {
  if (!value) return "";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map(extractText).filter(Boolean).join("\n");
  if (typeof value === "object") {
    if (typeof value.text === "string") return value.text;
    if (typeof value.content === "string") return value.content;
    if (value.content) return extractText(value.content);
    if (value.message) return extractText(value.message);
  }
  return "";
}

function transcriptTail(transcriptPath, sessionId, cwd) {
  if (!transcriptPath || !fs.existsSync(transcriptPath)) return "";
  const lines = fs.readFileSync(transcriptPath, "utf8").split(/\r?\n/).filter(Boolean);
  const entries = [];

  for (const line of lines) {
    try {
      const item = JSON.parse(line);
      const role = item.type || item.role || item.message?.role || "entry";
      const text = extractText(item.message || item.content || item);
      if (text.trim()) entries.push(`${role}: ${text.trim()}`);
    } catch {
      if (line.trim()) entries.push(line.trim());
    }
  }

  const body = entries.slice(-40).join("\n\n");
  if (!body.trim()) return "";

  return truncate(redactSecrets(`[Claude Code session]\nProject: ${projectName(cwd)}\nSession: ${sessionId || "unknown"}\n\n${body}`));
}

module.exports = {
  extractText,
  globalConfigPath,
  loadConfig,
  projectConfigPath,
  projectName,
  readStdin,
  redactSecrets,
  transcriptTail,
  truncate,
  writeJson,
};
