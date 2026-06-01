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
  return path.join(path.resolve(cwd), ".antigravity", ".xmem-antigravity", "config.json");
}

function globalConfigPath() {
  return path.join(os.homedir(), ".xmem-antigravity", "settings.json");
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
    .replace(/gsk_[A-Za-z0-9_-]{16,}/g, "[redacted-groq-key]")
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

/**
 * Parse an Antigravity transcript.jsonl file into a list of role: text entries.
 *
 * Antigravity stores one JSON object per line with the shape:
 *   { step_index, source, type, content, tool_calls? }
 *
 * source values: USER_EXPLICIT, MODEL, SYSTEM
 * type values:   USER_INPUT, PLANNER_RESPONSE, VIEW_FILE, RUN_COMMAND, ...
 */
function parseAntigravityTranscript(transcriptPath) {
  if (!transcriptPath || !fs.existsSync(transcriptPath)) return [];

  const lines = fs.readFileSync(transcriptPath, "utf8").split(/\r?\n/).filter(Boolean);
  const entries = [];

  for (const line of lines) {
    try {
      const step = JSON.parse(line);
      const source = step.source || "entry";
      const text = extractText(step.content || step.tool_calls || step);
      if (text.trim()) entries.push(`${source}: ${text.trim()}`);
    } catch {
      if (line.trim()) entries.push(line.trim());
    }
  }

  return entries;
}

function transcriptTail(transcriptPath, sessionId, cwd) {
  const entries = parseAntigravityTranscript(transcriptPath);
  if (!entries.length) return "";

  const body = entries.slice(-40).join("\n\n");
  if (!body.trim()) return "";

  return truncate(redactSecrets(`[Antigravity session]\nProject: ${projectName(cwd)}\nSession: ${sessionId || "unknown"}\n\n${body}`));
}

module.exports = {
  extractText,
  globalConfigPath,
  loadConfig,
  parseAntigravityTranscript,
  projectConfigPath,
  projectName,
  readStdin,
  redactSecrets,
  transcriptTail,
  truncate,
  writeJson,
};
