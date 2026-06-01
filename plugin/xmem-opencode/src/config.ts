import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { stripJsoncComments } from "./services/jsonc.js";
import { loadCredentials } from "./services/auth.js";

const CONFIG_DIR = join(homedir(), ".config", "opencode");
const CONFIG_FILES = [
  join(CONFIG_DIR, "xmem.jsonc"),
  join(CONFIG_DIR, "xmem.json"),
];

interface XMemConfig {
  apiKey?: string;
  apiUrl?: string;
  username?: string;
  maxMemories?: number;
  maxProjectMemories?: number;
  keywordPatterns?: string[];
  compactionThreshold?: number;
  autoRecallEveryPrompt?: boolean;
  defaultOrgId?: string;
  defaultRepo?: string;
}

const DEFAULT_KEYWORD_PATTERNS = [
  "remember",
  "memorize",
  "save\\s+this",
  "note\\s+this",
  "keep\\s+in\\s+mind",
  "don'?t\\s+forget",
  "learn\\s+this",
  "store\\s+this",
  "record\\s+this",
  "make\\s+a\\s+note",
  "take\\s+note",
  "jot\\s+down",
  "commit\\s+to\\s+memory",
  "remember\\s+that",
  "never\\s+forget",
  "always\\s+remember",
];

const DEFAULTS = {
  apiUrl: "https://api.xmem.in",
  maxMemories: 5,
  maxProjectMemories: 10,
  compactionThreshold: 0.80,
  autoRecallEveryPrompt: false,
};

function isValidRegex(pattern: string): boolean {
  try {
    new RegExp(pattern);
    return true;
  } catch {
    return false;
  }
}

function validateCompactionThreshold(value: number | undefined): number {
  if (value === undefined || typeof value !== "number" || isNaN(value)) {
    return DEFAULTS.compactionThreshold;
  }
  if (value <= 0 || value > 1) return DEFAULTS.compactionThreshold;
  return value;
}

function loadRawConfig(): { config: XMemConfig; existed: boolean } {
  for (const path of CONFIG_FILES) {
    if (existsSync(path)) {
      try {
        const content = readFileSync(path, "utf-8");
        const json = stripJsoncComments(content);
        return { config: JSON.parse(json) as XMemConfig, existed: true };
      } catch {
        return { config: {}, existed: true };
      }
    }
  }
  return { config: {}, existed: false };
}

const { config: fileConfig, existed: configExisted } = loadRawConfig();
const credentials = loadCredentials();

function getApiKey(): string | undefined {
  if (process.env.XMEM_API_KEY) return process.env.XMEM_API_KEY;
  if (fileConfig.apiKey) return fileConfig.apiKey;
  return credentials?.apiKey;
}

function getApiUrl(): string {
  if (process.env.XMEM_API_URL) return process.env.XMEM_API_URL;
  if (fileConfig.apiUrl) return fileConfig.apiUrl;
  if (credentials?.apiUrl) return credentials.apiUrl;
  return DEFAULTS.apiUrl;
}

function getUsername(): string | undefined {
  if (process.env.XMEM_USERNAME) return process.env.XMEM_USERNAME;
  if (fileConfig.username) return fileConfig.username;
  return credentials?.username;
}

export const XMEM_API_KEY = getApiKey();
export const XMEM_API_URL = getApiUrl();
export const XMEM_USERNAME = getUsername();
export const CONFIG_FILE = CONFIG_FILES[1]!;

export const CONFIG = {
  apiUrl: XMEM_API_URL,
  username: XMEM_USERNAME,
  maxMemories: fileConfig.maxMemories ?? DEFAULTS.maxMemories,
  maxProjectMemories: fileConfig.maxProjectMemories ?? DEFAULTS.maxProjectMemories,
  keywordPatterns: [
    ...DEFAULT_KEYWORD_PATTERNS,
    ...(fileConfig.keywordPatterns ?? []).filter(isValidRegex),
  ],
  compactionThreshold: validateCompactionThreshold(fileConfig.compactionThreshold),
  autoRecallEveryPrompt:
    fileConfig.autoRecallEveryPrompt ??
    (configExisted ? true : DEFAULTS.autoRecallEveryPrompt),
  defaultOrgId: fileConfig.defaultOrgId,
  defaultRepo: fileConfig.defaultRepo,
};

export function isConfigured(): boolean {
  return !!(XMEM_API_KEY && XMEM_USERNAME);
}

export function writeInstallDefaults(isExistingInstall: boolean): void {
  const current = loadRawConfig().config;
  const next: XMemConfig = { ...current };
  if (isExistingInstall) {
    if (next.autoRecallEveryPrompt === undefined) next.autoRecallEveryPrompt = true;
  } else {
    next.autoRecallEveryPrompt = false;
  }
  mkdirSync(CONFIG_DIR, { recursive: true });
  writeFileSync(CONFIG_FILE, JSON.stringify(next, null, 2));
}
