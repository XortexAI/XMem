import os from "node:os"

export type XMemOpenClawConfig = {
	apiKey: string
	apiUrl: string
	userId: string
	autoRecall: boolean
	autoCapture: boolean
	maxRecallResults: number
	debug: boolean
}

const DEFAULT_API_URL = "https://api.xmem.in"
const ALLOWED_KEYS = [
	"apiKey",
	"apiUrl",
	"userId",
	"autoRecall",
	"autoCapture",
	"maxRecallResults",
	"debug",
]

function assertAllowedKeys(value: Record<string, unknown>): void {
	const unknown = Object.keys(value).filter((key) => !ALLOWED_KEYS.includes(key))
	if (unknown.length > 0) {
		throw new Error(`xmem-openclaw config has unknown keys: ${unknown.join(", ")}`)
	}
}

function resolveEnvVars(value: string): string {
	return value.replace(/\$\{([^}]+)\}/g, (_, envVar: string) => process.env[envVar] || "")
}

function safeUsername(): string {
	try {
		return os.userInfo().username || "openclaw"
	} catch {
		return "openclaw"
	}
}

export function parseConfig(rawConfig: Record<string, unknown> | null | undefined = {}): XMemOpenClawConfig {
	const raw = rawConfig ?? {}
	if (Object.keys(raw).length > 0) {
		assertAllowedKeys(raw)
	}

	const envApiKey = process.env.XMEM_API_KEY || process.env.XMEM_OPENCLAW_API_KEY || ""
	const envApiUrl = process.env.XMEM_API_URL || process.env.XMEM_OPENCLAW_API_URL || ""
	const envUserId = process.env.XMEM_USER_ID || process.env.XMEM_OPENCLAW_USER_ID || ""

	return {
		apiKey: resolveEnvVars(String(raw.apiKey || envApiKey || "")),
		apiUrl: resolveEnvVars(String(raw.apiUrl || envApiUrl || DEFAULT_API_URL)).replace(/\/+$/, ""),
		userId: resolveEnvVars(String(raw.userId || envUserId || safeUsername())),
		autoRecall: raw.autoRecall !== false,
		autoCapture: raw.autoCapture !== false,
		maxRecallResults: Number(raw.maxRecallResults || 8),
		debug: Boolean(raw.debug),
	}
}

export const xmemOpenClawConfigSchema = {
	jsonSchema: {
		type: "object",
		additionalProperties: false,
		properties: {
			apiKey: { type: "string" },
			apiUrl: { type: "string" },
			userId: { type: "string" },
			autoRecall: { type: "boolean" },
			autoCapture: { type: "boolean" },
			maxRecallResults: { type: "number", minimum: 1, maximum: 20 },
			debug: { type: "boolean" },
		},
	},
	parse: parseConfig,
}
