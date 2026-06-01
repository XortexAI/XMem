export const MEMORY_CATEGORIES = [
	"preference",
	"architecture",
	"error-solution",
	"project-config",
	"learned-pattern",
	"conversation",
] as const

export type MemoryCategory = (typeof MEMORY_CATEGORIES)[number]

export function detectCategory(text: string): MemoryCategory {
	const value = text.toLowerCase()
	if (/(prefer|preference|style|always|never)/.test(value)) return "preference"
	if (/(architecture|design|pattern|module|service|api)/.test(value)) return "architecture"
	if (/(bug|error|fix|failure|root cause|regression)/.test(value)) return "error-solution"
	if (/(config|env|setting|secret|deploy|command)/.test(value)) return "project-config"
	if (/(learned|lesson|note|remember)/.test(value)) return "learned-pattern"
	return "conversation"
}

export function redactSecrets(text: string): string {
	return String(text || "")
		.replace(/xmem_[A-Za-z0-9_-]{12,}/g, "[redacted-xmem-key]")
		.replace(/sk-[A-Za-z0-9_-]{16,}/g, "[redacted-api-key]")
		.replace(/(authorization\s*[:=]\s*bearer\s+)[^\s"'`]+/gi, "$1[redacted]")
		.replace(/(bearer\s+)[^\s"'`]+/gi, "$1[redacted]")
		.replace(/((?:api[_-]?key|authorization|token)\s*[:=]\s*)[^\s"'`]+/gi, "$1[redacted]")
}

export function truncate(text: string, limit = 12000): string {
	const value = String(text || "").trim()
	if (value.length <= limit) return value
	return `${value.slice(0, limit)}\n\n[truncated]`
}
