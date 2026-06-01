import type { XMemClient } from "./client.ts"

type MemoryProviderStatus = {
	backend: "builtin"
	provider: string
	model?: string
	files?: number
	chunks?: number
	custom?: Record<string, unknown>
}

type RegisteredMemorySearchManager = {
	status(): MemoryProviderStatus
	probeEmbeddingAvailability(): Promise<{ ok: boolean; error?: string }>
	probeVectorAvailability(): Promise<boolean>
	sync?(): Promise<void>
	close?(): Promise<void>
}

export function buildMemoryRuntime(client: XMemClient) {
	const manager: RegisteredMemorySearchManager = {
		status() {
			return {
				backend: "builtin",
				provider: "xmem",
				model: "xmem-remote",
				files: 0,
				chunks: 0,
				custom: client.status(),
			}
		},
		async probeEmbeddingAvailability() {
			try {
				await client.search("connection probe", 1)
				return { ok: true }
			} catch (err) {
				return { ok: false, error: err instanceof Error ? err.message : "XMem unavailable" }
			}
		},
		async probeVectorAvailability() {
			return true
		},
		async sync() {},
		async close() {},
	}

	return {
		async getMemorySearchManager() {
			return { manager }
		},
		resolveMemoryBackendConfig() {
			return { backend: "builtin" as const }
		},
	}
}

export function buildPromptSection(params: { availableTools: Set<string> }): string[] {
	const hasSearch = params.availableTools.has("xmem_search")
	const hasStore = params.availableTools.has("xmem_store")
	if (!hasSearch && !hasStore) return []
	return [
		"## Memory (XMem)",
		"",
		"Memory is managed by XMem remote APIs. Do not store secrets in memory.",
		hasSearch ? "Use xmem_search to look up prior project context, decisions, and solved bugs." : "",
		hasStore ? "Use xmem_store when the user asks you to remember something important." : "",
	].filter(Boolean)
}
