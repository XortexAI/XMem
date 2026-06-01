import type { XMemOpenClawConfig } from "./config.ts"
import { redactSecrets, truncate } from "./memory.ts"

export type XMemSearchResult = {
	domain?: string
	content?: string
	score?: number
	metadata?: Record<string, unknown>
}

export class XMemClient {
	constructor(private cfg: XMemOpenClawConfig) {}

	status() {
		return {
			apiKeyConfigured: Boolean(this.cfg.apiKey),
			apiUrl: this.cfg.apiUrl,
			userId: this.cfg.userId,
		}
	}

	private async request(pathname: string, payload: Record<string, unknown>) {
		if (!this.cfg.apiKey) {
			throw new Error("XMEM_API_KEY is not configured.")
		}

		const response = await fetch(`${this.cfg.apiUrl}${pathname}`, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				Authorization: `Bearer ${this.cfg.apiKey}`,
			},
			body: JSON.stringify(payload),
		})

		const text = await response.text()
		let body: any
		try {
			body = JSON.parse(text)
		} catch {
			body = { error: text }
		}

		if (!response.ok || body?.status === "error") {
			throw new Error(body?.error || body?.detail || `XMem request failed with HTTP ${response.status}`)
		}

		return body?.data ?? body
	}

	async search(query: string, limit = 8): Promise<XMemSearchResult[]> {
		const data = await this.request("/v1/memory/search", {
			query: redactSecrets(query),
			user_id: this.cfg.userId,
			top_k: limit,
			domains: ["profile", "temporal", "summary"],
		})
		return data?.results || []
	}

	async addMemory(text: string, metadata: Record<string, unknown> = {}) {
		return this.request("/v1/memory/ingest", {
			user_query: truncate(redactSecrets(text)),
			agent_response: "",
			user_id: this.cfg.userId,
			session_datetime: new Date().toISOString(),
			effort_level: "low",
			metadata,
		})
	}
}
