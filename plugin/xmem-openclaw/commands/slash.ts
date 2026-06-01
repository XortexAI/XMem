import type { OpenClawPluginApi } from "openclaw/plugin-sdk"
import type { XMemClient } from "../client.ts"
import { log } from "../logger.ts"
import { detectCategory } from "../memory.ts"

export function registerStubCommands(api: OpenClawPluginApi): void {
	api.registerCommand({
		name: "remember",
		description: "Save something to XMem",
		acceptsArgs: true,
		requireAuth: true,
		handler: async () => ({ text: "XMem is not configured. Set XMEM_API_KEY or configure the plugin API key." }),
	})
	api.registerCommand({
		name: "recall",
		description: "Search XMem memories",
		acceptsArgs: true,
		requireAuth: true,
		handler: async () => ({ text: "XMem is not configured. Set XMEM_API_KEY or configure the plugin API key." }),
	})
}

export function registerCommands(api: OpenClawPluginApi, client: XMemClient): void {
	api.registerCommand({
		name: "remember",
		description: "Save something to XMem",
		acceptsArgs: true,
		requireAuth: true,
		handler: async (ctx: { args?: string }) => {
			const text = ctx.args?.trim()
			if (!text) return { text: "Usage: /remember <text to remember>" }
			try {
				await client.addMemory(text, { type: detectCategory(text), source: "openclaw_command" })
				const preview = text.length > 60 ? `${text.slice(0, 60)}...` : text
				return { text: `Remembered in XMem: "${preview}"` }
			} catch (error) {
				log.warn("xmem: /remember failed", error)
				return { text: "Failed to save memory to XMem. Check plugin logs for details." }
			}
		},
	})
	api.registerCommand({
		name: "recall",
		description: "Search XMem memories",
		acceptsArgs: true,
		requireAuth: true,
		handler: async (ctx: { args?: string }) => {
			const query = ctx.args?.trim()
			if (!query) return { text: "Usage: /recall <search query>" }
			try {
				const results = await client.search(query, 8)
				if (!results.length) return { text: `No XMem memories found for: "${query}"` }
				return { text: `Found ${results.length} XMem memories:\n\n${results.map((r, i) => `${i + 1}. ${r.content || ""}`).join("\n")}` }
			} catch (error) {
				log.warn("xmem: /recall failed", error)
				return { text: "Failed to search XMem memories. Check plugin logs for details." }
			}
		},
	})
}
