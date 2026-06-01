import { Type } from "@sinclair/typebox"
import type { OpenClawPluginApi } from "openclaw/plugin-sdk"
import type { XMemClient } from "../client.ts"

export function registerSearchTool(api: OpenClawPluginApi, client: XMemClient, toolName = "xmem_search"): void {
	api.registerTool(
		{
			name: toolName,
			label: "XMem Search",
			description: "Search XMem long-term memories for relevant coding context.",
			parameters: Type.Object({
				query: Type.String({ description: "Search query" }),
				limit: Type.Optional(Type.Number({ description: "Max results (default: 8)" })),
			}),
			async execute(_toolCallId: string, params: { query: string; limit?: number }) {
				const results = await client.search(params.query, params.limit ?? 8)
				if (results.length === 0) {
					return { content: [{ type: "text" as const, text: "No relevant XMem memories found." }] }
				}
				const text = results
					.map((r, i) => {
						const score = typeof r.score === "number" ? ` (${Math.round(r.score * 100)}%)` : ""
						return `${i + 1}. ${r.content || ""}${score}`
					})
					.join("\n")
				return {
					content: [{ type: "text" as const, text: `Found ${results.length} XMem memories:\n\n${text}` }],
					details: { count: results.length, memories: results },
				}
			},
		},
		{ name: toolName },
	)
}
