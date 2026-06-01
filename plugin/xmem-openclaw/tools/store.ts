import { Type } from "@sinclair/typebox"
import type { OpenClawPluginApi } from "openclaw/plugin-sdk"
import type { XMemClient } from "../client.ts"
import { detectCategory, MEMORY_CATEGORIES } from "../memory.ts"

export function registerStoreTool(api: OpenClawPluginApi, client: XMemClient, toolName = "xmem_store"): void {
	api.registerTool(
		{
			name: toolName,
			label: "XMem Store",
			description: "Save important information to XMem long-term memory.",
			parameters: Type.Object({
				text: Type.String({ description: "Information to remember" }),
				category: Type.Optional(Type.Unsafe<string>({ type: "string", enum: [...MEMORY_CATEGORIES] })),
			}),
			async execute(_toolCallId: string, params: { text: string; category?: string }) {
				const category = params.category ?? detectCategory(params.text)
				await client.addMemory(params.text, {
					type: category,
					source: "openclaw_tool",
				})
				const preview = params.text.length > 80 ? `${params.text.slice(0, 80)}...` : params.text
				return { content: [{ type: "text" as const, text: `Stored in XMem: "${preview}"` }] }
			},
		},
		{ name: toolName },
	)
}
