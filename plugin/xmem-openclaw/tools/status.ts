import { Type } from "@sinclair/typebox"
import type { OpenClawPluginApi } from "openclaw/plugin-sdk"
import type { XMemClient } from "../client.ts"

export function registerStatusTool(api: OpenClawPluginApi, client: XMemClient, toolName = "xmem_status"): void {
	api.registerTool(
		{
			name: toolName,
			label: "XMem Status",
			description: "Show XMem memory plugin status without printing secrets.",
			parameters: Type.Object({}),
			async execute() {
				return {
					content: [{ type: "text" as const, text: JSON.stringify(client.status(), null, 2) }],
				}
			},
		},
		{ name: toolName },
	)
}
