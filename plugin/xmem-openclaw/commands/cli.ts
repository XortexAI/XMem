import type { OpenClawPluginApi } from "openclaw/plugin-sdk"
import type { XMemClient } from "../client.ts"

export function registerCli(api: OpenClawPluginApi, client?: XMemClient): void {
	api.registerCommand?.({
		name: "xmem-status",
		description: "Show XMem memory plugin status",
		acceptsArgs: false,
		requireAuth: false,
		handler: async () => ({
			text: client ? JSON.stringify(client.status(), null, 2) : "XMem is not configured. Set XMEM_API_KEY.",
		}),
	})
}
