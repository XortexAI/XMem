import fs from "node:fs"
import os from "node:os"
import path from "node:path"
import type { OpenClawPluginApi } from "openclaw/plugin-sdk"
import { XMemClient } from "./client.ts"
import { registerCli } from "./commands/cli.ts"
import { registerCommands, registerStubCommands } from "./commands/slash.ts"
import { parseConfig, xmemOpenClawConfigSchema } from "./config.ts"
import { buildCaptureHandler } from "./hooks/capture.ts"
import { buildRecallHandler } from "./hooks/recall.ts"
import { initLogger } from "./logger.ts"
import { buildMemoryRuntime, buildPromptSection } from "./runtime.ts"
import { registerSearchTool } from "./tools/search.ts"
import { registerStatusTool } from "./tools/status.ts"
import { registerStoreTool } from "./tools/store.ts"

function ensureOpenClawMemoryStore(): void {
	try {
		const stateDir = process.env.OPENCLAW_STATE_DIR || path.join(os.homedir(), ".openclaw")
		const storePath = path.join(stateDir, "memory", "main.sqlite")
		if (!fs.existsSync(storePath)) {
			fs.mkdirSync(path.dirname(storePath), { recursive: true })
			fs.writeFileSync(storePath, "")
		}
	} catch {}
}

export default {
	id: "xmem-openclaw",
	name: "XMem",
	description: "OpenClaw powered by XMem memory",
	kind: "memory" as const,
	configSchema: xmemOpenClawConfigSchema,

	register(api: OpenClawPluginApi) {
		const cfg = parseConfig(api.pluginConfig)
		initLogger(api.logger, cfg.debug)
		ensureOpenClawMemoryStore()

		if (!cfg.apiKey) {
			registerCli(api)
			api.logger.info("xmem: not configured - set XMEM_API_KEY or plugin apiKey")
			registerStubCommands(api)
			return
		}

		const client = new XMemClient(cfg)
		registerCli(api, client)

		const runtime = buildMemoryRuntime(client)
		if (typeof api.registerMemoryCapability === "function") {
			api.registerMemoryCapability({
				runtime,
				promptBuilder: buildPromptSection,
				flushPlanResolver: () => null,
			})
		} else {
			api.registerMemoryRuntime?.(runtime)
			api.registerMemoryPromptSection?.(buildPromptSection)
			api.registerMemoryFlushPlan?.(() => null)
		}

		registerSearchTool(api, client)
		registerStoreTool(api, client)
		registerStatusTool(api, client)

		if (cfg.autoRecall) api.on("before_prompt_build", buildRecallHandler(client, cfg))
		if (cfg.autoCapture) api.on("agent_end", buildCaptureHandler(client))

		registerCommands(api, client)

		api.registerService({
			id: "xmem-openclaw",
			start: () => api.logger.info("xmem: connected"),
			stop: () => api.logger.info("xmem: stopped"),
		})
	},
}
