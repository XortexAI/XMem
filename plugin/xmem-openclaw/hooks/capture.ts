import type { XMemClient } from "../client.ts"
import { log } from "../logger.ts"
import { detectCategory } from "../memory.ts"

export function buildCaptureHandler(client: XMemClient) {
	return async (event: Record<string, unknown>) => {
		try {
			const text = String(event.text || event.output || event.message || "")
			if (text.trim().length < 80) return
			await client.addMemory(text, {
				type: detectCategory(text),
				source: "openclaw_auto_capture",
			})
		} catch (error) {
			log.warn("xmem: auto-capture failed", error)
		}
	}
}
