let debugEnabled = false
let pluginLogger: { debug?: (...args: unknown[]) => void; info?: (...args: unknown[]) => void; warn?: (...args: unknown[]) => void; error?: (...args: unknown[]) => void } | undefined

export function initLogger(logger: typeof pluginLogger, debug: boolean): void {
	pluginLogger = logger
	debugEnabled = debug
}

export const log = {
	debug(...args: unknown[]) {
		if (debugEnabled) pluginLogger?.debug?.(...args)
	},
	info(...args: unknown[]) {
		pluginLogger?.info?.(...args)
	},
	warn(...args: unknown[]) {
		pluginLogger?.warn?.(...args)
	},
	error(...args: unknown[]) {
		pluginLogger?.error?.(...args)
	},
}
