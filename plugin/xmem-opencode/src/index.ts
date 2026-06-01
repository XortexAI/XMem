import type { Plugin, PluginInput } from "@opencode-ai/plugin";
import type { Part } from "@opencode-ai/sdk";
import { tool } from "@opencode-ai/plugin";

import { xmemClient } from "./services/client.js";
import { formatContextForPrompt, formatSearchResults } from "./services/context.js";
import { getTags, resolveUserId } from "./services/tags.js";
import { stripPrivateContent, isFullyPrivate } from "./services/privacy.js";
import { createCompactionHook, type CompactionContext } from "./services/compaction.js";

import { isConfigured, CONFIG } from "./config.js";
import { log } from "./services/logger.js";
import type { MemoryScope } from "./types/index.js";

const CODE_BLOCK_PATTERN = /```[\s\S]*?```/g;
const INLINE_CODE_PATTERN = /`[^`]+`/g;

const MEMORY_KEYWORD_PATTERN = new RegExp(`\\b(${CONFIG.keywordPatterns.join("|")})\\b`, "i");

const MEMORY_NUDGE_MESSAGE = `[MEMORY TRIGGER DETECTED]
The user wants you to remember something. You MUST use the \`xmem\` tool with \`mode: "add"\` to save this information.

Extract the key information the user wants remembered and save it as a concise, searchable memory.
- Use \`scope: "project"\` for project-specific preferences (e.g., "run lint with tests")
- Use \`scope: "user"\` for cross-project preferences (e.g., "prefers concise responses")

DO NOT skip this step. The user explicitly asked you to remember.`;

function removeCodeBlocks(text: string): string {
  return text.replace(CODE_BLOCK_PATTERN, "").replace(INLINE_CODE_PATTERN, "");
}

function detectMemoryKeyword(text: string): boolean {
  const textWithoutCode = removeCodeBlocks(text);
  return MEMORY_KEYWORD_PATTERN.test(textWithoutCode);
}

export const XMemPlugin: Plugin = async (ctx: PluginInput) => {
  const { directory } = ctx;
  const tags = getTags(directory);
  const injectedSessions = new Set<string>();
  log("Plugin init", { directory, tags, configured: isConfigured() });

  if (!isConfigured()) {
    log("Plugin disabled - XMem credentials not set. Run: bunx opencode-xmem login");
  }

  const modelLimits = new Map<string, number>();

  (async () => {
    try {
      const response = await ctx.client.provider.list();
      if (response.data?.all) {
        for (const provider of response.data.all) {
          if (provider.models) {
            for (const [modelId, model] of Object.entries(provider.models)) {
              if (model.limit?.context) {
                modelLimits.set(`${provider.id}/${modelId}`, model.limit.context);
              }
            }
          }
        }
      }
      log("Model limits loaded", { count: modelLimits.size });
    } catch (error) {
      log("Failed to fetch model limits", { error: String(error) });
    }
  })();

  const getModelLimit = (providerID: string, modelID: string): number | undefined => {
    return modelLimits.get(`${providerID}/${modelID}`);
  };

  const compactionHook =
    isConfigured() && ctx.client
      ? createCompactionHook(ctx as CompactionContext, tags, {
          threshold: CONFIG.compactionThreshold,
          getModelLimit,
        })
      : null;

  return {
    "chat.message": async (input, output) => {
      if (!isConfigured()) return;

      const start = Date.now();

      try {
        const textParts = output.parts.filter(
          (p): p is Part & { type: "text"; text: string } => p.type === "text"
        );

        if (textParts.length === 0) return;

        const userMessage = textParts.map((p) => p.text).join("\n");
        if (!userMessage.trim()) return;

        if (detectMemoryKeyword(userMessage)) {
          const nudgePart: Part = {
            id: `prt_xmem-nudge-${Date.now()}`,
            sessionID: input.sessionID,
            messageID: output.message.id,
            type: "text",
            text: MEMORY_NUDGE_MESSAGE,
            synthetic: true,
          };
          output.parts.push(nudgePart);
        }

        const isFirstMessage = !injectedSessions.has(input.sessionID);
        const shouldInjectContext = CONFIG.autoRecallEveryPrompt || isFirstMessage;

        if (shouldInjectContext) {
          let memoryContext = "";

          if (CONFIG.autoRecallEveryPrompt) {
            const [userRetrieveResult, projectSearchResult, userSearchResult] = await Promise.all([
              xmemClient.retrieve(userMessage, tags.user),
              xmemClient.search(userMessage, tags.project, CONFIG.maxProjectMemories),
              xmemClient.search(userMessage, tags.user, CONFIG.maxMemories),
            ]);

            memoryContext = formatContextForPrompt(
              userRetrieveResult.success ? userRetrieveResult : null,
              projectSearchResult.success ? projectSearchResult : null,
              userSearchResult.success ? userSearchResult : null
            );
          } else {
            const userRetrieveResult = await xmemClient.retrieve("user preferences and context", tags.user);
            memoryContext = formatContextForPrompt(
              userRetrieveResult.success ? userRetrieveResult : null,
              null,
              null
            );
          }

          if (memoryContext) {
            const contextPart: Part = {
              id: `prt_xmem-context-${Date.now()}`,
              sessionID: input.sessionID,
              messageID: output.message.id,
              type: "text",
              text: memoryContext,
              synthetic: true,
            };

            output.parts.unshift(contextPart);

            log("chat.message: context injected", {
              duration: Date.now() - start,
              contextLength: memoryContext.length,
            });
          }

          if (isFirstMessage) {
            injectedSessions.add(input.sessionID);
          }
        }
      } catch (error) {
        log("chat.message: ERROR", { error: String(error) });
      }
    },

    tool: {
      xmem: tool({
        description:
          "Manage and query the XMem persistent memory system. Use 'add' to store knowledge, 'search' for raw records, 'recall' for synthesized answers, 'code' for indexed codebase queries.",
        args: {
          mode: tool.schema.enum(["add", "search", "recall", "code", "help"]).optional(),
          content: tool.schema.string().optional(),
          query: tool.schema.string().optional(),
          scope: tool.schema.enum(["user", "project"]).optional(),
          orgId: tool.schema.string().optional(),
          repo: tool.schema.string().optional(),
          limit: tool.schema.number().optional(),
        },
        async execute(args: {
          mode?: string;
          content?: string;
          query?: string;
          scope?: MemoryScope;
          orgId?: string;
          repo?: string;
          limit?: number;
        }) {
          if (!isConfigured()) {
            return JSON.stringify({
              success: false,
              error: "XMem not configured. Run: bunx opencode-xmem login",
            });
          }

          const mode = args.mode || "help";

          try {
            switch (mode) {
              case "help": {
                return JSON.stringify({
                  success: true,
                  message: "XMem Usage Guide",
                  commands: [
                    { command: "add", description: "Store a new memory", args: ["content", "scope?"] },
                    { command: "search", description: "Search raw memory records", args: ["query", "scope?"] },
                    { command: "recall", description: "Get synthesized answer from memories", args: ["query", "scope?"] },
                    { command: "code", description: "Query indexed codebase", args: ["query", "orgId?", "repo?"] },
                  ],
                  scopes: {
                    user: "Cross-project preferences and knowledge",
                    project: "Project-specific knowledge (default)",
                  },
                });
              }

              case "add": {
                if (!args.content) {
                  return JSON.stringify({ success: false, error: "content parameter is required for add mode" });
                }

                const sanitizedContent = stripPrivateContent(args.content);
                if (isFullyPrivate(args.content)) {
                  return JSON.stringify({ success: false, error: "Cannot store fully private content" });
                }

                const scope = args.scope || "project";
                const userId = resolveUserId(scope, directory);

                const result = await xmemClient.ingest(sanitizedContent, userId);

                if (!result.success) {
                  return JSON.stringify({ success: false, error: result.error || "Failed to add memory" });
                }

                return JSON.stringify({
                  success: true,
                  message: `Memory added to ${scope} scope`,
                  scope,
                  model: result.model,
                });
              }

              case "search": {
                if (!args.query) {
                  return JSON.stringify({ success: false, error: "query parameter is required for search mode" });
                }

                const scope = args.scope;

                if (scope === "user") {
                  const result = await xmemClient.search(args.query, tags.user, args.limit);
                  if (!result.success) {
                    return JSON.stringify({ success: false, error: result.error || "Failed to search memories" });
                  }
                  return formatSearchResults(args.query, scope, result.results || [], args.limit);
                }

                if (scope === "project") {
                  const result = await xmemClient.search(args.query, tags.project, args.limit);
                  if (!result.success) {
                    return JSON.stringify({ success: false, error: result.error || "Failed to search memories" });
                  }
                  return formatSearchResults(args.query, scope, result.results || [], args.limit);
                }

                const [userResult, projectResult] = await Promise.all([
                  xmemClient.search(args.query, tags.user, args.limit),
                  xmemClient.search(args.query, tags.project, args.limit),
                ]);

                if (!userResult.success || !projectResult.success) {
                  return JSON.stringify({
                    success: false,
                    error: userResult.error || projectResult.error || "Failed to search memories",
                  });
                }

                const combined = [
                  ...(userResult.results || []).map((r) => ({ ...r, scope: "user" as const })),
                  ...(projectResult.results || []).map((r) => ({ ...r, scope: "project" as const })),
                ].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

                return JSON.stringify({
                  success: true,
                  query: args.query,
                  count: combined.length,
                  results: combined.slice(0, args.limit || 10).map((r) => ({
                    domain: r.domain,
                    content: r.content,
                    score: Math.round((r.score ?? 0) * 100),
                    scope: r.scope,
                  })),
                });
              }

              case "recall": {
                if (!args.query) {
                  return JSON.stringify({ success: false, error: "query parameter is required for recall mode" });
                }

                const scope = args.scope || "user";
                const userId = resolveUserId(scope, directory);
                const result = await xmemClient.retrieve(args.query, userId, args.limit);

                if (!result.success) {
                  return JSON.stringify({ success: false, error: result.error || "Failed to recall memories" });
                }

                return JSON.stringify({
                  success: true,
                  query: args.query,
                  scope,
                  answer: result.answer,
                  confidence: result.confidence,
                  sources: (result.sources || []).slice(0, args.limit || 5).map((s) => ({
                    domain: s.domain,
                    content: s.content,
                    score: Math.round((s.score ?? 0) * 100),
                  })),
                });
              }

              case "code": {
                if (!args.query) {
                  return JSON.stringify({ success: false, error: "query parameter is required for code mode" });
                }

                const orgId = args.orgId || CONFIG.defaultOrgId;
                const repo = args.repo || CONFIG.defaultRepo;

                if (!orgId || !repo) {
                  return JSON.stringify({
                    success: false,
                    error: "orgId and repo are required for code mode (set in xmem.jsonc or pass as args)",
                  });
                }

                const result = await xmemClient.codeQuery(args.query, orgId, repo, tags.user, args.limit);

                if (!result.success) {
                  return JSON.stringify({ success: false, error: result.error || "Failed to query code" });
                }

                return JSON.stringify({
                  success: true,
                  query: args.query,
                  orgId,
                  repo,
                  answer: result.answer,
                  confidence: result.confidence,
                  sources: (result.sources || []).slice(0, args.limit || 5).map((s) => ({
                    domain: s.domain,
                    content: s.content,
                    score: Math.round((s.score ?? 0) * 100),
                  })),
                });
              }

              default:
                return JSON.stringify({ success: false, error: `Unknown mode: ${mode}` });
            }
          } catch (error) {
            return JSON.stringify({
              success: false,
              error: error instanceof Error ? error.message : String(error),
            });
          }
        },
      }),
    },

    event: async (input: { event: { type: string; properties?: unknown } }) => {
      if (compactionHook) {
        await compactionHook.event(input);
      }
    },
  };
};

export default XMemPlugin;
