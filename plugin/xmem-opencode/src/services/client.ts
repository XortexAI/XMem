import { XMemClient } from "xmem-ai";
import type { RetrieveResult, SearchResult, SourceRecord } from "xmem-ai";
import { CONFIG, isConfigured, XMEM_API_KEY, XMEM_API_URL, XMEM_USERNAME } from "../config.js";
import { log } from "./logger.js";

const TIMEOUT_MS = 30000;

function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error(`Timeout after ${ms}ms`)), ms)
    ),
  ]);
}

export class XMemServiceClient {
  private client: XMemClient | null = null;

  private getClient(): XMemClient {
    if (!this.client) {
      if (!isConfigured()) {
        throw new Error("XMem not configured. Run: bunx opencode-xmem login");
      }
      this.client = new XMemClient(XMEM_API_URL, XMEM_API_KEY!, XMEM_USERNAME!);
    }
    return this.client;
  }

  async ingest(content: string, userId: string, agentResponse?: string) {
    log("ingest: start", { userId, contentLength: content.length });
    try {
      const result = await withTimeout(
        this.getClient().ingest({
          user_query: content,
          user_id: userId,
          agent_response: agentResponse,
        }),
        TIMEOUT_MS
      );
      log("ingest: success", { userId });
      return { success: true as const, ...result };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      log("ingest: error", { error: errorMessage });
      return { success: false as const, error: errorMessage };
    }
  }

  async search(query: string, userId: string, topK = CONFIG.maxMemories) {
    log("search: start", { userId, query });
    try {
      const result = await withTimeout(
        this.getClient().search({
          query,
          user_id: userId,
          top_k: topK,
        }),
        TIMEOUT_MS
      );
      log("search: success", { count: result.results?.length || 0 });
      return { success: true as const, ...result };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      log("search: error", { error: errorMessage });
      return { success: false as const, error: errorMessage, results: [], total: 0 };
    }
  }

  async retrieve(query: string, userId: string, topK = CONFIG.maxMemories) {
    log("retrieve: start", { userId, query });
    try {
      const result = await withTimeout(
        this.getClient().retrieve({
          query,
          user_id: userId,
          top_k: topK,
        }),
        TIMEOUT_MS
      );
      log("retrieve: success", { userId });
      return { success: true as const, ...result };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      log("retrieve: error", { error: errorMessage });
      return { success: false as const, error: errorMessage, answer: "", sources: [], confidence: 0, model: "" };
    }
  }

  async codeQuery(query: string, orgId: string, repo: string, userId?: string, topK = 5) {
    log("codeQuery: start", { orgId, repo, query });
    try {
      const result = await withTimeout(
        this.getClient().codeQuery({
          query,
          org_id: orgId,
          repo,
          user_id: userId,
          top_k: topK,
        }),
        TIMEOUT_MS
      );
      log("codeQuery: success", { orgId, repo });
      return { success: true as const, ...result };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      log("codeQuery: error", { error: errorMessage });
      return { success: false as const, error: errorMessage, answer: "", sources: [], confidence: 0 };
    }
  }

  async searchProjectMemories(userId: string, limit = CONFIG.maxProjectMemories) {
    return this.search("project knowledge preferences architecture conventions", userId, limit);
  }
}

export const xmemClient = new XMemServiceClient();

export type { RetrieveResult, SearchResult, SourceRecord };
