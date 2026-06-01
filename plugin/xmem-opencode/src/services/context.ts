import type { RetrieveResult, SearchResult, SourceRecord } from "./client.js";

function formatSource(source: SourceRecord): string {
  const score = Math.round((source.score ?? 0) * 100);
  return `- [${source.domain}] (${score}%) ${source.content}`;
}

export function formatContextForPrompt(
  userRetrieve: RetrieveResult | null,
  projectSearch: SearchResult | null,
  userSearch: SearchResult | null
): string {
  const parts: string[] = ["[XMEM]"];

  if (userRetrieve?.answer) {
    parts.push("\nRelevant Context:");
    parts.push(userRetrieve.answer);

    if (userRetrieve.sources?.length) {
      parts.push("\nSources:");
      userRetrieve.sources.slice(0, 5).forEach((s) => parts.push(formatSource(s)));
    }
  }

  const projectResults = projectSearch?.results || [];
  if (projectResults.length > 0) {
    parts.push("\nProject Knowledge:");
    projectResults.forEach((s) => parts.push(formatSource(s)));
  }

  const userResults = userSearch?.results || [];
  if (userResults.length > 0) {
    parts.push("\nUser Memories:");
    userResults.forEach((s) => parts.push(formatSource(s)));
  }

  if (parts.length === 1) {
    return "";
  }

  return parts.join("\n");
}

export function formatSearchResults(
  query: string,
  scope: string | undefined,
  results: SourceRecord[],
  limit?: number
): string {
  return JSON.stringify({
    success: true,
    query,
    scope,
    count: results.length,
    results: results.slice(0, limit || 10).map((r) => ({
      domain: r.domain,
      content: r.content,
      score: Math.round((r.score ?? 0) * 100),
    })),
  });
}
