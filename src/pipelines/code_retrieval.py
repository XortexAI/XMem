"""
Code Retrieval Pipeline — multi-namespace search across the code knowledge base.

Searches four Pinecone namespaces (symbols, files, directories, annotations)
and uses Neo4j graph traversal for impact analysis, call chains, and
inheritance hierarchies.

Tools exposed to the LLM:

  search_symbols(query, repo)   → Pinecone semantic search in symbols namespace
  search_files(query, repo)     → Pinecone semantic search in files namespace
  search_annotations(query)     → Pinecone semantic search in annotations namespace
  impact_analysis(symbol, repo) → Neo4j graph traversal (callers, callees, inheritance)
  get_file_context(file, repo)  → Neo4j: symbols defined in file + import graph

Usage::

    pipeline = CodeRetrievalPipeline(org_id="zinnia")
    result = await pipeline.run(
        query="What are the known bugs in PaymentProcessor?",
        user_id="alice",
    )
    print(result.answer)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from src.config import settings
from src.graph.code_graph_client import CodeGraphClient
from src.schemas.code import (
    annotations_namespace,
    directories_namespace,
    files_namespace,
    symbols_namespace,
)
from src.schemas.retrieval import RetrievalResult, SourceRecord
from src.storage.pinecone import PineconeVectorStore

logger = logging.getLogger("xmem.pipelines.code_retrieval")


# ═══════════════════════════════════════════════════════════════════════════
# Tool schemas
# ═══════════════════════════════════════════════════════════════════════════

class SearchSymbols(BaseModel):
    """Search for functions, methods, classes by description or name.
    Use when the question asks about a specific function, class, or API."""

    query: str = Field(description="Short query describing the symbol, e.g. 'payment processing with retry logic'")
    repo: str = Field(default="", description="Repository name to scope the search (optional)")


class SearchFiles(BaseModel):
    """Search for files by their content summary.
    Use when the question is about a file or module rather than a specific function."""

    query: str = Field(description="Short query describing the file purpose, e.g. 'stripe integration'")
    repo: str = Field(default="", description="Repository name to scope the search (optional)")


class SearchAnnotations(BaseModel):
    """Search team knowledge: bug reports, fixes, warnings, explanations.
    Use when asking about known issues, design decisions, or team insights."""

    query: str = Field(description="Short query, e.g. 'duplicate charge bug' or 'retry logic concerns'")


class ImpactAnalysis(BaseModel):
    """Analyze the impact of changing a symbol: who calls it, what it calls,
    inheritance chain, and attached annotations.
    Use when asking 'what would break if I change X?' or 'who uses X?'"""

    symbol_name: str = Field(description="Fully qualified symbol name, e.g. 'PaymentProcessor.process'")
    repo: str = Field(description="Repository name")
    depth: int = Field(default=2, description="How many hops of callers/callees to traverse")


class GetFileContext(BaseModel):
    """Get full context for a file: all symbols it defines and its import graph.
    Use when asking 'what does file X contain?' or 'what does X depend on?'"""

    file_path: str = Field(description="File path, e.g. 'src/services/payment/processor.py'")
    repo: str = Field(description="Repository name")


CODE_TOOLS = [SearchSymbols, SearchFiles, SearchAnnotations, ImpactAnalysis, GetFileContext]


# ═══════════════════════════════════════════════════════════════════════════
# System prompt
# ═══════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are the CODE RETRIEVAL agent in XMem — an enterprise code knowledge system.

Your job is to answer questions about codebases by searching indexed code
knowledge. You have access to:

1. **Symbols** — Every function, method, class in the codebase with summaries,
   signatures, and metadata.
2. **Files** — File-level summaries showing what each file is responsible for.
3. **Annotations** — Team knowledge: bug reports, fixes, warnings, design
   explanations attached to specific symbols or files.
4. **Code Graph** — Dependency graph showing who calls what, import chains,
   and inheritance hierarchies.

═══════════════════════════════════════════════════════════════════════════
AVAILABLE TOOLS
═══════════════════════════════════════════════════════════════════════════

### 1. search_symbols(query, repo?)
   Semantic search over function/class summaries in the symbols index.
   Use when the question is about a specific function, method, or class.

### 2. search_files(query, repo?)
   Semantic search over file-level summaries.
   Use when the question is about a module, service, or file.

### 3. search_annotations(query)
   Search team annotations: bug reports, fixes, warnings, explanations.
   Use when asking about known issues, design decisions, or gotchas.

### 4. impact_analysis(symbol_name, repo, depth?)
   Graph traversal showing callers, callees, inheritance, and annotations
   for a specific symbol. Use for "what breaks if I change X?" questions.

### 5. get_file_context(file_path, repo)
   Full file context: symbols defined + import graph.
   Use for "what's in this file?" or "what does this file depend on?"

═══════════════════════════════════════════════════════════════════════════
DECISION RULES
═══════════════════════════════════════════════════════════════════════════

1. **Specific symbol questions** → search_symbols first, then impact_analysis
   if the user wants dependency info.
2. **Bug / issue questions** → search_annotations + search_symbols.
3. **File / module questions** → search_files, then get_file_context for detail.
4. **Impact / dependency questions** → impact_analysis is your primary tool.
5. **Broad architecture questions** → search_files + search_annotations.
6. **Multi-tool is encouraged** — combine tools for complete answers.
7. **Don't guess** — always search before answering.

═══════════════════════════════════════════════════════════════════════════
INDEXED REPOSITORIES
═══════════════════════════════════════════════════════════════════════════

{repo_catalog}

"""

_ANSWER_PROMPT = """\
You are a senior software engineer assistant. Answer the developer's question
based on the retrieved code knowledge below.

## Retrieved Context:
{context}

## Developer's Question:
{query}

## Instructions:
1. Answer directly and technically. Developers want specifics, not fluff.
2. Reference file paths, function names, and line numbers when available.
3. If annotations mention bugs or warnings, highlight them prominently.
4. For impact analysis results, explain the dependency chain clearly.
5. If the context shows callers of a function, explain who depends on it.
6. Use code formatting (backticks) for symbol names, file paths, and signatures.
7. Only say "I don't have information about that" if the context is truly empty.

Answer:"""


# ═══════════════════════════════════════════════════════════════════════════
# Embedding helper
# ═══════════════════════════════════════════════════════════════════════════

def _get_embed_fn() -> Callable[[str], List[float]]:
    from src.pipelines.ingest import embed_text
    return embed_text


# ═══════════════════════════════════════════════════════════════════════════
# CodeRetrievalPipeline
# ═══════════════════════════════════════════════════════════════════════════

class CodeRetrievalPipeline:
    """Multi-namespace code retrieval with graph traversal."""

    def __init__(
        self,
        org_id: str = "default",
        model: Optional[BaseChatModel] = None,
        code_graph: Optional[CodeGraphClient] = None,
        repos: Optional[List[str]] = None,
    ) -> None:
        self.org_id = org_id
        self.repos = repos or []

        # ── LLM ───────────────────────────────────────────────────────
        if model is None:
            from src.models import get_model
            override = settings.retrieval_model
            self.model = get_model(model_name=override) if override else get_model()
        else:
            self.model = model

        self.model_with_tools = self.model.bind_tools(CODE_TOOLS)

        # ── Pinecone stores (one per namespace type) ──────────────────
        self._stores: Dict[str, PineconeVectorStore] = {}

        # ── Code graph (Neo4j) ────────────────────────────────────────
        self.embed_fn = _get_embed_fn()
        if code_graph is None:
            self.code_graph = CodeGraphClient(
                uri=settings.neo4j_uri,
                username=settings.neo4j_username,
                password=settings.neo4j_password,
                embedding_fn=self.embed_fn,
            )
            self.code_graph.connect()
        else:
            self.code_graph = code_graph

        logger.info("CodeRetrievalPipeline initialized (org=%s)", org_id)

    def _get_store(self, namespace: str) -> PineconeVectorStore:
        """Get or create a PineconeVectorStore for a given namespace."""
        if namespace not in self._stores:
            self._stores[namespace] = PineconeVectorStore(
                api_key=settings.pinecone_api_key,
                index_name=settings.pinecone_index_name,
                dimension=settings.pinecone_dimension,
                metric=settings.pinecone_metric,
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
                namespace=namespace,
                create_if_not_exists=False,
            )
        return self._stores[namespace]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        query: str,
        user_id: str = "",
        repo: str = "",
        top_k: int = 10,
    ) -> RetrievalResult:
        logger.info("=" * 60)
        logger.info("CODE RETRIEVAL START")
        logger.info("  query: %s", query)
        logger.info("  org: %s, repo: %s", self.org_id, repo or "(all)")
        logger.info("=" * 60)

        repo_catalog = self._build_repo_catalog()
        system_prompt = _SYSTEM_PROMPT.format(repo_catalog=repo_catalog)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]

        ai_response: AIMessage = await self.model_with_tools.ainvoke(messages)
        logger.info("LLM tool_calls=%d", len(ai_response.tool_calls or []))

        sources: List[SourceRecord] = []
        tool_messages: List[ToolMessage] = []

        if ai_response.tool_calls:
            for tc in ai_response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc["id"]

                logger.info("  Tool: %s(%s)", tool_name, tool_args)

                records = await self._execute_tool(
                    tool_name, tool_args, repo=repo, top_k=top_k,
                )
                sources.extend(records)

                tool_result_text = self._format_tool_results(records)
                tool_messages.append(
                    ToolMessage(content=tool_result_text, tool_call_id=tool_id)
                )

            context_text = "\n".join(tm.content for tm in tool_messages)
            answer_prompt = _ANSWER_PROMPT.format(
                context=context_text,
                query=query,
            )
            final_response = await self.model.ainvoke(
                [HumanMessage(content=answer_prompt)]
            )
            answer = final_response.content
        else:
            answer = ai_response.content
            logger.info("LLM answered without tool calls")

        if isinstance(answer, list):
            answer = "\n".join(str(c) for c in answer)

        confidence = min(1.0, len(sources) * 0.15) if sources else 0.1

        logger.info("=" * 60)
        logger.info("CODE RETRIEVAL COMPLETE — %d sources", len(sources))
        logger.info("=" * 60)

        return RetrievalResult(
            query=query,
            answer=answer,
            sources=sources,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        repo: str,
        top_k: int,
    ) -> List[SourceRecord]:
        name = tool_name.lower().replace("_", "")

        if name == "searchsymbols":
            return await self._search_symbols(
                query=tool_args.get("query", ""),
                repo=tool_args.get("repo", "") or repo,
                top_k=top_k,
            )
        elif name == "searchfiles":
            return await self._search_files(
                query=tool_args.get("query", ""),
                repo=tool_args.get("repo", "") or repo,
                top_k=top_k,
            )
        elif name == "searchannotations":
            return await self._search_annotations(
                query=tool_args.get("query", ""),
                top_k=top_k,
            )
        elif name == "impactanalysis":
            return self._impact_analysis(
                symbol_name=tool_args.get("symbol_name", ""),
                repo=tool_args.get("repo", "") or repo,
                depth=tool_args.get("depth", 2),
            )
        elif name == "getfilecontext":
            return self._get_file_context(
                file_path=tool_args.get("file_path", ""),
                repo=tool_args.get("repo", "") or repo,
            )
        else:
            logger.warning("Unknown tool: %s", tool_name)
            return []

    # -- Symbols: Pinecone semantic search ─────────────────────────────

    async def _search_symbols(
        self, query: str, repo: str, top_k: int = 10,
    ) -> List[SourceRecord]:
        if not repo:
            logger.warning("search_symbols called without repo — searching all repos")
            results = []
            for r in self.repos:
                results.extend(await self._search_namespace(
                    namespace=symbols_namespace(self.org_id, r),
                    query=query,
                    domain="symbol",
                    top_k=top_k,
                ))
            return results[:top_k]

        return await self._search_namespace(
            namespace=symbols_namespace(self.org_id, repo),
            query=query,
            domain="symbol",
            top_k=top_k,
        )

    # -- Files: Pinecone semantic search ───────────────────────────────

    async def _search_files(
        self, query: str, repo: str, top_k: int = 10,
    ) -> List[SourceRecord]:
        if not repo:
            results = []
            for r in self.repos:
                results.extend(await self._search_namespace(
                    namespace=files_namespace(self.org_id, r),
                    query=query,
                    domain="file",
                    top_k=top_k,
                ))
            return results[:top_k]

        return await self._search_namespace(
            namespace=files_namespace(self.org_id, repo),
            query=query,
            domain="file",
            top_k=top_k,
        )

    # -- Annotations: Pinecone semantic search ─────────────────────────

    async def _search_annotations(
        self, query: str, top_k: int = 10,
    ) -> List[SourceRecord]:
        return await self._search_namespace(
            namespace=annotations_namespace(self.org_id),
            query=query,
            domain="annotation",
            top_k=top_k,
        )

    # -- Impact Analysis: Neo4j graph traversal ────────────────────────

    def _impact_analysis(
        self, symbol_name: str, repo: str, depth: int = 2,
    ) -> List[SourceRecord]:
        if not symbol_name or not repo:
            return []

        impact = self.code_graph.impact_analysis(
            org_id=self.org_id, repo=repo,
            symbol_name=symbol_name, depth=depth,
        )

        records: List[SourceRecord] = []

        # Callers
        for caller in impact.get("callers", []):
            content = (
                f"CALLER: `{caller['symbol_name']}` in `{caller['file_path']}` "
                f"(distance: {caller.get('distance', 1)} hop{'s' if caller.get('distance', 1) > 1 else ''})"
            )
            if caller.get("summary"):
                content += f" — {caller['summary']}"
            records.append(SourceRecord(
                domain="impact_caller", content=content, metadata=caller,
            ))

        # Callees
        for callee in impact.get("callees", []):
            content = (
                f"CALLEE: `{callee['symbol_name']}` in `{callee['file_path']}` "
                f"(distance: {callee.get('distance', 1)} hop{'s' if callee.get('distance', 1) > 1 else ''})"
            )
            if callee.get("summary"):
                content += f" — {callee['summary']}"
            records.append(SourceRecord(
                domain="impact_callee", content=content, metadata=callee,
            ))

        # Inheritance
        for rel in impact.get("inheritance", []):
            direction = "PARENT" if rel.get("relation") == "parent" else "CHILD"
            content = f"{direction}: `{rel['name']}` in `{rel['file']}`"
            records.append(SourceRecord(
                domain="impact_inheritance", content=content, metadata=rel,
            ))

        # Annotations on target
        for ann in impact.get("annotations", []):
            sev = f" [{ann.get('severity', '')}]" if ann.get("severity") else ""
            content = (
                f"ANNOTATION ({ann.get('annotation_type', 'note')}){sev}: "
                f"{ann.get('content', '')}"
            )
            records.append(SourceRecord(
                domain="annotation", content=content,
                score=1.0, metadata=ann,
            ))

        if not records:
            records.append(SourceRecord(
                domain="impact",
                content=f"No dependencies or annotations found for `{symbol_name}` in `{repo}`.",
            ))

        logger.info("  → Impact [%s]: %d results", symbol_name, len(records))
        return records

    # -- File Context: Neo4j ───────────────────────────────────────────

    def _get_file_context(
        self, file_path: str, repo: str,
    ) -> List[SourceRecord]:
        if not file_path or not repo:
            return []

        records: List[SourceRecord] = []

        symbols = self.code_graph.get_file_symbols(
            org_id=self.org_id, repo=repo, file_path=file_path,
        )
        for sym in symbols:
            visibility = "public" if sym.get("is_public") else "private"
            content = (
                f"{sym['symbol_type']} `{sym['symbol_name']}` ({visibility})"
            )
            if sym.get("signature"):
                content += f"\n  Signature: `{sym['signature']}`"
            if sym.get("summary"):
                content += f"\n  Summary: {sym['summary']}"
            records.append(SourceRecord(
                domain="file_symbol", content=content, metadata=sym,
            ))

        imports_data = self.code_graph.get_file_imports(
            org_id=self.org_id, repo=repo, file_path=file_path,
        )
        if imports_data.get("imports"):
            content = f"FILE `{file_path}` IMPORTS: {', '.join(f'`{f}`' for f in imports_data['imports'])}"
            records.append(SourceRecord(
                domain="file_imports", content=content, metadata=imports_data,
            ))
        if imports_data.get("imported_by"):
            content = f"FILE `{file_path}` IMPORTED BY: {', '.join(f'`{f}`' for f in imports_data['imported_by'])}"
            records.append(SourceRecord(
                domain="file_imported_by", content=content, metadata=imports_data,
            ))

        annotations = self.code_graph.get_annotations_for_file(
            org_id=self.org_id, repo=repo, file_path=file_path,
        )
        for ann in annotations:
            sev = f" [{ann.get('severity', '')}]" if ann.get("severity") else ""
            content = (
                f"ANNOTATION ({ann.get('annotation_type', 'note')}){sev}: "
                f"{ann.get('content', '')}"
            )
            records.append(SourceRecord(
                domain="annotation", content=content,
                score=1.0, metadata=ann,
            ))

        if not records:
            records.append(SourceRecord(
                domain="file",
                content=f"No indexed information found for `{file_path}` in `{repo}`.",
            ))

        logger.info("  → FileContext [%s]: %d results", file_path, len(records))
        return records

    # ------------------------------------------------------------------
    # Shared namespace search
    # ------------------------------------------------------------------

    async def _search_namespace(
        self,
        namespace: str,
        query: str,
        domain: str,
        top_k: int = 10,
    ) -> List[SourceRecord]:
        try:
            store = self._get_store(namespace)
            results = await store.search_by_text(
                query_text=query,
                top_k=top_k,
            )

            records = []
            for r in results:
                meta = r.metadata or {}
                if domain == "symbol":
                    sym_name = meta.get("symbol_name", "")
                    sym_type = meta.get("symbol_type", "")
                    file_path = meta.get("file_path", "")
                    sig = meta.get("signature", "")
                    content = f"{sym_type} `{sym_name}` in `{file_path}`"
                    if sig:
                        content += f"\n  Signature: `{sig}`"
                    content += f"\n  Summary: {r.content}"
                elif domain == "file":
                    file_path = meta.get("file_path", "")
                    lang = meta.get("language", "")
                    content = f"File `{file_path}` ({lang}): {r.content}"
                else:
                    content = r.content

                records.append(SourceRecord(
                    domain=domain,
                    content=content,
                    score=r.score,
                    metadata={"id": r.id, **meta},
                ))

            logger.info("  → %s [%s]: %d results", domain, query[:40], len(records))
            return records

        except Exception as exc:
            logger.warning("Namespace search failed (%s): %s", namespace, exc)
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_repo_catalog(self) -> str:
        if not self.repos:
            return "(No repositories configured — search will cover all available namespaces)"
        lines = [f"  - {repo}" for repo in self.repos]
        return "\n".join(lines)

    def _format_tool_results(self, records: List[SourceRecord]) -> str:
        if not records:
            return "No results found."
        lines = []
        for i, rec in enumerate(records, 1):
            score_str = f" (score: {rec.score:.2f})" if rec.score > 0 else ""
            lines.append(f"{i}. [{rec.domain}]{score_str} {rec.content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            self.code_graph.close()
        except Exception:
            pass
        logger.info("CodeRetrievalPipeline closed")
