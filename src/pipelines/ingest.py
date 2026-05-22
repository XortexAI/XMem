"""
Ingest Pipeline — the full LangGraph workflow for storing user memories.

Takes raw user input, processes it through extraction agents, judges each
domain, and executes writes via the Weaver.

Flow::

    ┌─────────┐     ┌──────────────┐
    │  START   │────>│  classify    │
    └─────────┘     └──────┬───────┘
                           │ fan-out (conditional)
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ profile  │ │ temporal │ │ summary  │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
              │            │            │
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ judge_p  │ │ judge_t  │ │ judge_s  │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
              │            │            │
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ weave_p  │ │ weave_t  │ │ weave_s  │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
              │            │            │
              └────────────┼────────────┘
                           ▼
                      ┌─────────┐
                      │   END   │
                      └─────────┘

Each vertical lane (profile / temporal / summary) runs independently via
LangGraph's ``Send`` (fan-out).  All three converge at END.

Usage::

    from src.pipelines.ingest import IngestPipeline

    pipeline = IngestPipeline()     # reads config from env / .env
    result = await pipeline.run({
        "user_query": "I just got a new job at Google!",
        "agent_response": "Congratulations!",
        "user_id": "user_123",
    })
"""

from __future__ import annotations

import functools
import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional
import operator

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict, Annotated

from src.agents.classifier import ClassifierAgent
from src.agents.code import CodeAgent
from src.agents.image import ImageAgent
from src.agents.judge import JudgeAgent
from src.agents.profiler import ProfilerAgent
from src.agents.snippet import SnippetAgent
from src.agents.summarizer import SummarizerAgent
from src.agents.temporal import TemporalAgent
from src.config import settings
from src.graph.code_graph_client import CodeGraphClient
from src.graph.neo4j_client import Neo4jClient
from src.graph.schema import setup_constraints
from src.models import get_model, get_vision_model
from src.pipelines.weaver import Weaver
from src.schemas.classification import ClassificationResult
from src.schemas.code import (
    CodeAnnotationResult,
    SnippetExtractionResult,
    annotations_namespace,
    snippets_namespace,
)
from src.schemas.events import EventResult
from src.schemas.image import ImageResult
from src.schemas.judge import JudgeDomain, JudgeResult, OperationType, Operation
from src.schemas.profile import ProfileResult
from src.schemas.summary import SummaryResult
from src.schemas.weaver import WeaverResult
from src.storage.base import BaseVectorStore, SearchResult
from src.storage.factory import get_vector_store
from src.config.effort import EffortLevel, EffortConfig, get_effort_config, chunk_text, estimate_tokens

logger = logging.getLogger("xmem.pipelines.ingest")


# ---------------------------------------------------------------------------
# Embedding helper — supports Google GenAI, OpenAI, Amazon Bedrock, Ollama, FastEmbed
# ---------------------------------------------------------------------------

import json as _json

from google import genai
from google.genai import types

_embedding_client: Optional[genai.Client] = None
_openai_embedding_client = None
_bedrock_embedding_client = None
_fastembed_model = None


def _is_bedrock_embedding() -> bool:
    """Check if the configured embedding model is an Amazon Bedrock model."""
    return settings.embedding_model.lower().startswith("amazon.")


def _is_openai_embedding() -> bool:
    """Check if the configured embedding model is an OpenAI embedding model."""
    return settings.embedding_model.lower().startswith("text-embedding")


def _embedding_provider() -> str:
    provider = (settings.embedding_provider or "auto").strip().lower()
    if provider == "auto":
        if _is_bedrock_embedding():
            return "bedrock"
        if _is_openai_embedding():
            return "openai"
        return "gemini"
    return provider


def get_embedding_client() -> genai.Client:
    global _embedding_client
    if _embedding_client is None:
        api_key_to_use = settings.gemini_api_key or None
        _embedding_client = genai.Client(api_key=api_key_to_use) if api_key_to_use else genai.Client()
        logger.info("Loaded Gemini embedding client for model: %s", settings.embedding_model)
    return _embedding_client


def _get_bedrock_embedding_client():
    global _bedrock_embedding_client
    if _bedrock_embedding_client is None:
        import boto3
        from botocore.config import Config

        kwargs = {
            "region_name": settings.bedrock_region,
            "config": Config(read_timeout=60),
        }
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            kwargs["aws_access_key_id"] = settings.aws_access_key_id
            kwargs["aws_secret_access_key"] = settings.aws_secret_access_key

        _bedrock_embedding_client = boto3.client("bedrock-runtime", **kwargs)
        logger.info("Loaded Bedrock embedding client for model: %s", settings.embedding_model)
    return _bedrock_embedding_client


def _get_openai_embedding_client():
    """Lazily create an OpenAI client for embeddings."""
    global _openai_embedding_client
    if _openai_embedding_client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is not installed. Install with: pip install openai"
            ) from exc
        api_key = settings.openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set but EMBEDDING_PROVIDER=openai")
        _openai_embedding_client = OpenAI(api_key=api_key)
        logger.info("Loaded OpenAI embedding client for model: %s", settings.embedding_model)
    return _openai_embedding_client


def _get_fastembed_model():
    global _fastembed_model
    if _fastembed_model is None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "FastEmbed is not installed. Install local embedding dependencies "
                "with: pip install -e \".[local]\""
            ) from exc
        _fastembed_model = TextEmbedding(model_name=settings.fastembed_model)
        logger.info("Loaded FastEmbed model: %s", settings.fastembed_model)
    return _fastembed_model


def _ensure_embedding_dimension(values: tuple[float, ...], provider: str) -> tuple[float, ...]:
    expected = int(settings.pinecone_dimension)
    if len(values) != expected:
        raise ValueError(
            f"{provider} embedding dimension is {len(values)}, but PINECONE_DIMENSION "
            f"is {expected}. Set PINECONE_DIMENSION to match the selected embedding model "
            "before creating vector indexes."
        )
    return values


@functools.lru_cache(maxsize=4096)
def embed_text(text: str) -> tuple[float, ...]:
    """Embed a single text string → tuple of floats.

    Dispatches to the configured embedding provider (auto-detected or explicit).
    Supported: gemini, openai, bedrock, ollama, fastembed.
    """
    provider = _embedding_provider()
    if provider == "gemini":
        return _embed_text_gemini(text)
    if provider == "openai":
        return _embed_text_openai(text)
    if provider == "bedrock":
        return _embed_text_bedrock(text)
    if provider == "ollama":
        return _embed_text_ollama(text)
    if provider == "fastembed":
        return _embed_text_fastembed(text)
    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER={provider!r}. "
        "Use auto, gemini, openai, bedrock, ollama, or fastembed."
    )


def _embed_text_gemini(text: str) -> tuple[float, ...]:
    import time as _time
    client = get_embedding_client()
    start = _time.perf_counter()
    result = client.models.embed_content(
        model=settings.embedding_model,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=settings.pinecone_dimension),
    )
    elapsed = _time.perf_counter() - start
    [embedding_obj] = result.embeddings

    # Track embedding call for cost analytics
    input_tokens = getattr(result, "input_tokens", 0) or len(text.split())
    try:
        from src.config.analytics import analytics
        analytics.track_llm_call(
            provider="gemini",
            model=settings.embedding_model,
            agent="embedding",
            latency_ms=round(elapsed * 1000, 2),
            input_tokens=input_tokens,
            output_tokens=0,
            total_tokens=input_tokens,
        )
    except Exception:
        pass

    return tuple(embedding_obj.values)


def _embed_text_openai(text: str) -> tuple[float, ...]:
    """Embed text using the OpenAI Embeddings API.

    Supports text-embedding-3-small, text-embedding-3-large, and
    text-embedding-ada-002.  The v3 models accept a ``dimensions``
    parameter for native dimension reduction (e.g. 384 for Pinecone).
    """
    import time as _time

    client = _get_openai_embedding_client()
    model = settings.embedding_model
    dimension = int(settings.pinecone_dimension)

    start = _time.perf_counter()

    # text-embedding-3-* supports the dimensions parameter;
    # ada-002 does not (fixed at 1536).
    kwargs: dict = {"model": model, "input": text}
    if model.startswith("text-embedding-3"):
        kwargs["dimensions"] = dimension

    response = client.embeddings.create(**kwargs)
    elapsed = _time.perf_counter() - start
    embedding = response.data[0].embedding

    # Track embedding call for cost analytics
    input_tokens = getattr(response.usage, "total_tokens", 0) or len(text.split())
    try:
        from src.config.analytics import analytics
        analytics.track_llm_call(
            provider="openai",
            model=model,
            agent="embedding",
            latency_ms=round(elapsed * 1000, 2),
            input_tokens=input_tokens,
            output_tokens=0,
            total_tokens=input_tokens,
        )
    except Exception:
        pass

    values = tuple(float(v) for v in embedding)
    return _ensure_embedding_dimension(values, "OpenAI")


def _embed_text_bedrock(text: str) -> tuple[float, ...]:
    client = _get_bedrock_embedding_client()

    request_body = {
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "GENERIC_INDEX",
            "embeddingDimension": settings.pinecone_dimension,
            "text": {
                "truncationMode": "END",
                "value": text,
            },
        },
    }

    response = client.invoke_model(
        body=_json.dumps(request_body),
        modelId=settings.embedding_model,
        accept="application/json",
        contentType="application/json",
    )

    response_body = _json.loads(response["body"].read())
    return tuple(response_body["embeddings"][0]["embedding"])


def _embed_text_ollama(text: str) -> tuple[float, ...]:
    """Embed text with a local Ollama server.

    Supports Ollama's newer /api/embed endpoint and falls back to the older
    /api/embeddings shape for compatibility.
    """
    import httpx

    model = settings.ollama_embedding_model or settings.embedding_model
    if model == "gemini-embedding-001":
        model = "nomic-embed-text"

    base_url = settings.ollama_base_url.rstrip("/")
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{base_url}/api/embed",
            json={"model": model, "input": text},
        )
        if response.status_code == 404:
            response = client.post(
                f"{base_url}/api/embeddings",
                json={"model": model, "prompt": text},
            )
        response.raise_for_status()
        data = response.json()

    if "embeddings" in data:
        [embedding] = data["embeddings"]
    else:
        embedding = data["embedding"]
    return _ensure_embedding_dimension(tuple(float(v) for v in embedding), "Ollama")


def _embed_text_fastembed(text: str) -> tuple[float, ...]:
    model = _get_fastembed_model()
    embedding = next(model.embed([text]))
    return _ensure_embedding_dimension(tuple(float(v) for v in embedding), "FastEmbed")


# ---------------------------------------------------------------------------
# LangGraph state (typed dict shared across all nodes)
# ---------------------------------------------------------------------------

class IngestState(TypedDict, total=False):
    # ── input ─────────────────────────────────────────────────────────
    user_query: str
    agent_response: str
    user_id: str
    image_url: str
    session_datetime: str

    # ── routing (internal — set by _route_after_classify) ─────────────
    profile_queries: List[str]      # batched profile sub-queries
    temporal_queries: List[str]     # batched temporal sub-queries
    image_queries: List[str]        # batched image sub-queries
    code_queries: List[str]         # batched code sub-queries

    # ── classification ────────────────────────────────────────────────
    classification_result: ClassificationResult

    # ── extraction outputs ────────────────────────────────────────────
    profile_result: ProfileResult
    temporal_result: EventResult
    summary_result: SummaryResult
    image_result: ImageResult
    code_result: CodeAnnotationResult
    snippet_result: SnippetExtractionResult

    # ── judge outputs ─────────────────────────────────────────────────
    profile_judge: JudgeResult
    temporal_judge: JudgeResult
    summary_judge: JudgeResult
    image_judge: JudgeResult
    code_judge: JudgeResult
    snippet_judge: JudgeResult

    disabled_domains: List[str]

    # ── weaver outputs ────────────────────────────────────────────────
    profile_weaver: WeaverResult
    temporal_weaver: WeaverResult
    summary_weaver: WeaverResult
    image_weaver: WeaverResult
    code_weaver: WeaverResult
    snippet_weaver: WeaverResult

    # ── metadata ──────────────────────────────────────────────────────
    status: Annotated[str, lambda a, b: b]
    errors: Annotated[List[str], operator.add]


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class IngestPipeline:
    """End-to-end ingest pipeline wired with real Pinecone + Neo4j."""

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        neo4j_client: Optional[Neo4jClient] = None,
        code_graph_client: Optional[CodeGraphClient] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        org_id: str = "default",
    ) -> None:
        self.org_id = org_id

        # ── Embedding function ────────────────────────────────────────
        self.embed_fn = embed_fn or embed_text

        # ── Pinecone (vector store) ───────────────────────────────────
        if vector_store:
            self.vector_store = vector_store
        else:
            self.vector_store = get_vector_store(
                namespace=settings.pinecone_namespace,
            )
        logger.info("Vector store initialised (provider=%s).", settings.vector_store_provider)

        # ── Code annotations Pinecone store (annotations namespace) ──
        self.code_vector_store = get_vector_store(
            namespace=annotations_namespace(org_id),
            create_if_not_exists=False,
        )
        logger.info("Code annotations vector store initialised (ns=%s).", annotations_namespace(org_id))

        # ── Neo4j (graph store — temporal) ────────────────────────────
        if neo4j_client:
            self.neo4j = neo4j_client
        else:
            self.neo4j = Neo4jClient(
                uri=settings.neo4j_uri,
                username=settings.neo4j_username,
                password=settings.neo4j_password,
                embedding_fn=self.embed_fn,
            )
            self.neo4j.connect()
            try:
                setup_constraints(self.neo4j.driver)
                self.neo4j.initialize_date_nodes()
            except Exception as exc:
                logger.warning("Neo4j init (constraints/dates) failed: %s", exc)
        logger.info("Neo4j client initialised.")

        # ── Neo4j (code graph) ────────────────────────────────────────
        if code_graph_client:
            self.code_graph = code_graph_client
        else:
            self.code_graph = CodeGraphClient(
                uri=settings.neo4j_uri,
                username=settings.neo4j_username,
                password=settings.neo4j_password,
                embedding_fn=self.embed_fn,
            )
            self.code_graph.connect()
            try:
                self.code_graph.setup()
            except Exception as exc:
                logger.warning("Code graph init (constraints) failed: %s", exc)
        logger.info("Code graph client initialised.")

        # ── LLM ──────────────────────────────────────────────────────
        self.model = get_model()

        def _agent_model(agent_name: str):
            """Get model for a specific agent, falling back to default."""
            override = getattr(settings, f"{agent_name}_model", None)
            if override:
                return get_model(model_name=override)
            return self.model

        # ── Agents ────────────────────────────────────────────────────
        self.classifier = ClassifierAgent(model=_agent_model("classifier"))
        self.profiler = ProfilerAgent(model=_agent_model("profiler"))
        self.temporal = TemporalAgent(model=_agent_model("temporal"))
        self.summarizer = SummarizerAgent(model=_agent_model("summarizer"))
        self.image_agent = ImageAgent(model=get_vision_model())
        self.code_agent = CodeAgent(model=_agent_model("code"))
        self.snippet_agent = SnippetAgent(model=_agent_model("code"))

        self.judge = JudgeAgent(
            model=_agent_model("judge"),
            vector_store=self.vector_store,
            graph_event_search=self._graph_event_search_wrapper,
            top_k=3,
        )

        # Snippet stores are user-scoped — lazily created per user_id
        self._snippet_stores: Dict[str, BaseVectorStore] = {}

        # ── Weaver ────────────────────────────────────────────────────
        self.weaver = Weaver(
            vector_store=self.vector_store,
            embed_fn=self.embed_fn,
            graph_create_event=self._graph_create_event,
            graph_update_event=self._graph_update_event,
            graph_delete_event=self._graph_delete_event,
            code_vector_store=self.code_vector_store,
            graph_create_annotation=self._graph_create_annotation,
        )

        # ── Build graph ───────────────────────────────────────────────
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Neo4j callable wrappers (injected into Judge + Weaver)
    # ------------------------------------------------------------------

    async def _graph_event_search_wrapper(
        self, event_name: str, user_id: str, top_k: int = 1,
    ) -> List[SearchResult]:
        """Bridge Neo4j search results → SearchResult for the Judge."""
        loop = asyncio.get_running_loop()
        from functools import partial
        raw = await loop.run_in_executor(
            None,
            partial(
                self.neo4j.search_events_by_name,
                event_name=event_name,
                user_id=user_id,
                top_k=top_k,
            )
        )
        results: List[SearchResult] = []
        for r in raw:
            content = (
                f"{r.get('date', '')} | {r.get('event_name', '')} | {r.get('desc', '')}"
            )
            results.append(
                SearchResult(
                    id=f"{r.get('date', '')}_{r.get('event_name', '')}",
                    content=content,
                    score=1.0,
                    metadata=r,
                )
            )
        return results

    async def _graph_create_event(
        self, user_id: str, date_str: str, event_data: Dict[str, Any],
    ) -> None:
        loop = asyncio.get_running_loop()
        from functools import partial
        await loop.run_in_executor(
            None,
            partial(
                self.neo4j.create_event,
                user_id=user_id,
                date_str=date_str,
                event_data=event_data,
            )
        )

    async def _graph_update_event(
        self, user_id: str, date_str: str, event_data: Dict[str, Any],
    ) -> None:
        loop = asyncio.get_running_loop()
        from functools import partial
        await loop.run_in_executor(
            None,
            partial(
                self.neo4j.update_event,
                user_id=user_id,
                date_str=date_str,
                event_data=event_data,
            )
        )

    async def _graph_delete_event(
        self, user_id: str, embedding_id: str = "", **kwargs,
    ) -> None:
        # embedding_id for temporal is "date_str_event_name"
        parts = embedding_id.split("_", 1)
        date_str = parts[0] if parts else ""
        event_name = parts[1] if len(parts) > 1 else None
        
        loop = asyncio.get_running_loop()
        from functools import partial
        await loop.run_in_executor(
            None,
            partial(
                self.neo4j.delete_event,
                user_id=user_id,
                date_str=date_str,
                event_name=event_name,
            )
        )

    async def _graph_create_annotation(
        self,
        content: str,
        annotation_type: str = "explanation",
        severity: Optional[str] = None,
        author_id: Optional[str] = None,
        repo: Optional[str] = None,
        target_file: Optional[str] = None,
        target_symbol: Optional[str] = None,
    ) -> str:
        """Bridge for creating code annotations in the code graph."""
        loop = asyncio.get_running_loop()
        from functools import partial
        return await loop.run_in_executor(
            None,
            partial(
                self.code_graph.create_annotation,
                org_id=self.org_id,
                content=content,
                annotation_type=annotation_type,
                severity=severity,
                author_id=author_id,
                repo=repo,
                target_file=target_file,
                target_symbol=target_symbol,
            )
        )

    # ------------------------------------------------------------------
    # User-scoped snippet store
    # ------------------------------------------------------------------

    def _get_snippet_store(self, user_id: str) -> BaseVectorStore:
        """Get or create a vector store for a user's snippets namespace."""
        if user_id not in self._snippet_stores:
            ns = snippets_namespace(user_id)
            self._snippet_stores[user_id] = get_vector_store(
                namespace=ns,
                create_if_not_exists=False,
            )
            logger.info("Snippet store initialised (ns=%s).", ns)
        return self._snippet_stores[user_id]

    # ------------------------------------------------------------------
    # LangGraph node functions
    # ------------------------------------------------------------------

    async def _node_classify(self, state: IngestState) -> Dict[str, Any]:
        """Run the classifier on the user query."""
        user_query = state.get("user_query", "")
        # Hint the classifier if an image is attached
        if state.get("image_url"):
            user_query += " [User has attached an image]"

        result = await self.classifier.arun({
            "user_query": user_query,
        })
        return {"classification_result": result}

    def _route_after_classify(self, state: IngestState) -> List[Send]:
        """Fan out to extraction agents based on classification."""
        routes: List[Send] = []
        user_id = state.get("user_id", "default")
        user_query = state.get("user_query", "").strip()
        disabled_domains = set(state.get("disabled_domains") or [])

        # Collect queries per domain — merge duplicates so each agent runs once
        profile_queries: List[str] = []
        temporal_queries: List[str] = []
        image_queries: List[str] = []
        code_queries: List[str] = []

        classification_result = state.get("classification_result")
        if classification_result and classification_result.classifications:
            for c in classification_result.classifications:
                if c["source"] == "profile":
                    profile_queries.append(c["query"])
                elif c["source"] == "event":
                    temporal_queries.append(c["query"])
                elif c["source"] == "image":
                    image_queries.append(c["query"])
                elif c["source"] == "code":
                    code_queries.append(c["query"])

        # Determine if we should run the summary extraction
        # Heuristic: Don't summarize tiny acknowledgments or greetings (unless they had classified facts)
        words = user_query.split()
        is_trivial = len(words) < 4 and not any([profile_queries, temporal_queries, code_queries, image_queries])

        if not is_trivial:
            routes.append(Send("extract_summary", {
                **state,
                "user_id": user_id,
            }))
        else:
            logger.info("Skipping summary extraction for trivial query.")

        if profile_queries:
            routes.append(Send("extract_profile", {
                **state,
                "profile_queries": profile_queries,
                "user_id": user_id,
            }))

        if temporal_queries:
            routes.append(Send("extract_temporal", {
                **state,
                "temporal_queries": temporal_queries,
                "user_id": user_id,
            }))

        if code_queries and not {"code", "snippet"}.issubset(disabled_domains):
            # Enterprise users → team annotation extraction (Code Agent)
            # Single users → personal snippet extraction (Snippet Agent)
            # Tier determined by org_id: "default" means single user
            is_enterprise = self.org_id != "default"

            if is_enterprise and "code" not in disabled_domains:
                routes.append(Send("extract_code", {
                    **state,
                    "code_queries": code_queries,
                    "user_id": user_id,
                }))
            elif not is_enterprise and "snippet" not in disabled_domains:
                routes.append(Send("extract_snippet", {
                    **state,
                    "code_queries": code_queries,
                    "user_id": user_id,
                }))

        # Image route
        if state.get("image_url"):
            if not image_queries:
                image_queries.append("Analyze this image for memory-relevant details.")

            combined_query = " ".join(image_queries)
            routes.append(Send("extract_image", {
                **state,
                "classifier_output": combined_query,
                "user_id": user_id,
            }))

        return routes

    # ── Extraction nodes ──────────────────────────────────────────────

    # ── Decoupled helpers ─────────────────────────────────────────────

    async def _extract_profile(self, combined_query: str) -> ProfileResult:
        return await self.profiler.arun({"classifier_output": combined_query})

    async def _judge_profile(
        self, items: list, user_id: str, pending_ops: Optional[List[Operation]] = None
    ) -> JudgeResult:
        return await self.judge.arun_deterministic({
            "domain": "profile",
            "new_items": items,
            "user_id": user_id,
        }, pending_ops=pending_ops)

    async def _weave_profile(self, judge_result: JudgeResult, user_id: str) -> WeaverResult:
        return await self.weaver.execute(
            judge_result=judge_result,
            domain=JudgeDomain.PROFILE,
            user_id=user_id,
        )

    async def _extract_temporal(self, combined_query: str, session_dt: str) -> EventResult:
        return await self.temporal.arun({
            "classifier_output": combined_query,
            "session_datetime": session_dt,
        })

    async def _judge_temporal(
        self, items: list, user_id: str, pending_ops: Optional[List[Operation]] = None
    ) -> JudgeResult:
        return await self.judge.arun_deterministic({
            "domain": "temporal",
            "new_items": items,
            "user_id": user_id,
        }, pending_ops=pending_ops)

    async def _weave_temporal(self, judge_result: JudgeResult, user_id: str) -> WeaverResult:
        return await self.weaver.execute(
            judge_result=judge_result,
            domain=JudgeDomain.TEMPORAL,
            user_id=user_id,
        )

    async def _extract_image(self, state: IngestState) -> ImageResult:
        return await self.image_agent.arun(state)

    async def _extract_code(self, combined_query: str) -> CodeAnnotationResult:
        return await self.code_agent.arun({"classifier_output": combined_query})

    async def _judge_code(
        self, items: list, user_id: str, pending_ops: Optional[List[Operation]] = None
    ) -> JudgeResult:
        return await self.judge.arun({
            "domain": JudgeDomain.CODE,
            "new_items": items,
            "user_id": user_id,
        }, pending_ops=pending_ops)

    async def _weave_code(self, judge_result: JudgeResult, user_id: str) -> WeaverResult:
        return await self.weaver.execute(
            judge_result=judge_result,
            domain=JudgeDomain.CODE,
            user_id=user_id,
        )

    async def _extract_snippet(self, combined_query: str) -> SnippetExtractionResult:
        return await self.snippet_agent.arun({"classifier_output": combined_query})

    async def _judge_snippet(
        self, items: list, user_id: str, pending_ops: Optional[List[Operation]] = None
    ) -> JudgeResult:
        return await self.judge.arun({
            "domain": JudgeDomain.SNIPPET,
            "new_items": items,
            "user_id": user_id,
        }, pending_ops=pending_ops)

    async def _weave_snippet(self, judge_result: JudgeResult, user_id: str) -> WeaverResult:
        self.weaver.snippet_vector_store = self._get_snippet_store(user_id)
        return await self.weaver.execute(
            judge_result=judge_result,
            domain=JudgeDomain.SNIPPET,
            user_id=user_id,
        )

    async def _extract_summary(self, user_query: str, agent_response: str) -> SummaryResult:
        return await self.summarizer.arun({
            "user_query": user_query,
            "agent_response": agent_response,
        })

    async def _judge_summary(
        self, items: list, user_id: str, pending_ops: Optional[List[Operation]] = None
    ) -> JudgeResult:
        return await self.judge.arun({
            "domain": JudgeDomain.SUMMARY,
            "new_items": items,
            "user_id": user_id,
        }, pending_ops=pending_ops)

    async def _weave_summary(self, judge_result: JudgeResult, user_id: str) -> WeaverResult:
        return await self.weaver.execute(
            judge_result=judge_result,
            domain=JudgeDomain.SUMMARY,
            user_id=user_id,
        )

    # ── Extraction nodes ──────────────────────────────────────────────

    async def _node_extract_profile(self, state: IngestState) -> Dict[str, Any]:
        """Extract profile facts from the classifier query."""
        queries = state.get("profile_queries", [])
        user_id = state.get("user_id", "default")

        combined_query = " ".join(queries)
        result = await self._extract_profile(combined_query)

        if result.is_empty:
            return {"status": "no_profile_facts"}

        items = [f.model_dump() for f in result.facts]
        judge_result = await self._judge_profile(items, user_id)

        weaver_result = await self._weave_profile(judge_result, user_id)
        return {
            "profile_result": result,
            "profile_judge": judge_result,
            "profile_weaver": weaver_result,
        }

    async def _node_extract_temporal(self, state: IngestState) -> Dict[str, Any]:
        """Extract temporal events from the classifier query."""
        queries = state.get("temporal_queries", [])
        user_id = state.get("user_id", "default")
        session_dt = state.get("session_datetime", "")

        combined_query = " ".join(queries)
        result = await self._extract_temporal(combined_query, session_dt)

        if result.is_empty:
            return {"status": "no_temporal_event"}

        all_items: List[Dict[str, str]] = []
        for event in result.events:
            all_items.append({
                "date": event.date,
                "event_name": event.event_name or "",
                "desc": event.desc or "",
                "year": event.year or "",
                "time": event.time or "",
                "date_expression": event.date_expression or "",
            })

        judge_result = await self._judge_temporal(all_items, user_id)

        weaver_result = await self._weave_temporal(judge_result, user_id)
        return {
            "temporal_result": result,
            "temporal_judge": judge_result,
            "temporal_weaver": weaver_result,
        }

    async def _node_extract_image(self, state: IngestState) -> Dict[str, Any]:
        """Extract visual observations from the image and store them as summary."""
        user_id = state.get("user_id", "default")

        result = await self._extract_image(state)

        if result.is_empty:
            return {"status": "no_image_observations"}

        items = []
        if result.description:
            items.append(f"[Image] {result.description}")
        for obs in result.observations:
            conf = f" ({obs.confidence})" if obs.confidence else ""
            items.append(f"[Image/{obs.category}] {obs.description}{conf}")

        if not items:
            return {"status": "no_image_observations"}

        judge_result = await self._judge_summary(items, user_id)

        weaver_result = await self._weave_summary(judge_result, user_id)

        return {
            "image_result": result,
            "image_judge": judge_result,
            "image_weaver": weaver_result,
        }

    async def _node_extract_code(self, state: IngestState) -> Dict[str, Any]:
        """Extract code annotations from the classifier query."""
        queries = state.get("code_queries", [])
        user_id = state.get("user_id", "default")

        combined_query = " ".join(queries)
        result = await self._extract_code(combined_query)

        if result.is_empty:
            return {"status": "no_code_annotations"}

        all_items: List[str] = []
        for ann in result.annotations:
            parts = [
                ann.annotation_type.value,
                ann.target_symbol or "",
                ann.target_file or "",
                ann.repo or "",
                ann.severity.value if ann.severity else "",
                ann.content,
            ]
            all_items.append(" | ".join(parts))

        judge_result = await self._judge_code(all_items, user_id)

        weaver_result = await self._weave_code(judge_result, user_id)
        return {
            "code_result": result,
            "code_judge": judge_result,
            "code_weaver": weaver_result,
        }

    async def _node_extract_snippet(self, state: IngestState) -> Dict[str, Any]:
        """Extract personal code snippets from the classifier query (single-user)."""
        queries = state.get("code_queries", [])
        user_id = state.get("user_id", "default")

        combined_query = " ".join(queries)
        result = await self._extract_snippet(combined_query)

        if result.is_empty:
            return {"status": "no_snippets"}

        all_items: List[str] = []
        for snip in result.snippets:
            parts = [
                snip.content,
                snip.code_snippet.replace("\n", "\\n") if snip.code_snippet else "",
                snip.language,
                snip.snippet_type.value,
                ",".join(snip.tags),
            ]
            all_items.append(" | ".join(parts))

        judge_result = await self._judge_snippet(all_items, user_id)

        weaver_result = await self._weave_snippet(judge_result, user_id)
        return {
            "snippet_result": result,
            "snippet_judge": judge_result,
            "snippet_weaver": weaver_result,
        }

    async def _node_extract_summary(self, state: IngestState) -> Dict[str, Any]:
        user_id = state.get("user_id", "default")
        result = await self._extract_summary(
            user_query=state.get("user_query", ""),
            agent_response=state.get("agent_response", ""),
        )
        if result.is_empty:
            return {"status": "no_summary"}

        items = [
            line.lstrip("- •").strip()
            for line in result.summary.strip().splitlines()
            if line.strip() and line.strip() not in ("-", "•")
        ]
        if not items:
            return {"status": "no_summary_items"}

        judge_result = await self._judge_summary(items, user_id)

        weaver_result = await self._weave_summary(judge_result, user_id)
        return {
            "summary_result": result,
            "summary_judge": judge_result,
            "summary_weaver": weaver_result,
        }

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        workflow = StateGraph(IngestState)

        # Nodes
        workflow.add_node("classify", self._node_classify)
        workflow.add_node("extract_profile", self._node_extract_profile)
        workflow.add_node("extract_temporal", self._node_extract_temporal)
        workflow.add_node("extract_summary", self._node_extract_summary)
        workflow.add_node("extract_image", self._node_extract_image)
        workflow.add_node("extract_code", self._node_extract_code)
        workflow.add_node("extract_snippet", self._node_extract_snippet)

        # Edges
        workflow.add_edge(START, "classify")
        workflow.add_conditional_edges(
            "classify",
            self._route_after_classify,
            ["extract_profile", "extract_temporal", "extract_summary",
             "extract_image", "extract_code", "extract_snippet"],
        )

        # All extraction lanes → END
        workflow.add_edge("extract_profile", END)
        workflow.add_edge("extract_temporal", END)
        workflow.add_edge("extract_summary", END)
        workflow.add_edge("extract_image", END)
        workflow.add_edge("extract_code", END)
        workflow.add_edge("extract_snippet", END)

        return workflow.compile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        user_query: str,
        agent_response: str = "",
        user_id: str = "default",
        session_datetime: str = "",
        image_url: str = "",
        effort_level: EffortLevel | str = EffortLevel.LOW,
        disabled_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run the full ingest pipeline.

        Args:
            user_query:       The raw user message.
            agent_response:   The assistant's response (for summary extraction).
            user_id:          User identifier for storage scoping.
            session_datetime: Optional datetime context for temporal events.
            image_url:        URL or base64 data-URI of an attached image.
            effort_level:     ``'low'`` (default) or ``'high'``.
            disabled_domains: Domains to skip during extraction, used by
                              enterprise chat to keep project annotations out
                              of personal memory ingest.

                              * **LOW**  — single pipeline call, fast.
                              * **HIGH** — splits ``user_query`` into
                                overlapping ≈200-token chunks, runs the full
                                pipeline on every chunk **in parallel**, then
                                merges the results.  Ensures nothing is missed
                                in long inputs at the cost of more LLM calls.

        Returns:
            Final LangGraph state dict with all intermediate results.
            In HIGH mode this is the merged state across all chunks.
        """
        effort_cfg: EffortConfig = get_effort_config(effort_level)

        logger.info("=" * 60)
        logger.info("INGEST PIPELINE START  [effort=%s]", effort_cfg.level.value.upper())
        logger.info("  user_query: %s", user_query[:80])
        logger.info("  user_id:    %s", user_id)
        if image_url:
            logger.info(
                "  image_url:  %s",
                image_url[:50] + "..." if len(image_url) > 50 else image_url,
            )
        logger.info("=" * 60)

        # ── HIGH effort: chunk + parallel dispatch ────────────────────
        if (
            effort_cfg.level == EffortLevel.HIGH
            and estimate_tokens(user_query) > effort_cfg.chunk_threshold_tokens
        ):
            result = await self._run_high_effort(
                user_query=user_query,
                agent_response=agent_response,
                user_id=user_id,
                session_datetime=session_datetime,
                image_url=image_url,
                cfg=effort_cfg,
                disabled_domains=disabled_domains,
            )
        else:
            # ── LOW effort (or short input): single pipeline call ─────
            result = await self._invoke_graph(
                user_query=user_query,
                agent_response=agent_response,
                user_id=user_id,
                session_datetime=session_datetime,
                image_url=image_url,
                disabled_domains=disabled_domains,
            )

        logger.info("=" * 60)
        logger.info("INGEST PIPELINE COMPLETE")
        self._log_summary(result)
        logger.info("=" * 60)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _invoke_graph(
        self,
        user_query: str,
        agent_response: str,
        user_id: str,
        session_datetime: str,
        image_url: str,
        disabled_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Single, unconditional pipeline call (LOW path)."""
        initial_state: IngestState = {
            "user_query": user_query,
            "agent_response": agent_response,
            "user_id": user_id,
            "session_datetime": session_datetime,
            "image_url": image_url,
            "disabled_domains": disabled_domains or [],
            "errors": [],
            "status": "running",
        }
        return await self.graph.ainvoke(initial_state)

    async def _process_item_phase_a(self, idx: int, item: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Phase A - Classification and domain extraction concurrently for a single item."""
        user_query = item.get("user_query", "")
        agent_response = item.get("agent_response", "") or "Acknowledged."
        session_dt = item.get("session_datetime", "")
        image_url = item.get("image_url", "")
        disabled_domains = set(item.get("disabled_domains") or [])

        # Run Classifier
        classifier_query = user_query
        if image_url:
            classifier_query += " [User has attached an image]"

        classification_result = await self.classifier.arun({
            "user_query": classifier_query,
        })

        # Collect sub-queries per domain
        profile_queries = []
        temporal_queries = []
        image_queries = []
        code_queries = []

        if classification_result and classification_result.classifications:
            for c in classification_result.classifications:
                if c["source"] == "profile":
                    profile_queries.append(c["query"])
                elif c["source"] == "event":
                    temporal_queries.append(c["query"])
                elif c["source"] == "image":
                    image_queries.append(c["query"])
                elif c["source"] == "code":
                    code_queries.append(c["query"])

        words = user_query.split()
        is_trivial = len(words) < 4 and not any([profile_queries, temporal_queries, code_queries, image_queries])

        tasks = []
        task_names = []

        if not is_trivial:
            tasks.append(self._extract_summary(user_query, agent_response))
            task_names.append("summary")

        if profile_queries:
            combined_profile = " ".join(profile_queries)
            tasks.append(self._extract_profile(combined_profile))
            task_names.append("profile")

        if temporal_queries:
            combined_temporal = " ".join(temporal_queries)
            tasks.append(self._extract_temporal(combined_temporal, session_dt))
            task_names.append("temporal")

        if code_queries and not {"code", "snippet"}.issubset(disabled_domains):
            is_enterprise = self.org_id != "default"
            if is_enterprise and "code" not in disabled_domains:
                combined_code = " ".join(code_queries)
                tasks.append(self._extract_code(combined_code))
                task_names.append("code")
            elif not is_enterprise and "snippet" not in disabled_domains:
                combined_snippet = " ".join(code_queries)
                tasks.append(self._extract_snippet(combined_snippet))
                task_names.append("snippet")

        if image_url:
            if not image_queries:
                image_queries.append("Analyze this image for memory-relevant details.")
            combined_image = " ".join(image_queries)
            image_state = {
                "classifier_output": combined_image,
                "image_url": image_url,
                "user_id": user_id,
            }
            tasks.append(self._extract_image(image_state))
            task_names.append("image")

        extraction_results = await asyncio.gather(*tasks, return_exceptions=True)

        item_state = {
            "user_query": user_query,
            "agent_response": agent_response,
            "user_id": user_id,
            "session_datetime": session_dt,
            "image_url": image_url,
            "disabled_domains": list(disabled_domains),
            "classification_result": classification_result,
            "errors": [],
            "status": "extracted",
        }

        for name, result in zip(task_names, extraction_results):
            if isinstance(result, Exception):
                logger.error(f"Error during {name} extraction for batch item {idx}: {result}")
                item_state["errors"].append(f"{name}_extraction_error: {str(result)}")
            else:
                item_state[f"{name}_result"] = result

        return {"idx": idx, "item_state": item_state}

    async def run_staged_batch(
        self,
        items: List[Dict[str, Any]],
        user_id: str,
    ) -> List[Dict[str, Any]]:
        """Run batch memory ingestion using a staged parallel/sequential hybrid pipeline."""
        logger.info("=" * 60)
        logger.info("RUN STAGED BATCH: %d items", len(items))
        logger.info("=" * 60)

        # Phase A: Concurrently run classification + domain extraction across all items
        phase_a_tasks = [self._process_item_phase_a(idx, item, user_id) for idx, item in enumerate(items)]
        phase_a_outputs = await asyncio.gather(*phase_a_tasks)

        # Phase B: Sequentially run Judge across all items with pending_ops tracking
        pending_ops: List[Operation] = []

        for phase_a_out in phase_a_outputs:
            item_state = phase_a_out["item_state"]
            idx = phase_a_out["idx"]

            judge_tasks = []
            judge_domains = []

            # 1. Profile facts
            profile_res = item_state.get("profile_result")
            if profile_res and not profile_res.is_empty:
                items_data = [f.model_dump() for f in profile_res.facts]
                judge_tasks.append(self._judge_profile(items_data, user_id, pending_ops=pending_ops))
                judge_domains.append("profile")

            # 2. Temporal events
            temporal_res = item_state.get("temporal_result")
            if temporal_res and not temporal_res.is_empty:
                items_data = []
                for event in temporal_res.events:
                    items_data.append({
                        "date": event.date,
                        "event_name": event.event_name or "",
                        "desc": event.desc or "",
                        "year": event.year or "",
                        "time": event.time or "",
                        "date_expression": event.date_expression or "",
                    })
                judge_tasks.append(self._judge_temporal(items_data, user_id, pending_ops=pending_ops))
                judge_domains.append("temporal")

            # 3. Summary (and Image)
            summary_res = item_state.get("summary_result")
            image_res = item_state.get("image_result")

            summary_items = []
            if summary_res and not summary_res.is_empty:
                summary_items.extend([
                    line.lstrip("- •").strip()
                    for line in summary_res.summary.strip().splitlines()
                    if line.strip() and line.strip() not in ("-", "•")
                ])

            if image_res and not image_res.is_empty:
                if image_res.description:
                    summary_items.append(f"[Image] {image_res.description}")
                for obs in image_res.observations:
                    conf = f" ({obs.confidence})" if obs.confidence else ""
                    summary_items.append(f"[Image/{obs.category}] {obs.description}{conf}")

            if summary_items:
                judge_tasks.append(self._judge_summary(summary_items, user_id, pending_ops=pending_ops))
                judge_domains.append("summary")

            # 4. Code annotations
            code_res = item_state.get("code_judge") or item_state.get("code_result")
            # Wait, let's look at the result schema. It's code_result
            code_res = item_state.get("code_result")
            if code_res and not code_res.is_empty:
                items_data = []
                for ann in code_res.annotations:
                    parts = [
                        ann.annotation_type.value,
                        ann.target_symbol or "",
                        ann.target_file or "",
                        ann.repo or "",
                        ann.severity.value if ann.severity else "",
                        ann.content,
                    ]
                    items_data.append(" | ".join(parts))
                judge_tasks.append(self._judge_code(items_data, user_id, pending_ops=pending_ops))
                judge_domains.append("code")

            # 5. Personal code snippets
            snippet_res = item_state.get("snippet_result")
            if snippet_res and not snippet_res.is_empty:
                items_data = []
                for snip in snippet_res.snippets:
                    parts = [
                        snip.content,
                        snip.code_snippet.replace("\n", "\\n") if snip.code_snippet else "",
                        snip.language,
                        snip.snippet_type.value,
                        ",".join(snip.tags),
                    ]
                    items_data.append(" | ".join(parts))
                judge_tasks.append(self._judge_snippet(items_data, user_id, pending_ops=pending_ops))
                judge_domains.append("snippet")

            if judge_tasks:
                judge_results = await asyncio.gather(*judge_tasks, return_exceptions=True)
                for domain_name, jr in zip(judge_domains, judge_results):
                    if isinstance(jr, Exception):
                        logger.error(f"Error during {domain_name} judge for batch item {idx}: {jr}")
                        item_state["errors"].append(f"{domain_name}_judge_error: {str(jr)}")
                    else:
                        item_state[f"{domain_name}_judge"] = jr
                        if domain_name == "summary" and image_res and not image_res.is_empty:
                            item_state["image_judge"] = jr

                        if jr and jr.operations:
                            pending_ops.extend(jr.operations)

        # Phase C: Concurrently run Weaver to write changes in parallel across all items
        weave_tasks = []
        weave_mappings = []

        for phase_a_out in phase_a_outputs:
            item_state = phase_a_out["item_state"]
            idx = phase_a_out["idx"]

            # Profile
            profile_judge = item_state.get("profile_judge")
            if profile_judge:
                weave_tasks.append(self._weave_profile(profile_judge, user_id))
                weave_mappings.append((item_state, "profile_weaver"))

            # Temporal
            temporal_judge = item_state.get("temporal_judge")
            if temporal_judge:
                weave_tasks.append(self._weave_temporal(temporal_judge, user_id))
                weave_mappings.append((item_state, "temporal_weaver"))

            # Summary
            summary_judge = item_state.get("summary_judge")
            if summary_judge:
                weave_tasks.append(self._weave_summary(summary_judge, user_id))
                weave_mappings.append((item_state, "summary_weaver"))

            # Image
            image_judge = item_state.get("image_judge")
            if image_judge and "image_result" in item_state:
                weave_tasks.append(self._weave_summary(image_judge, user_id))
                weave_mappings.append((item_state, "image_weaver"))

            # Code
            code_judge = item_state.get("code_judge")
            if code_judge:
                weave_tasks.append(self._weave_code(code_judge, user_id))
                weave_mappings.append((item_state, "code_weaver"))

            # Snippet
            snippet_judge = item_state.get("snippet_judge")
            if snippet_judge:
                weave_tasks.append(self._weave_snippet(snippet_judge, user_id))
                weave_mappings.append((item_state, "snippet_weaver"))

        if weave_tasks:
            weave_results = await asyncio.gather(*weave_tasks, return_exceptions=True)
            for (item_state, key), wr in zip(weave_mappings, weave_results):
                if isinstance(wr, Exception):
                    logger.error(f"Error during weaving for key {key}: {wr}")
                    item_state["errors"].append(f"{key}_error: {str(wr)}")
                else:
                    item_state[key] = wr

        # Complete all items
        for phase_a_out in phase_a_outputs:
            item_state = phase_a_out["item_state"]
            item_state["status"] = "completed"

        return [out["item_state"] for out in phase_a_outputs]

    async def _run_high_effort(
        self,
        user_query: str,
        agent_response: str,
        user_id: str,
        session_datetime: str,
        image_url: str,
        cfg: EffortConfig,
        disabled_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """HIGH-effort path: chunk user_query -> parallel staged staged batch run -> merge."""
        chunks = chunk_text(
            user_query,
            chunk_size_tokens=cfg.chunk_size_tokens,
            overlap_tokens=cfg.overlap_tokens,
        )

        logger.info(
            "HIGH-effort ingest: %d chunk(s)  "
            "(chunk_size=%d tok, overlap=%d tok, threshold=%d tok)",
            len(chunks),
            cfg.chunk_size_tokens,
            cfg.overlap_tokens,
            cfg.chunk_threshold_tokens,
        )

        batch_items = [
            {
                "user_query": chunk,
                "agent_response": agent_response,
                "user_id": user_id,
                "session_datetime": session_datetime,
                "image_url": image_url if idx == 0 else "",
                "disabled_domains": disabled_domains or [],
            }
            for idx, chunk in enumerate(chunks)
        ]

        chunk_results = await self.run_staged_batch(batch_items, user_id=user_id)

        # ── Merge states ─────────────────────────────────────────────
        merged: Dict[str, Any] = {}
        all_errors: List[str] = []

        for state in chunk_results:
            # Accumulate errors from every chunk.
            all_errors.extend(state.get("errors") or [])

            # For every key, prefer the last non-None value
            for key, value in state.items():
                if key == "errors":
                    continue
                if value is not None:
                    merged[key] = value

        merged["errors"] = all_errors
        merged["status"] = "completed"
        logger.info(
            "HIGH-effort merge complete: %d chunk(s) processed, %d error(s).",
            len(chunk_results),
            len(all_errors),
        )
        return merged

    def run_sync(
        self,
        user_query: str,
        agent_response: str = "",
        user_id: str = "default",
        session_datetime: str = "",
        image_url: str = "",
        effort_level: EffortLevel | str = EffortLevel.LOW,
        disabled_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for run."""
        return asyncio.run(
            self.run(
                user_query,
                agent_response,
                user_id,
                session_datetime,
                image_url,
                effort_level,
                disabled_domains,
            )
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release external connections."""
        if self.neo4j:
            self.neo4j.close()
        if self.code_graph:
            self.code_graph.close()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _log_summary(state: Dict[str, Any]) -> None:
        for domain in ("profile", "temporal", "summary", "image", "code", "snippet"):
            weaver_key = f"{domain}_weaver"
            wr: Optional[WeaverResult] = state.get(weaver_key)
            if wr:
                logger.info(
                    "  %s: %d ops (%d ok, %d skip, %d fail)",
                    domain, wr.total, wr.succeeded, wr.skipped, wr.failed,
                )
            else:
                logger.info("  %s: (not executed)", domain)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

_default_pipeline: Optional[IngestPipeline] = None


def get_ingest_pipeline() -> IngestPipeline:
    """Get or create the default ingest pipeline (singleton)."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = IngestPipeline()
    return _default_pipeline
