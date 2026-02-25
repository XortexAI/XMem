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

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

from src.agents.classifier import ClassifierAgent
from src.agents.image import ImageAgent
from src.agents.judge import JudgeAgent
from src.agents.profiler import ProfilerAgent
from src.agents.summarizer import SummarizerAgent
from src.agents.temporal import TemporalAgent
from src.config import settings
from src.graph.neo4j_client import Neo4jClient
from src.graph.schema import setup_constraints
from src.models import get_model, get_vision_model
from src.pipelines.weaver import Weaver
from src.schemas.classification import ClassificationResult
from src.schemas.events import EventResult
from src.schemas.image import ImageResult
from src.schemas.judge import JudgeDomain, JudgeResult, OperationType
from src.schemas.profile import ProfileResult
from src.schemas.summary import SummaryResult
from src.schemas.weaver import WeaverResult
from src.storage.base import BaseVectorStore, SearchResult
from src.storage.pinecone import PineconeVectorStore

logger = logging.getLogger("xmem.pipelines.ingest")


# ---------------------------------------------------------------------------
# Embedding helper — wraps Google GenAI into a simple callable
# ---------------------------------------------------------------------------

from google import genai
from google.genai import types

_embedding_client: Optional[genai.Client] = None


def get_embedding_client() -> genai.Client:
    global _embedding_client
    if _embedding_client is None:
        api_key_to_use = settings.gemini_api_key or None
        _embedding_client = genai.Client(api_key=api_key_to_use) if api_key_to_use else genai.Client()
        logger.info("Loaded embedding client for model: %s", settings.embedding_model)
    return _embedding_client


def embed_text(text: str) -> List[float]:
    """Embed a single text string → list of floats."""
    client = get_embedding_client()
    result = client.models.embed_content(
        model=settings.embedding_model,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=settings.pinecone_dimension)
    )
    [embedding_obj] = result.embeddings
    return embedding_obj.values


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

    # ── classification ────────────────────────────────────────────────
    classification_result: ClassificationResult

    # ── extraction outputs ────────────────────────────────────────────
    profile_result: ProfileResult
    temporal_result: EventResult
    summary_result: SummaryResult
    image_result: ImageResult

    # ── judge outputs ─────────────────────────────────────────────────
    profile_judge: JudgeResult
    temporal_judge: JudgeResult
    summary_judge: JudgeResult
    image_judge: JudgeResult

    # ── weaver outputs ────────────────────────────────────────────────
    profile_weaver: WeaverResult
    temporal_weaver: WeaverResult
    summary_weaver: WeaverResult
    image_weaver: WeaverResult

    # ── metadata ──────────────────────────────────────────────────────
    status: str
    errors: List[str]


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class IngestPipeline:
    """End-to-end ingest pipeline wired with real Pinecone + Neo4j."""

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        neo4j_client: Optional[Neo4jClient] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
    ) -> None:
        # ── Embedding function ────────────────────────────────────────
        self.embed_fn = embed_fn or embed_text

        # ── Pinecone (vector store) ───────────────────────────────────
        if vector_store:
            self.vector_store = vector_store
        else:
            self.vector_store = PineconeVectorStore(
                api_key=settings.pinecone_api_key,
                index_name=settings.pinecone_index_name,
                dimension=settings.pinecone_dimension,
                metric=settings.pinecone_metric,
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
                namespace=settings.pinecone_namespace,
            )
        logger.info("Pinecone vector store initialised.")

        # ── Neo4j (graph store) ───────────────────────────────────────
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

        self.judge = JudgeAgent(
            model=_agent_model("judge"),
            vector_store=self.vector_store,
            graph_event_search=self._graph_event_search_wrapper,
            top_k=3,
        )

        # ── Weaver ────────────────────────────────────────────────────
        self.weaver = Weaver(
            vector_store=self.vector_store,
            embed_fn=self.embed_fn,
            graph_create_event=self._graph_create_event,
            graph_update_event=self._graph_update_event,
            graph_delete_event=self._graph_delete_event,
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
        raw = self.neo4j.search_events_by_name(
            event_name=event_name, user_id=user_id, top_k=top_k,
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
        self.neo4j.create_event(user_id=user_id, date_str=date_str, event_data=event_data)

    async def _graph_update_event(
        self, user_id: str, date_str: str, event_data: Dict[str, Any],
    ) -> None:
        self.neo4j.update_event(user_id=user_id, date_str=date_str, event_data=event_data)

    async def _graph_delete_event(
        self, user_id: str, embedding_id: str = "", **kwargs,
    ) -> None:
        # embedding_id for temporal is "date_str_event_name"
        parts = embedding_id.split("_", 1)
        date_str = parts[0] if parts else ""
        event_name = parts[1] if len(parts) > 1 else None
        self.neo4j.delete_event(user_id=user_id, date_str=date_str, event_name=event_name)

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

        # Summary always runs
        routes.append(Send("extract_summary", {
            **state,
            "user_id": user_id,
        }))

        # Batch profile, temporal, & image queries
        profile_queries: List[str] = []
        temporal_queries: List[str] = []
        image_queries: List[str] = []

        classification_result = state.get("classification_result")
        if classification_result and classification_result.classifications:
            for c in classification_result.classifications:
                if c["source"] == "profile":
                    profile_queries.append(c["query"])
                elif c["source"] == "event":
                    temporal_queries.append(c["query"])
                elif c["source"] == "image":
                    image_queries.append(c["query"])

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

    async def _node_extract_profile(self, state: IngestState) -> Dict[str, Any]:
        """Extract profile facts from all batched profile queries."""
        queries = state.get("profile_queries", [])
        user_id = state.get("user_id", "default")

        all_facts = []
        last_result = None

        sem = asyncio.Semaphore(5)

        async def _run(q: str) -> ProfileResult:
            async with sem:
                return await self.profiler.arun({"classifier_output": q})

        tasks = [_run(q) for q in queries]
        results = await asyncio.gather(*tasks)

        for result in results:
            if not result.is_empty:
                all_facts.extend(result.facts)
                last_result = result

        if not all_facts:
            return {"status": "no_profile_facts"}

        # Judge all facts together for better dedup
        items = [f.model_dump() for f in all_facts]
        judge_result = await self.judge.arun({
            "domain": "profile",
            "new_items": items,
            "user_id": user_id,
        })

        # Weave
        weaver_result = await self.weaver.execute(
            judge_result=judge_result,
            domain=JudgeDomain.PROFILE,
            user_id=user_id,
        )
        return {
            "profile_result": last_result,
            "profile_judge": judge_result,
            "profile_weaver": weaver_result,
        }

    async def _node_extract_temporal(self, state: IngestState) -> Dict[str, Any]:
        """Extract temporal events from all batched temporal queries."""
        queries = state.get("temporal_queries", [])
        user_id = state.get("user_id", "default")
        session_dt = state.get("session_datetime", "")

        all_items: List[Dict[str, str]] = []
        last_result = None

        sem = asyncio.Semaphore(5)

        async def _run(q: str) -> EventResult:
            async with sem:
                return await self.temporal.arun({
                    "classifier_output": q,
                    "session_datetime": session_dt,
                })

        tasks = [_run(q) for q in queries]
        results = await asyncio.gather(*tasks)

        for result in results:
            if not result.is_empty:
                event = result.event
                all_items.append({
                    "date": event.date,
                    "event_name": event.event_name or "",
                    "desc": event.desc or "",
                    "year": event.year or "",
                    "time": event.time or "",
                    "date_expression": event.date_expression or "",
                })
                last_result = result
        if not all_items:
            return {"status": "no_temporal_event"}

        judge_result = await self.judge.arun({
            "domain": "temporal",
            "new_items": all_items,
            "user_id": user_id,
        })

        weaver_result = await self.weaver.execute(
            judge_result=judge_result,
            domain=JudgeDomain.TEMPORAL,
            user_id=user_id,
        )
        return {
            "temporal_result": last_result,
            "temporal_judge": judge_result,
            "temporal_weaver": weaver_result,
        }

    async def _node_extract_image(self, state: IngestState) -> Dict[str, Any]:
        """Extract visual observations from the image and store them as summary."""
        user_id = state.get("user_id", "default")

        # ImageAgent reads classifier_output and image_url from state
        result = await self.image_agent.arun(state)

        if result.is_empty:
            return {"status": "no_image_observations"}

        # Convert observations to list of dicts for Judge
        # items = [obs.model_dump() for obs in result.observations]

        #converted observation of images to summary and stored as summary
        items = []
        if result.description:
            items.append(f"[Image] {result.description}")
        for obs in result.observations:
            conf = f" ({obs.confidence})" if obs.confidence else ""
            items.append(f"[Image/{obs.category}] {obs.description}{conf}")

        if not items:
            return {"status": "no_image_observations"}

        judge_result = await self.judge.arun({
            "domain": JudgeDomain.SUMMARY,
            "new_items": items,
            "user_id": user_id,
        })

        weaver_result = await self.weaver.execute(
            judge_result=judge_result,
            domain=JudgeDomain.SUMMARY,
            user_id=user_id,
        )

        return {
            "image_result": result,
            "image_judge": judge_result,
            "image_weaver": weaver_result,
        }

    async def _node_extract_summary(self, state: IngestState) -> Dict[str, Any]:
        result = await self.summarizer.arun({
            "user_query": state.get("user_query", ""),
            "agent_response": state.get("agent_response", ""),
        })
        if result.is_empty:
            return {"status": "no_summary"}

        # Split bullet summary into individual items
        items = [
            line.lstrip("- •").strip()
            for line in result.summary.strip().splitlines()
            if line.strip() and line.strip() not in ("-", "•")
        ]
        if not items:
            return {"status": "no_summary_items"}

        judge_result = await self.judge.arun({
            "domain": "summary",
            "new_items": items,
            "user_id": state.get("user_id", "default"),
        })

        weaver_result = await self.weaver.execute(
            judge_result=judge_result,
            domain=JudgeDomain.SUMMARY,
            user_id=state.get("user_id", "default"),
        )
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

        # Edges
        workflow.add_edge(START, "classify")
        workflow.add_conditional_edges(
            "classify",
            self._route_after_classify,
            ["extract_profile", "extract_temporal", "extract_summary", "extract_image"],
        )

        # All extraction lanes → END
        workflow.add_edge("extract_profile", END)
        workflow.add_edge("extract_temporal", END)
        workflow.add_edge("extract_summary", END)
        workflow.add_edge("extract_image", END)

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
    ) -> Dict[str, Any]:
        """Run the full ingest pipeline.

        Args:
            user_query: The raw user message.
            agent_response: The assistant's response (for summary extraction).
            user_id: User identifier for storage scoping.
            session_datetime: Optional datetime context for temporal events.
            image_url: URL or base64 data-URI of an attached image.

        Returns:
            Final LangGraph state dict with all intermediate results.
        """
        initial_state: IngestState = {
            "user_query": user_query,
            "agent_response": agent_response,
            "user_id": user_id,
            "session_datetime": session_datetime,
            "image_url": image_url,
            "errors": [],
            "status": "running",
        }

        logger.info("=" * 60)
        logger.info("INGEST PIPELINE START")
        logger.info("  user_query: %s", user_query[:80])
        logger.info("  user_id:    %s", user_id)
        if image_url:
            logger.info("  image_url:  %s", image_url[:50] + "..." if len(image_url) > 50 else image_url)
        logger.info("=" * 60)

        result = await self.graph.ainvoke(initial_state)

        logger.info("=" * 60)
        logger.info("INGEST PIPELINE COMPLETE")
        self._log_summary(result)
        logger.info("=" * 60)

        return result

    def run_sync(
        self,
        user_query: str,
        agent_response: str = "",
        user_id: str = "default",
        session_datetime: str = "",
        image_url: str = "",
    ) -> Dict[str, Any]:
        """Synchronous wrapper for run."""
        return asyncio.run(
            self.run(user_query, agent_response, user_id, session_datetime, image_url)
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release external connections."""
        if self.neo4j:
            self.neo4j.close()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _log_summary(state: Dict[str, Any]) -> None:
        for domain in ("profile", "temporal", "summary", "image"):
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
