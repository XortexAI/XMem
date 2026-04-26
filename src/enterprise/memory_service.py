"""Enterprise adapters for user-scoped memory retrieval and ingest."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from src.enterprise.schemas import EnterpriseIngestResult, EnterpriseMemoryContext

logger = logging.getLogger("xmem.enterprise.memory_service")

PipelineFactory = Callable[[], Any]


class EnterpriseMemoryService:
    """Fetch and store user-scoped memory around enterprise chats."""

    def __init__(
        self,
        retrieval_pipeline_factory: Optional[PipelineFactory] = None,
        ingest_pipeline_factory: Optional[PipelineFactory] = None,
        source_limit: int = 5,
    ) -> None:
        self._retrieval_pipeline_factory = retrieval_pipeline_factory
        self._ingest_pipeline_factory = ingest_pipeline_factory
        self.source_limit = source_limit

    async def fetch_relevant_memory(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
    ) -> EnterpriseMemoryContext:
        """Retrieve user-scoped memory sources for enterprise chat context."""
        logger.info("MEMORY SERVICE: fetch_relevant_memory — user=%s, top_k=%d, query=%s",
                     user_id, top_k, query[:100])
        try:
            pipeline = self._get_retrieval_pipeline()
            logger.debug("MEMORY SERVICE: retrieval pipeline obtained — %s",
                          type(pipeline).__name__)
            result = await pipeline.run(
                query=query,
                user_id=user_id,
                top_k=max(1, min(top_k, self.source_limit)),
            )
        except Exception as exc:
            logger.warning("Enterprise memory retrieval failed: %s", exc, exc_info=True)
            return EnterpriseMemoryContext(error=str(exc))

        sources = []
        for source in (result.sources or [])[:self.source_limit]:
            sources.append({
                "domain": source.domain,
                "content": source.content,
                "score": source.score,
                "metadata": source.metadata,
            })

        logger.info("MEMORY SERVICE: fetch_relevant_memory returned %d sources", len(sources))
        return EnterpriseMemoryContext(
            answer=result.answer or "",
            sources=sources,
        )

    async def ingest_conversation(
        self,
        query: str,
        answer_text: str,
        user_id: str,
    ) -> EnterpriseIngestResult:
        """Store non-code personal memory from an enterprise chat turn.

        Code/project knowledge is handled by EnterpriseAnnotationService, so the
        memory ingest call disables code and snippet lanes when supported.
        """
        logger.info("MEMORY SERVICE: ingest_conversation — user=%s, query_len=%d, answer_len=%d",
                     user_id, len(query), len(answer_text))
        if not query.strip() and not answer_text.strip():
            logger.info("MEMORY SERVICE: skipping ingest (empty conversation)")
            return EnterpriseIngestResult(success=False, error="empty conversation")

        try:
            pipeline = self._get_ingest_pipeline()
            await pipeline.run(
                user_query=query,
                agent_response=answer_text or "Acknowledged.",
                user_id=user_id,
                disabled_domains=["code", "snippet"],
            )
            return EnterpriseIngestResult(success=True)
        except TypeError as exc:
            if "disabled_domains" not in str(exc):
                logger.warning("Enterprise memory ingest failed: %s", exc)
                return EnterpriseIngestResult(success=False, error=str(exc))

            try:
                pipeline = self._get_ingest_pipeline()
                await pipeline.run(
                    user_query=query,
                    agent_response=answer_text or "Acknowledged.",
                    user_id=user_id,
                )
                return EnterpriseIngestResult(success=True)
            except Exception as fallback_exc:
                logger.warning("Enterprise memory ingest failed: %s", fallback_exc)
                return EnterpriseIngestResult(
                    success=False,
                    error=str(fallback_exc),
                )
        except Exception as exc:
            logger.warning("Enterprise memory ingest failed: %s", exc)
            return EnterpriseIngestResult(success=False, error=str(exc))

    def _get_retrieval_pipeline(self) -> Any:
        if self._retrieval_pipeline_factory is not None:
            return self._retrieval_pipeline_factory()

        from src.api.dependencies import get_retrieval_pipeline

        return get_retrieval_pipeline()

    def _get_ingest_pipeline(self) -> Any:
        if self._ingest_pipeline_factory is not None:
            return self._ingest_pipeline_factory()

        from src.api.dependencies import get_ingest_pipeline

        return get_ingest_pipeline()
