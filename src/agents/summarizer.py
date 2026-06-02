"""
Summarizer Agent — extracts memorable user facts from conversation turns.

Takes a user query and agent response, produces a concise bullet-point
summary of facts worth remembering about the user.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from langchain_core.language_models import BaseChatModel

from src.agents.base import BaseAgent
from src.prompts.summarizer import build_system_prompt, pack_summary_query
from src.schemas.summary import SummaryResult
from src.models.registry import get_model_context_window


class SummarizerAgent(BaseAgent):
    # Dynamic Chunking Configuration
    MAX_RECURSION_DEPTH = 3
    CHUNK_OVERLAP_TOKENS = 200
    SAFE_THRESHOLD_RATIO = 0.8  # Use 80% of context window as safe limit

    # Rate limit retry configuration
    MAX_RETRY_ATTEMPTS = 3
    INITIAL_BACKOFF_SECONDS = 1.0

    def __init__(self, model: BaseChatModel) -> None:
        super().__init__(
            model=model,
            name="summarizer",
            system_prompt=build_system_prompt(),
        )
        # Dynamically determine max chunk tokens from the model's context window
        self._init_dynamic_chunk_tokens()

    def _init_dynamic_chunk_tokens(self) -> None:
        """
        Initialize MAX_CHUNK_TOKENS based on the active model's context window.
        This ensures we stay well within the model's limits across all providers.
        """
        try:
            # Try to detect provider and model name from the model instance
            provider = self._detect_provider()
            model_name = getattr(
                self.model, "model", getattr(self.model, "model_name", None)
            )

            context_window = get_model_context_window(provider, model_name)
            # Calculate safe chunk size at 80% of context window, capped at 12k
            # to prevent "lost in the middle" degradation and output token exhaustion
            self.MAX_CHUNK_TOKENS = min(
                max(int(context_window * self.SAFE_THRESHOLD_RATIO), 2000),
                12000
            )

            self.logger.info(
                f"Initialized dynamic chunking: context_window={context_window}, "
                f"max_chunk_tokens={self.MAX_CHUNK_TOKENS}"
            )
        except Exception as e:
            # Fallback to conservative default
            self.MAX_CHUNK_TOKENS = 3000
            self.logger.warning(
                f"Failed to initialize dynamic chunk tokens, using fallback (3000): {e}"
            )

    def _detect_provider(self) -> str:
        """Detect the provider from the model instance, unwrapping RunnableBinding if needed."""
        # Unwrap RunnableBinding and RunnableWithFallbacks to get the actual model
        model = self.model
        while hasattr(model, "bound"):
            model = model.bound

        model_type = type(model).__name__

        # Map LangChain model classes to providers
        provider_map = {
            "ChatAnthropic": "claude",
            "ChatOpenAI": "openai",
            "ChatGoogleGenerativeAI": "gemini",
            "ChatGroq": "groq",
            "OllamaLLM": "ollama",
            "ChatOllama": "ollama",
            "ChatBedrock": "bedrock",
            "ChatDeepSeek": "deepseek",
            "ChatMimo": "mimo",
        }

        for class_name, provider in provider_map.items():
            if class_name in model_type:
                return provider

        # Default to openai if we can't determine
        return "openai"

    async def _call_model_with_retry(self, messages: list) -> str:
        """
        Call the model with exponential backoff retry logic for rate limits.

        Handles rate limit (429) responses gracefully by retrying with
        exponential backoff instead of failing immediately.
        """
        backoff_seconds = self.INITIAL_BACKOFF_SECONDS

        for attempt in range(self.MAX_RETRY_ATTEMPTS):
            try:
                return await self._call_model(messages)
            except Exception as e:
                error_msg = str(e).lower()
                is_rate_limit = (
                    "429" in error_msg
                    or "rate limit" in error_msg
                    or "quota" in error_msg
                    or "too many requests" in error_msg
                )

                if not is_rate_limit or attempt == self.MAX_RETRY_ATTEMPTS - 1:
                    # Not a rate limit error, or last attempt - raise
                    raise

                self.logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{self.MAX_RETRY_ATTEMPTS}). "
                    f"Retrying in {backoff_seconds:.1f}s..."
                )
                await asyncio.sleep(backoff_seconds)
                backoff_seconds *= 2  # Exponential backoff

    def _estimate_tokens(self, text: str) -> int:
        """Lightweight token estimation (approx 4 characters per token)."""
        return len(text) // 4

    def _chunk_payload(self, text: str) -> list[str]:
        """Splits text into overlapping chunks based on token limits.
        
        Fixes: Uses text.split() for proper whitespace handling and ensures
        overlap calculation doesn't create infinite loops when single words
        exceed MAX_CHUNK_TOKENS.
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_tokens = 0

        for word in words:
            word_tokens = self._estimate_tokens(word + " ")
            if current_tokens + word_tokens > self.MAX_CHUNK_TOKENS and current_chunk:
                # Save the current chunk
                chunks.append(" ".join(current_chunk))

                # Calculate overlap by counting tokens from the end of current_chunk
                overlap_words = []
                overlap_tokens = 0
                for w in reversed(current_chunk):
                    w_tokens = self._estimate_tokens(w + " ")
                    if overlap_tokens + w_tokens > self.CHUNK_OVERLAP_TOKENS:
                        break
                    overlap_words.insert(0, w)
                    overlap_tokens += w_tokens

                # Safety check: ensure overlap is strictly smaller than current_chunk
                # to prevent infinite loops/bloat when single words exceed MAX_CHUNK_TOKENS
                if len(overlap_words) >= len(current_chunk):
                    overlap_words = current_chunk[1:] if len(current_chunk) > 1 else []

                current_chunk = overlap_words + [word]
                current_tokens = sum(
                    self._estimate_tokens(w + " ") for w in current_chunk
                )
            else:
                current_chunk.append(word)
                current_tokens += word_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    async def _recursive_summarize(self, text: str, depth: int = 0) -> str:
        """Stateful graph-based loop to chunk, summarize, and map-reduce."""
        if depth >= self.MAX_RECURSION_DEPTH:
            self.logger.warning(
                f"Max recursion depth ({self.MAX_RECURSION_DEPTH}) reached. Truncating payload."
            )
            messages = self._build_messages(text[: self.MAX_CHUNK_TOKENS * 4])
            return await self._call_model_with_retry(messages)

        estimated_tokens = self._estimate_tokens(text)

        # Base Case: Payload fits safely within the context window
        if estimated_tokens <= self.MAX_CHUNK_TOKENS:
            messages = self._build_messages(text)
            return await self._call_model_with_retry(messages)

        # Recursive Case: Split large payloads and map-reduce
        self.logger.info(
            f"Payload too large ({estimated_tokens} tokens). Splitting into chunks (Depth: {depth})."
        )
        chunks = self._chunk_payload(text)

        # Summarize chunks concurrently to improve performance
        # (avoids sequential processing that causes high latency)
        tasks = []
        for i, chunk in enumerate(chunks):
            self.logger.debug(f"Queuing chunk {i + 1}/{len(chunks)} for concurrent summarization...")
            messages = self._build_messages(chunk)
            tasks.append(self._call_model_with_retry(messages))

        self.logger.debug(f"Processing {len(chunks)} chunks concurrently...")
        chunk_summaries = await asyncio.gather(*tasks)
        chunk_summaries = [s.strip() for s in chunk_summaries]

        # Map-reduce: Combine partial summaries and feed them back into the loop
        aggregated_text = "\n\n--- PARTIAL SUMMARIES ---\n\n".join(chunk_summaries)

        return await self._recursive_summarize(aggregated_text, depth=depth + 1)

    async def arun(
        self,
        state: Dict[str, Any],
    ) -> SummaryResult:
        user_query = state.get("user_query", "")
        agent_response = state.get("agent_response", "")

        if not user_query and not agent_response:
            self.logger.debug("Empty input — returning empty summary.")
            return SummaryResult()

        user_message = pack_summary_query(user_query, agent_response)

        # Route through the new dynamic chunking pipeline
        raw_content = await self._recursive_summarize(user_message)
        summary = raw_content.strip()

        # Treat empty-like responses as no summary
        if summary in ('""', "''", "empty", "(empty)", "(empty string)"):
            summary = ""

        result = SummaryResult(summary=summary)

        if not result.is_empty:
            self.logger.info("=" * 50)
            self.logger.info("Generated Summary:")
            self.logger.info(summary)
            self.logger.info("=" * 50)
        else:
            self.logger.info("No memorable facts extracted (trivial input).")

        return result
