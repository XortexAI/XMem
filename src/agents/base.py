import logging
from abc import ABC, abstractmethod
from typing import Any, Dict
from dataclasses import dataclass, field
from langchain_core.language_models import BaseChatModel
import time

@dataclass
class BaseAgent(ABC):
    model: BaseChatModel
    name: str
    system_prompt: str = ""
    #Use field(init=False) so 'logger' is NOT required as an argument in __init__
    logger: logging.Logger = field(init=False)
    #Use __post_init__ to set up variables after __init__ is done
    def __post_init__(self):
        self.logger = logging.getLogger(f"xmem.agents.{self.name}")

    @abstractmethod
    async def arun(self, state: Dict[str, Any]) -> Any:
        ...

    def run(self, state: Dict[str, Any]) -> Any:
        import asyncio
        return asyncio.run(self.arun(state))

    def _build_messages(self, user_message: str) -> list:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_message})
        return messages

    async def _call_model(self, messages: list) -> str:
        start = time.perf_counter()
        response = await self.model.ainvoke(messages)
        elapsed = time.perf_counter() - start

        content = response.content
        if isinstance(content, list):
            # Gemini thinking models may return list of dicts like
            # [{'type': 'text', 'text': '...', 'extras': {...}}]
            parts = []
            for c in content:
                if isinstance(c, dict) and "text" in c:
                    parts.append(c["text"])
                elif isinstance(c, str):
                    parts.append(c)
                else:
                    parts.append(str(c))
            content = "\n".join(parts)

        # ── Track LLM call metrics (fire-and-forget) ──────────────────
        self._track_llm_call(response, elapsed)

        return content

    def _track_llm_call(self, response: Any, elapsed: float) -> None:
        """Extract token usage from LLM response and emit metrics + analytics."""
        try:
            # Extract model name and provider
            model_name = getattr(self.model, "model", getattr(self.model, "model_name", "unknown"))
            provider = self._detect_provider()

            input_tokens = 0
            output_tokens = 0
            total_tokens = 0

            # Extract token usage from response metadata (LangChain standard)
            usage = getattr(response, "usage_metadata", None) or {}
            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
            else:
                input_tokens = getattr(usage, "input_tokens", 0)
                output_tokens = getattr(usage, "output_tokens", 0)
                total_tokens = getattr(usage, "total_tokens", 0)

            # Check response_metadata for providers that put it there
            resp_meta = getattr(response, "response_metadata", {}) or {}
            if isinstance(resp_meta, dict):
                # Try various token usage locations
                token_usage = resp_meta.get("token_usage") or resp_meta.get("usage") or {}
                if isinstance(token_usage, dict):
                    input_tokens = input_tokens or token_usage.get("prompt_tokens") or token_usage.get("input_tokens", 0)
                    output_tokens = output_tokens or token_usage.get("completion_tokens") or token_usage.get("output_tokens", 0)
                    total_tokens = total_tokens or token_usage.get("total_tokens", 0)

                # Gemini puts usage in different location
                if not total_tokens and "usage_metadata" in resp_meta:
                    gemini_usage = resp_meta["usage_metadata"]
                    if isinstance(gemini_usage, dict):
                        input_tokens = input_tokens or gemini_usage.get("prompt_token_count", 0)
                        output_tokens = output_tokens or gemini_usage.get("candidates_token_count", 0)
                        total_tokens = total_tokens or gemini_usage.get("total_token_count", 0)

            if not total_tokens and (input_tokens or output_tokens):
                total_tokens = input_tokens + output_tokens

            # ── Prometheus metrics ────────────────────────────────────
            try:
                from src.config.metrics import METRICS
                METRICS.llm_calls_total.labels(
                    provider=provider, model=model_name, agent=self.name,
                ).inc()
                METRICS.llm_latency.labels(
                    provider=provider, model=model_name, agent=self.name,
                ).observe(elapsed)
                if input_tokens:
                    METRICS.llm_tokens_total.labels(
                        provider=provider, model=model_name, agent=self.name, token_type="input",
                    ).inc(input_tokens)
                if output_tokens:
                    METRICS.llm_tokens_total.labels(
                        provider=provider, model=model_name, agent=self.name, token_type="output",
                    ).inc(output_tokens)
            except Exception:
                pass

            # ── Analytics (fire-and-forget) ───────────────────────────
            try:
                from src.config.analytics import analytics
                analytics.track_llm_call(
                    provider=provider,
                    model=model_name,
                    agent=self.name,
                    latency_ms=round(elapsed * 1000, 2),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                )
            except Exception:
                pass

            # ── Sentry breadcrumb ─────────────────────────────────────
            try:
                from src.config.monitoring import add_breadcrumb
                add_breadcrumb(
                    message=f"LLM call: {self.name} → {model_name}",
                    category="llm",
                    data={
                        "provider": provider,
                        "model": model_name,
                        "latency_ms": round(elapsed * 1000, 2),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                )
            except Exception:
                pass

        except Exception:
            pass  # never let tracking break the pipeline

    def _detect_provider(self) -> str:
        """Best-effort provider detection from the model instance."""
        cls_name = type(self.model).__name__.lower()
        if "gemini" in cls_name or "google" in cls_name:
            return "gemini"
        if "claude" in cls_name or "anthropic" in cls_name:
            return "claude"
        if "openai" in cls_name or "chatopen" in cls_name:
            return "openai"
        if "bedrock" in cls_name:
            return "bedrock"
        if "openrouter" in cls_name:
            return "openrouter"
        return "unknown"
