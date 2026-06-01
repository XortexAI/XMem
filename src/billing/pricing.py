from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Optional

from src.utils import billing as billing_config

TOKENS_PER_MILLION = 1_000_000


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return value.dict()
        except Exception:
            pass
    result: dict[str, Any] = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(value, name)
        except Exception:
            continue
        if callable(attr):
            continue
        if isinstance(attr, (str, int, float, bool, type(None), dict, list, tuple)):
            result[name] = attr
    return result


def _int_at(mapping: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _nested_int(mapping: Mapping[str, Any], container: str, *keys: str) -> int:
    nested = mapping.get(container) or {}
    if not isinstance(nested, Mapping):
        nested = _as_dict(nested)
    return _int_at(nested, *keys)


@dataclass(frozen=True)
class ModelUsage:
    provider: str
    model: str
    agent: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    total_tokens: int = 0
    raw_usage: dict[str, Any] = field(default_factory=dict)
    raw_response_metadata: dict[str, Any] = field(default_factory=dict)

    def total_billable_input_tokens(self) -> int:
        return max(
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cached_input_tokens,
            0,
        )


@dataclass(frozen=True)
class ModelPricing:
    provider: str
    model: str
    input_per_million_usd: float
    output_per_million_usd: float
    cached_input_per_million_usd: Optional[float] = None
    cache_creation_per_million_usd: Optional[float] = None
    thinking_billed_as_output: bool = True
    long_context_threshold: Optional[int] = None
    long_context_input_per_million_usd: Optional[float] = None
    long_context_output_per_million_usd: Optional[float] = None

    def rates_for(self, usage: ModelUsage) -> tuple[float, float]:
        if (
            self.long_context_threshold
            and usage.total_billable_input_tokens() > self.long_context_threshold
        ):
            return (
                self.long_context_input_per_million_usd or self.input_per_million_usd,
                self.long_context_output_per_million_usd or self.output_per_million_usd,
            )
        return self.input_per_million_usd, self.output_per_million_usd


@dataclass(frozen=True)
class CostBreakdown:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    cached_input_tokens: int
    cache_creation_input_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    cache_read_cost_usd: float
    cache_write_cost_usd: float
    total_cost_usd: float
    charged_paise: float
    charged_credits: int
    pricing_source: str

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class ProviderPricingCatalog:
    def __init__(self, entries: Iterable[ModelPricing]) -> None:
        self._entries = {
            (entry.provider.lower(), self._normalize_model(entry.model)): entry
            for entry in entries
        }

    @staticmethod
    def _normalize_model(model: str) -> str:
        return (model or "").strip().lower().replace("_", "-")

    @classmethod
    def _model_aliases(cls, model: str) -> list[str]:
        normalized = cls._normalize_model(model)
        aliases = [normalized]
        if "/" in normalized:
            aliases.append(normalized.rsplit("/", 1)[-1])
        for alias in list(aliases):
            if alias.startswith("claude-"):
                aliases.append(alias.replace(".", "-"))
        return list(dict.fromkeys(alias for alias in aliases if alias))

    def resolve(self, provider: str, model: str) -> Optional[ModelPricing]:
        provider_key = (provider or "").strip().lower()
        model_aliases = self._model_aliases(model)
        if not provider_key or not model_aliases:
            return None
        for model_key in model_aliases:
            direct = self._entries.get((provider_key, model_key))
            if direct:
                return direct

        # LangChain/OpenRouter/model aliases often include date suffixes or
        # provider prefixes. Prefer the longest configured prefix match.
        candidates = [
            entry
            for (entry_provider, configured_model), entry in self._entries.items()
            if entry_provider == provider_key
            and any(
                alias.startswith(configured_model)
                or configured_model.startswith(alias)
                or alias.endswith(configured_model)
                for alias in model_aliases
            )
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: len(item.model), reverse=True)[0]


class UsageNormalizer:
    def normalize(
        self,
        *,
        provider: str,
        model: str,
        agent: str = "",
        response: Any = None,
        usage_metadata: Any = None,
        response_metadata: Any = None,
    ) -> ModelUsage:
        usage = _as_dict(
            usage_metadata
            if usage_metadata is not None
            else getattr(response, "usage_metadata", None)
        )
        metadata = _as_dict(
            response_metadata
            if response_metadata is not None
            else getattr(response, "response_metadata", None)
        )
        token_usage = _as_dict(metadata.get("token_usage") or metadata.get("usage"))
        gemini_usage = _as_dict(metadata.get("usage_metadata"))
        combined = {**token_usage, **gemini_usage, **usage}
        provider_key = self._infer_provider(provider, model)

        if provider_key == "openai":
            return self._normalize_openai(
                provider_key, model, agent, combined, metadata
            )
        if provider_key == "gemini":
            return self._normalize_gemini(
                provider_key, model, agent, combined, metadata
            )
        if provider_key == "claude":
            return self._normalize_claude(
                provider_key, model, agent, combined, metadata
            )
        return self._normalize_generic(provider_key, model, agent, combined, metadata)

    @staticmethod
    def _infer_provider(provider: str, model: str) -> str:
        provider_key = (provider or "").strip().lower()
        model_key = ProviderPricingCatalog._normalize_model(model)
        if model_key.startswith("google/") or model_key.startswith("gemini-"):
            return "gemini"
        if model_key.startswith("anthropic/") or model_key.startswith("claude-"):
            return "claude"
        if model_key.startswith("openai/") or model_key.startswith("gpt-"):
            return "openai"
        return provider_key

    def _normalize_openai(
        self,
        provider: str,
        model: str,
        agent: str,
        usage: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> ModelUsage:
        input_tokens = _int_at(usage, "input_tokens", "prompt_tokens")
        output_tokens = _int_at(usage, "output_tokens", "completion_tokens")
        cached = (
            _nested_int(usage, "input_token_details", "cache_read", "cached_tokens")
            or _nested_int(usage, "prompt_tokens_details", "cached_tokens")
            or _int_at(usage, "cached_tokens", "input_cached_tokens")
        )
        reasoning = (
            _nested_int(usage, "output_token_details", "reasoning")
            or _nested_int(usage, "completion_tokens_details", "reasoning_tokens")
            or _int_at(usage, "reasoning_tokens")
        )
        total = _int_at(usage, "total_tokens") or input_tokens + output_tokens
        return ModelUsage(
            provider=provider,
            model=model,
            agent=agent,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=reasoning,
            cached_input_tokens=cached,
            total_tokens=total,
            raw_usage=dict(usage),
            raw_response_metadata=dict(metadata),
        )

    def _normalize_gemini(
        self,
        provider: str,
        model: str,
        agent: str,
        usage: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> ModelUsage:
        input_tokens = _int_at(
            usage, "input_tokens", "prompt_token_count", "promptTokenCount"
        )
        output_tokens = _int_at(
            usage, "output_tokens", "candidates_token_count", "candidatesTokenCount"
        )
        thinking = _int_at(
            usage, "thinking_tokens", "thoughts_token_count", "thoughtsTokenCount"
        )
        cached = _int_at(
            usage,
            "cached_input_tokens",
            "cached_content_token_count",
            "cachedContentTokenCount",
        )
        total = _int_at(usage, "total_tokens", "total_token_count", "totalTokenCount")
        if not total:
            total = input_tokens + output_tokens + thinking
        return ModelUsage(
            provider=provider,
            model=model,
            agent=agent,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking,
            cached_input_tokens=cached,
            total_tokens=total,
            raw_usage=dict(usage),
            raw_response_metadata=dict(metadata),
        )

    def _normalize_claude(
        self,
        provider: str,
        model: str,
        agent: str,
        usage: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> ModelUsage:
        input_tokens = _int_at(usage, "input_tokens")
        output_tokens = _int_at(usage, "output_tokens")
        cache_creation = _int_at(usage, "cache_creation_input_tokens")
        cache_read = _int_at(usage, "cache_read_input_tokens")
        total = input_tokens + output_tokens + cache_creation + cache_read
        return ModelUsage(
            provider=provider,
            model=model,
            agent=agent,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
            total_tokens=total,
            raw_usage=dict(usage),
            raw_response_metadata=dict(metadata),
        )

    def _normalize_generic(
        self,
        provider: str,
        model: str,
        agent: str,
        usage: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> ModelUsage:
        input_tokens = _int_at(
            usage, "input_tokens", "prompt_tokens", "prompt_token_count"
        )
        output_tokens = _int_at(
            usage, "output_tokens", "completion_tokens", "candidates_token_count"
        )
        total = (
            _int_at(usage, "total_tokens", "total_token_count")
            or input_tokens + output_tokens
        )
        return ModelUsage(
            provider=provider or "unknown",
            model=model,
            agent=agent,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            raw_usage=dict(usage),
            raw_response_metadata=dict(metadata),
        )


class CreditConverter:
    def __init__(
        self,
        *,
        usd_to_inr_rate: float = billing_config.USD_TO_INR_RATE,
        credit_value_paise: float = billing_config.CREDIT_VALUE_PAISE,
        markup_multiplier: float = billing_config.MODEL_COST_MARKUP_MULTIPLIER,
        min_credits: int = billing_config.MIN_MODEL_USAGE_CREDITS,
    ) -> None:
        self.usd_to_inr_rate = usd_to_inr_rate
        self.credit_value_paise = credit_value_paise
        self.markup_multiplier = markup_multiplier
        self.min_credits = min_credits

    def paise_for_cost(self, cost_usd: float) -> float:
        return max(cost_usd, 0.0) * self.usd_to_inr_rate * 100 * self.markup_multiplier

    def credits_for_cost(self, cost_usd: float) -> int:
        if cost_usd <= 0:
            return 0
        return max(
            self.min_credits,
            math.ceil(self.paise_for_cost(cost_usd) / self.credit_value_paise),
        )


class UsageCostCalculator:
    def __init__(
        self,
        catalog: ProviderPricingCatalog | None = None,
        converter: CreditConverter | None = None,
    ) -> None:
        self.catalog = catalog or default_pricing_catalog()
        self.converter = converter or CreditConverter()

    def calculate(self, usage: ModelUsage) -> Optional[CostBreakdown]:
        pricing = self.catalog.resolve(usage.provider, usage.model)
        if not pricing:
            return None

        input_rate, output_rate = pricing.rates_for(usage)
        if usage.provider == "claude":
            cached_tokens = usage.cached_input_tokens
            uncached_input_tokens = usage.input_tokens
        else:
            cached_tokens = min(usage.cached_input_tokens, usage.input_tokens)
            uncached_input_tokens = max(usage.input_tokens - cached_tokens, 0)
        output_tokens = usage.output_tokens
        if pricing.thinking_billed_as_output:
            output_tokens += usage.thinking_tokens

        input_cost = uncached_input_tokens * input_rate / TOKENS_PER_MILLION
        output_cost = output_tokens * output_rate / TOKENS_PER_MILLION
        cache_read_rate = pricing.cached_input_per_million_usd
        cache_write_rate = pricing.cache_creation_per_million_usd
        cache_read_cost = (
            cached_tokens * cache_read_rate / TOKENS_PER_MILLION
            if cache_read_rate is not None
            else cached_tokens * input_rate / TOKENS_PER_MILLION
        )
        cache_write_cost = (
            usage.cache_creation_input_tokens * cache_write_rate / TOKENS_PER_MILLION
            if cache_write_rate is not None
            else usage.cache_creation_input_tokens * input_rate / TOKENS_PER_MILLION
        )
        total_cost = input_cost + output_cost + cache_read_cost + cache_write_cost
        return CostBreakdown(
            provider=usage.provider,
            model=usage.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            thinking_tokens=usage.thinking_tokens,
            cached_input_tokens=cached_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            cache_read_cost_usd=cache_read_cost,
            cache_write_cost_usd=cache_write_cost,
            total_cost_usd=total_cost,
            charged_paise=self.converter.paise_for_cost(total_cost),
            charged_credits=self.converter.credits_for_cost(total_cost),
            pricing_source="provider_usage",
        )

    def aggregate(self, breakdowns: Iterable[CostBreakdown]) -> CostBreakdown:
        items = list(breakdowns)
        total_cost = sum(item.total_cost_usd for item in items)
        return CostBreakdown(
            provider=(
                "mixed"
                if len({item.provider for item in items}) > 1
                else (items[0].provider if items else "unknown")
            ),
            model=(
                "mixed"
                if len({item.model for item in items}) > 1
                else (items[0].model if items else "unknown")
            ),
            input_tokens=sum(item.input_tokens for item in items),
            output_tokens=sum(item.output_tokens for item in items),
            thinking_tokens=sum(item.thinking_tokens for item in items),
            cached_input_tokens=sum(item.cached_input_tokens for item in items),
            cache_creation_input_tokens=sum(
                item.cache_creation_input_tokens for item in items
            ),
            input_cost_usd=sum(item.input_cost_usd for item in items),
            output_cost_usd=sum(item.output_cost_usd for item in items),
            cache_read_cost_usd=sum(item.cache_read_cost_usd for item in items),
            cache_write_cost_usd=sum(item.cache_write_cost_usd for item in items),
            total_cost_usd=total_cost,
            charged_paise=self.converter.paise_for_cost(total_cost),
            charged_credits=self.converter.credits_for_cost(total_cost),
            pricing_source="provider_usage",
        )


def default_pricing_catalog() -> ProviderPricingCatalog:
    return ProviderPricingCatalog(
        [
            # OpenAI pricing, USD per 1M text tokens.
            ModelPricing(
                "openai",
                "gpt-5.5",
                5.00,
                30.00,
                cached_input_per_million_usd=0.50,
                thinking_billed_as_output=False,
                long_context_threshold=270_000,
                long_context_input_per_million_usd=10.00,
                long_context_output_per_million_usd=45.00,
            ),
            ModelPricing(
                "openai",
                "gpt-5.4",
                2.50,
                15.00,
                cached_input_per_million_usd=0.25,
                thinking_billed_as_output=False,
                long_context_threshold=270_000,
                long_context_input_per_million_usd=5.00,
                long_context_output_per_million_usd=22.50,
            ),
            ModelPricing(
                "openai",
                "gpt-5.4-mini",
                0.75,
                4.50,
                cached_input_per_million_usd=0.075,
                thinking_billed_as_output=False,
                long_context_threshold=270_000,
                long_context_input_per_million_usd=1.50,
                long_context_output_per_million_usd=6.75,
            ),
            ModelPricing(
                "openai",
                "gpt-4.1",
                2.00,
                8.00,
                cached_input_per_million_usd=0.50,
                thinking_billed_as_output=False,
            ),
            ModelPricing(
                "openai",
                "gpt-4.1-mini",
                0.40,
                1.60,
                cached_input_per_million_usd=0.10,
                thinking_billed_as_output=False,
            ),
            ModelPricing(
                "openai",
                "gpt-4.1-nano",
                0.10,
                0.40,
                cached_input_per_million_usd=0.025,
                thinking_billed_as_output=False,
            ),
            ModelPricing(
                "openai",
                "gpt-4o",
                2.50,
                10.00,
                cached_input_per_million_usd=1.25,
                thinking_billed_as_output=False,
            ),
            ModelPricing(
                "openai",
                "gpt-4o-mini",
                0.15,
                0.60,
                cached_input_per_million_usd=0.075,
                thinking_billed_as_output=False,
            ),
            ModelPricing(
                "openai",
                "gpt-5",
                1.25,
                10.00,
                cached_input_per_million_usd=0.125,
                thinking_billed_as_output=False,
            ),
            ModelPricing(
                "openai",
                "gpt-5-mini",
                0.25,
                2.00,
                cached_input_per_million_usd=0.025,
                thinking_billed_as_output=False,
            ),
            ModelPricing(
                "openai",
                "gpt-5-nano",
                0.05,
                0.40,
                cached_input_per_million_usd=0.005,
                thinking_billed_as_output=False,
            ),
            # Gemini Developer API standard tier, USD per 1M text/image/video tokens.
            ModelPricing(
                "gemini",
                "gemini-3-flash-preview",
                0.50,
                3.00,
                cached_input_per_million_usd=0.05,
            ),
            ModelPricing("gemini", "gemini-3.1-flash-live-preview", 0.75, 4.50),
            ModelPricing("gemini", "gemini-3.1-flash-image", 0.50, 3.00),
            ModelPricing(
                "gemini",
                "gemini-2.5-flash",
                0.30,
                2.50,
                cached_input_per_million_usd=0.03,
            ),
            ModelPricing(
                "gemini",
                "gemini-2.5-flash-lite",
                0.10,
                0.40,
                cached_input_per_million_usd=0.01,
            ),
            ModelPricing(
                "gemini",
                "gemini-2.5-pro",
                1.25,
                10.00,
                cached_input_per_million_usd=0.125,
                long_context_threshold=200_000,
                long_context_input_per_million_usd=2.50,
                long_context_output_per_million_usd=15.00,
            ),
            # Anthropic pricing, USD per 1M tokens. Cache write defaults to 5m.
            ModelPricing(
                "claude",
                "claude-opus-4.8",
                5.00,
                25.00,
                cached_input_per_million_usd=0.50,
                cache_creation_per_million_usd=6.25,
                thinking_billed_as_output=False,
            ),
            ModelPricing(
                "claude",
                "claude-opus-4.7",
                5.00,
                25.00,
                cached_input_per_million_usd=0.50,
                cache_creation_per_million_usd=6.25,
                thinking_billed_as_output=False,
            ),
            ModelPricing(
                "claude",
                "claude-opus-4.6",
                5.00,
                25.00,
                cached_input_per_million_usd=0.50,
                cache_creation_per_million_usd=6.25,
                thinking_billed_as_output=False,
            ),
            ModelPricing(
                "claude",
                "claude-opus-4.5",
                5.00,
                25.00,
                cached_input_per_million_usd=0.50,
                cache_creation_per_million_usd=6.25,
                thinking_billed_as_output=False,
            ),
            ModelPricing(
                "claude",
                "claude-sonnet-4.6",
                3.00,
                15.00,
                cached_input_per_million_usd=0.30,
                cache_creation_per_million_usd=3.75,
                thinking_billed_as_output=False,
            ),
            ModelPricing(
                "claude",
                "claude-sonnet-4.5",
                3.00,
                15.00,
                cached_input_per_million_usd=0.30,
                cache_creation_per_million_usd=3.75,
                thinking_billed_as_output=False,
            ),
            ModelPricing(
                "claude",
                "claude-haiku-4.5",
                1.00,
                5.00,
                cached_input_per_million_usd=0.10,
                cache_creation_per_million_usd=1.25,
                thinking_billed_as_output=False,
            ),
            ModelPricing(
                "claude",
                "claude-opus-4.1",
                15.00,
                75.00,
                cached_input_per_million_usd=1.50,
                cache_creation_per_million_usd=18.75,
                thinking_billed_as_output=False,
            ),
            ModelPricing(
                "claude",
                "claude-opus-4",
                15.00,
                75.00,
                cached_input_per_million_usd=1.50,
                cache_creation_per_million_usd=18.75,
                thinking_billed_as_output=False,
            ),
            ModelPricing(
                "claude",
                "claude-sonnet-4",
                3.00,
                15.00,
                cached_input_per_million_usd=0.30,
                cache_creation_per_million_usd=3.75,
                thinking_billed_as_output=False,
            ),
            ModelPricing(
                "claude",
                "claude-3-7-sonnet",
                3.00,
                15.00,
                cached_input_per_million_usd=0.30,
                cache_creation_per_million_usd=3.75,
                thinking_billed_as_output=False,
            ),
            ModelPricing(
                "claude",
                "claude-3-5-sonnet",
                3.00,
                15.00,
                cached_input_per_million_usd=0.30,
                cache_creation_per_million_usd=3.75,
                thinking_billed_as_output=False,
            ),
            ModelPricing(
                "claude",
                "claude-3-5-haiku",
                0.80,
                4.00,
                cached_input_per_million_usd=0.08,
                cache_creation_per_million_usd=1.00,
                thinking_billed_as_output=False,
            ),
            ModelPricing(
                "claude",
                "claude-3-haiku",
                0.25,
                1.25,
                cached_input_per_million_usd=0.03,
                cache_creation_per_million_usd=0.30,
                thinking_billed_as_output=False,
            ),
        ]
    )
