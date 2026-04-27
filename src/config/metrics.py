"""
Prometheus metric definitions for XMem.

All metrics are lazily created so the module can be imported safely
even when ``prometheus_client`` is not installed.

Usage::

    from src.config.metrics import METRICS
    METRICS.http_requests_total.labels(method="POST", path="/v1/memory/ingest", status=200).inc()
    METRICS.pipeline_duration.labels(pipeline="ingest").observe(1.23)
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("xmem.metrics")

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST

    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False
    logger.info("prometheus_client not installed — metrics disabled.")


class _NoOpMetric:
    """Drop-in stub when prometheus_client is missing."""

    def labels(self, *args, **kwargs):
        return self

    def inc(self, *args, **kwargs):
        pass

    def dec(self, *args, **kwargs):
        pass

    def set(self, *args, **kwargs):
        pass

    def observe(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass


class MetricsRegistry:
    """Singleton holding all Prometheus metrics for XMem."""

    def __init__(self) -> None:
        if _HAS_PROMETHEUS:
            self._init_real()
        else:
            self._init_noop()

    # ── Real metrics ──────────────────────────────────────────────────

    def _init_real(self) -> None:
        # HTTP layer
        self.http_requests_total = Counter(
            "xmem_http_requests_total",
            "Total HTTP requests",
            ["method", "path", "status"],
        )
        self.http_request_duration = Histogram(
            "xmem_http_request_duration_seconds",
            "HTTP request latency",
            ["method", "path"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
        )
        self.http_active_requests = Gauge(
            "xmem_http_active_requests",
            "Number of in-flight HTTP requests",
        )

        # Pipeline layer
        self.pipeline_duration = Histogram(
            "xmem_pipeline_duration_seconds",
            "End-to-end pipeline latency",
            ["pipeline"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
        )
        self.pipeline_stage_duration = Histogram(
            "xmem_pipeline_stage_duration_seconds",
            "Individual pipeline stage latency",
            ["pipeline", "stage"],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
        )
        self.pipeline_errors_total = Counter(
            "xmem_pipeline_errors_total",
            "Pipeline errors",
            ["pipeline", "stage", "error_type"],
        )

        # LLM layer
        self.llm_calls_total = Counter(
            "xmem_llm_calls_total",
            "Total LLM API calls",
            ["provider", "model", "agent"],
        )
        self.llm_latency = Histogram(
            "xmem_llm_latency_seconds",
            "LLM call latency",
            ["provider", "model", "agent"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
        )
        self.llm_tokens_total = Counter(
            "xmem_llm_tokens_total",
            "Total LLM tokens used",
            ["provider", "model", "agent", "token_type"],
        )
        self.llm_errors_total = Counter(
            "xmem_llm_errors_total",
            "LLM call errors",
            ["provider", "model", "agent", "error_type"],
        )

        # Embedding layer
        self.embedding_calls_total = Counter(
            "xmem_embedding_calls_total",
            "Total embedding API calls",
            ["model"],
        )
        self.embedding_latency = Histogram(
            "xmem_embedding_latency_seconds",
            "Embedding call latency",
            ["model"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
        )

        # Weaver layer
        self.weaver_operations_total = Counter(
            "xmem_weaver_operations_total",
            "Weaver operations executed",
            ["domain", "operation", "result"],
        )

        # App info
        self.app_info = Info(
            "xmem_app",
            "XMem application metadata",
        )
        self.app_info.info({
            "version": "1.0.0",
            "service": "xmem-api",
        })

        logger.info("Prometheus metrics initialised.")

    # ── No-op stubs ───────────────────────────────────────────────────

    def _init_noop(self) -> None:
        noop = _NoOpMetric()
        self.http_requests_total = noop
        self.http_request_duration = noop
        self.http_active_requests = noop
        self.pipeline_duration = noop
        self.pipeline_stage_duration = noop
        self.pipeline_errors_total = noop
        self.llm_calls_total = noop
        self.llm_latency = noop
        self.llm_tokens_total = noop
        self.llm_errors_total = noop
        self.embedding_calls_total = noop
        self.embedding_latency = noop
        self.weaver_operations_total = noop
        self.app_info = noop


# Module-level singleton
METRICS = MetricsRegistry()


def metrics_endpoint_content() -> tuple[bytes, str]:
    """Return ``(body, content_type)`` for the ``/metrics`` endpoint."""
    if _HAS_PROMETHEUS:
        return generate_latest(), CONTENT_TYPE_LATEST
    return b"# prometheus_client not installed\n", "text/plain"
