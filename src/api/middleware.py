"""
Production middleware stack for the XMem API.

- Request-ID injection (idempotent tracing)
- Request timing
- Security headers (OWASP best-practice)
- Body-size guard
- Prometheus metrics collection
- Fire-and-forget analytics
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.config import settings

logger = logging.getLogger("xmem.api.middleware")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID and measure wall-clock time."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        # Skip WebSocket connections — BaseHTTPMiddleware breaks WS protocol
        if request.scope.get("type") == "websocket":
            return await call_next(request)

        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(elapsed_ms)

        logger.info(
            "%s %s %s %.0fms %s",
            request.method, request.url.path,
            response.status_code, elapsed_ms, request_id,
        )
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject OWASP-recommended response headers."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.scope.get("type") == "websocket":
            return await call_next(request)

        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        return response


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Content-Length exceeds the configured max."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.scope.get("type") == "websocket":
            return await call_next(request)

        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.max_request_body_bytes:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=413,
                content={
                    "status": "error",
                    "error": (
                        f"Request body too large. "
                        f"Max allowed: {settings.max_request_body_bytes} bytes."
                    ),
                },
            )
        return await call_next(request)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Collect HTTP request metrics for Prometheus."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.scope.get("type") == "websocket":
            return await call_next(request)

        from src.config.metrics import METRICS

        # Normalise path to avoid high-cardinality (collapse IDs)
        path = request.url.path
        method = request.method

        METRICS.http_active_requests.inc()
        start = time.perf_counter()

        try:
            response = await call_next(request)
            elapsed = time.perf_counter() - start

            METRICS.http_requests_total.labels(
                method=method, path=path, status=response.status_code,
            ).inc()
            METRICS.http_request_duration.labels(
                method=method, path=path,
            ).observe(elapsed)

            return response
        except Exception:
            elapsed = time.perf_counter() - start
            METRICS.http_requests_total.labels(
                method=method, path=path, status=500,
            ).inc()
            METRICS.http_request_duration.labels(
                method=method, path=path,
            ).observe(elapsed)
            raise
        finally:
            METRICS.http_active_requests.dec()


class AnalyticsMiddleware(BaseHTTPMiddleware):
    """Fire-and-forget analytics tracking for every API request."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.scope.get("type") == "websocket":
            return await call_next(request)

        from src.config.analytics import analytics

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        # Extract user_id from request state (set by auth middleware)
        user_id = ""
        if hasattr(request.state, "user"):
            user = getattr(request.state, "user", {})
            if isinstance(user, dict):
                user_id = user.get("id", user.get("username", ""))

        request_id = getattr(request.state, "request_id", "")

        # Fire and forget — this just appends to an in-memory deque
        analytics.track_api_call(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=elapsed_ms,
            user_id=user_id,
            request_id=request_id,
        )

        return response
