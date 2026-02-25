"""
Production middleware stack for the XMem API.

- Request-ID injection (idempotent tracing)
- Request timing
- Security headers (OWASP best-practice)
- Body-size guard
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
