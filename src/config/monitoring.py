"""
Sentry SDK initialization and configuration for XMem production monitoring.

Initialises Sentry error tracking and performance monitoring.  Call
``init_sentry()`` once *before* the FastAPI app object is created so the
FastAPI integration is picked up automatically.

Usage::

    from src.config.monitoring import init_sentry
    init_sentry()   # reads DSN + options from Settings
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("xmem.monitoring")

_sentry_initialised = False


def _before_send(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Scrub sensitive values before sending to Sentry."""
    # Remove API keys / passwords from breadcrumbs and event data
    sensitive_keys = {
        "api_key", "apikey", "api-key",
        "password", "secret", "token",
        "authorization", "cookie",
        "pinecone_api_key", "gemini_api_key",
        "claude_api_key", "openai_api_key",
        "openrouter_api_key", "aws_secret_access_key",
        "neo4j_password", "jwt_secret_key",
    }

    def _scrub(data: Any) -> Any:
        if isinstance(data, dict):
            return {
                k: ("***" if k.lower() in sensitive_keys else _scrub(v))
                for k, v in data.items()
            }
        if isinstance(data, list):
            return [_scrub(item) for item in data]
        return data

    return _scrub(event)


def init_sentry() -> None:
    """Initialise the Sentry SDK using values from ``settings``.

    Safe to call multiple times — only the first call has any effect.
    If ``SENTRY_DSN`` is not set the function returns silently.
    """
    global _sentry_initialised
    if _sentry_initialised:
        return

    from src.config import settings

    dsn = getattr(settings, "sentry_dsn", None)
    if not dsn:
        logger.info("Sentry DSN not configured — error tracking disabled.")
        return

    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=dsn,
            send_default_pii=True,
            enable_logs=True,
            traces_sample_rate=getattr(settings, "sentry_traces_sample_rate", 1.0),
            profile_session_sample_rate=getattr(settings, "sentry_profile_sample_rate", 1.0),
            profile_lifecycle="trace",
            before_send=_before_send,
            environment=getattr(settings, "environment", "production"),
            release=f"xmem@1.0.0",
            # Attach server name for multi-instance identification
            server_name=getattr(settings, "sentry_server_name", None),
        )

        _sentry_initialised = True
        logger.info("Sentry SDK initialised (env=%s).", getattr(settings, "environment", "production"))
    except ImportError:
        logger.warning("sentry-sdk not installed — error tracking disabled.")
    except Exception as exc:
        logger.error("Failed to initialise Sentry: %s", exc)


def capture_exception(exc: Exception) -> None:
    """Forward an exception to Sentry (no-op if SDK not initialised)."""
    if not _sentry_initialised:
        return
    try:
        import sentry_sdk
        sentry_sdk.capture_exception(exc)
    except Exception:
        pass


def set_user_context(user_id: str, username: str = "", email: str = "") -> None:
    """Set the current Sentry user context."""
    if not _sentry_initialised:
        return
    try:
        import sentry_sdk
        sentry_sdk.set_user({
            "id": user_id,
            "username": username or user_id,
            "email": email,
        })
    except Exception:
        pass


def add_breadcrumb(message: str, category: str = "pipeline", level: str = "info", data: dict | None = None) -> None:
    """Add a breadcrumb to the current Sentry transaction."""
    if not _sentry_initialised:
        return
    try:
        import sentry_sdk
        sentry_sdk.add_breadcrumb(
            message=message,
            category=category,
            level=level,
            data=data or {},
        )
    except Exception:
        pass
