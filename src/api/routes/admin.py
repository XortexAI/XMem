"""
/admin/* routes — internal admin dashboard with live logs, analytics, GitHub traffic.

Authentication: simple username/password stored in MongoDB ``admin_users`` collection.
Default credentials are seeded on first boot: admin / admin@123
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import itertools
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.config import settings
from src.config.analytics import analytics

logger = logging.getLogger("xmem.api.admin")

router = APIRouter(prefix="/admin", tags=["admin"])


# ═══════════════════════════════════════════════════════════════════════════
# Admin auth
# ═══════════════════════════════════════════════════════════════════════════

_admin_collection = None
_admin_sessions: Dict[str, Dict[str, Any]] = {}  # token → {user, expires}


def _get_admin_collection():
    global _admin_collection
    if _admin_collection is not None:
        return _admin_collection
    try:
        from pymongo import MongoClient
        client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        db = client[settings.mongodb_database]
        _admin_collection = db["admin_users"]

        # Seed default admin user if collection is empty
        if _admin_collection.count_documents({}) == 0:
            _admin_collection.insert_one({
                "username": "admin",
                "password_hash": hashlib.sha256("admin@123".encode()).hexdigest(),
                "role": "superadmin",
                "created_at": datetime.now(timezone.utc),
            })
            logger.info("Seeded default admin user (admin / admin@123).")

        return _admin_collection
    except Exception as exc:
        logger.error("Admin MongoDB connection failed: %s", exc)
        return None


class AdminLoginRequest(BaseModel):
    username: str
    password: str


def _verify_admin_token(request: Request) -> Dict[str, Any]:
    """Validate admin session token from cookie or Authorization header."""
    token = request.cookies.get("xmem_admin_token")
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]

    if not token or token not in _admin_sessions:
        raise HTTPException(status_code=401, detail="Not authenticated")

    session = _admin_sessions[token]
    if datetime.now(timezone.utc) > session["expires"]:
        del _admin_sessions[token]
        raise HTTPException(status_code=401, detail="Session expired")

    return session["user"]


# ═══════════════════════════════════════════════════════════════════════════
# Auth endpoints
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/api/login")
async def admin_login(req: AdminLoginRequest):
    collection = _get_admin_collection()
    if collection is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    pwd_hash = hashlib.sha256(req.password.encode()).hexdigest()
    user = collection.find_one({"username": req.username, "password_hash": pwd_hash})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Generate session token
    token = hashlib.sha256(f"{req.username}{time.time()}".encode()).hexdigest()
    _admin_sessions[token] = {
        "user": {"username": user["username"], "role": user.get("role", "admin")},
        "expires": datetime.now(timezone.utc) + timedelta(hours=24),
    }

    response = JSONResponse({"status": "ok", "token": token, "username": user["username"]})
    response.set_cookie(
        key="xmem_admin_token",
        value=token,
        httponly=True,
        max_age=86400,
        samesite="lax",
    )
    return response


@router.post("/api/logout")
async def admin_logout(request: Request):
    token = request.cookies.get("xmem_admin_token")
    if token and token in _admin_sessions:
        del _admin_sessions[token]
    response = JSONResponse({"status": "ok"})
    response.delete_cookie("xmem_admin_token")
    return response


# ═══════════════════════════════════════════════════════════════════════════
# Live log streaming (WebSocket)
# ═══════════════════════════════════════════════════════════════════════════

# Ring buffer of recent log records
_log_counter = itertools.count()
_log_buffer: deque[Dict[str, Any]] = deque(maxlen=500)
_ws_clients: List[WebSocket] = []


class WebSocketLogHandler(logging.Handler):
    """Logging handler that pushes records to connected WebSocket clients."""

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "id": next(_log_counter),
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_text:
            entry["exc"] = record.exc_text

        _log_buffer.append(entry)

        # Broadcast to all connected WebSocket clients
        # Use call_soon_threadsafe to safely schedule from any thread
        for ws in list(_ws_clients):
            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(lambda w=ws, e=entry: asyncio.create_task(_send_log_safe(w, e)))
            except Exception:
                pass


async def _send_log_safe(websocket: WebSocket, entry: Dict[str, Any]) -> None:
    """Safely send a log entry to a WebSocket client."""
    try:
        await websocket.send_json(entry)
    except Exception:
        # Client disconnected or other error — ignore
        pass


# Install the handler on the root logger so ALL logs are captured
_ws_log_handler = WebSocketLogHandler()
_ws_log_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(_ws_log_handler)
# Also capture xmem-specific loggers
logging.getLogger("xmem").addHandler(_ws_log_handler)
logging.getLogger("src").addHandler(_ws_log_handler)
logging.getLogger("uvicorn").addHandler(_ws_log_handler)
# Explicitly capture AWS boto3 logs at INFO level
logging.getLogger("boto3").setLevel(logging.INFO)
logging.getLogger("botocore").setLevel(logging.INFO)
logging.getLogger("boto3").addHandler(_ws_log_handler)
logging.getLogger("botocore").addHandler(_ws_log_handler)


@router.websocket("/ws/logs")
async def ws_live_logs(websocket: WebSocket):
    """WebSocket endpoint for live log streaming."""
    await websocket.accept()

    # Validate auth token from query param
    token = websocket.query_params.get("token", "")
    if token not in _admin_sessions:
        await websocket.close(code=4001, reason="Not authenticated")
        return

    _ws_clients.append(websocket)

    try:
        last_id = -1
        # Send buffered logs first
        for entry in list(_log_buffer):
            await websocket.send_json(entry)
            last_id = entry.get("id", -1)

        # Keep alive — wait for disconnect or new logs
        while True:
            # Send any new logs that arrived in the buffer
            for entry in list(_log_buffer):
                entry_id = entry.get("id", -1)
                if entry_id > last_id:
                    await websocket.send_json(entry)
                    last_id = entry_id

            try:
                # Sleep briefly, check if client disconnected
                await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive occasionally
                pass
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


# ═══════════════════════════════════════════════════════════════════════════
# Analytics summary API
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/api/analytics/summary")
async def analytics_summary(request: Request, user: dict = Depends(_verify_admin_token)):
    """Return analytics summaries for the dashboard."""
    try:
        from pymongo import MongoClient
        client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=3000)
        db = client[settings.mongodb_database]
        collection = db["analytics"]

        now = datetime.now(timezone.utc)
        last_24h = now - timedelta(hours=24)
        last_7d = now - timedelta(days=7)
        last_30d = now - timedelta(days=30)

        # API call stats (last 24h)
        api_calls_24h = list(collection.aggregate([
            {"$match": {"event": "api_call", "ts": {"$gte": last_24h}}},
            {"$group": {
                "_id": {"path": "$path", "method": "$method"},
                "count": {"$sum": 1},
                "avg_latency": {"$avg": "$latency_ms"},
                "p95_latency": {"$max": "$latency_ms"},  # approximation
                "errors": {"$sum": {"$cond": [{"$gte": ["$status", 400]}, 1, 0]}},
            }},
            {"$sort": {"count": -1}},
        ]))

        # LLM call stats (last 24h)
        llm_stats_24h = list(collection.aggregate([
            {"$match": {"event": "llm_call", "ts": {"$gte": last_24h}}},
            {"$group": {
                "_id": {"provider": "$provider", "model": "$model", "agent": "$agent"},
                "count": {"$sum": 1},
                "total_input_tokens": {"$sum": "$input_tokens"},
                "total_output_tokens": {"$sum": "$output_tokens"},
                "total_tokens": {"$sum": "$total_tokens"},
                "avg_latency": {"$avg": "$latency_ms"},
                "errors": {"$sum": {"$cond": [{"$eq": ["$success", False]}, 1, 0]}},
            }},
            {"$sort": {"count": -1}},
        ]))

        # Hourly request volume (last 24h)
        hourly_volume = list(collection.aggregate([
            {"$match": {"event": "api_call", "ts": {"$gte": last_24h}}},
            {"$group": {
                "_id": {
                    "hour": {"$hour": "$ts"},
                    "day": {"$dayOfMonth": "$ts"},
                },
                "count": {"$sum": 1},
                "errors": {"$sum": {"$cond": [{"$gte": ["$status", 400]}, 1, 0]}},
            }},
            {"$sort": {"_id.day": 1, "_id.hour": 1}},
        ]))

        # Unique users (last 24h)
        unique_users = collection.distinct("user_id", {
            "event": "api_call", "ts": {"$gte": last_24h}, "user_id": {"$ne": ""},
        })

        # Total token usage (last 7d, last 30d)
        token_usage_7d = list(collection.aggregate([
            {"$match": {"event": "llm_call", "ts": {"$gte": last_7d}}},
            {"$group": {
                "_id": None,
                "total_input": {"$sum": "$input_tokens"},
                "total_output": {"$sum": "$output_tokens"},
                "total": {"$sum": "$total_tokens"},
                "call_count": {"$sum": 1},
            }},
        ]))

        # Daily LLM calls (last 7d) for chart
        daily_llm = list(collection.aggregate([
            {"$match": {"event": "llm_call", "ts": {"$gte": last_7d}}},
            {"$group": {
                "_id": {
                    "year": {"$year": "$ts"},
                    "month": {"$month": "$ts"},
                    "day": {"$dayOfMonth": "$ts"},
                },
                "count": {"$sum": 1},
                "tokens": {"$sum": "$total_tokens"},
            }},
            {"$sort": {"_id.year": 1, "_id.month": 1, "_id.day": 1}},
        ]))

        return JSONResponse({
            "api_calls_24h": _bson_safe(api_calls_24h),
            "llm_stats_24h": _bson_safe(llm_stats_24h),
            "hourly_volume": _bson_safe(hourly_volume),
            "unique_users_24h": len(unique_users),
            "token_usage_7d": _bson_safe(token_usage_7d[0] if token_usage_7d else {}),
            "daily_llm_calls": _bson_safe(daily_llm),
        })
    except Exception as exc:
        logger.exception("Analytics summary failed")
        return JSONResponse({"error": str(exc)}, status_code=500)


# ═══════════════════════════════════════════════════════════════════════════
# GitHub traffic API
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/api/github/traffic")
async def github_traffic(request: Request, user: dict = Depends(_verify_admin_token)):
    """Fetch GitHub traffic data (views, clones, referrers, paths)."""
    token = settings.github_token
    owner = settings.github_repo_owner
    repo = settings.github_repo_name

    if not token:
        return JSONResponse({"error": "GITHUB_TOKEN not configured"}, status_code=400)

    # Use Bearer format for OAuth tokens (Fine-grained PATs) or token format for classic PATs
    # Try Bearer first (modern format), fallback logic handles both
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    base_url = f"https://api.github.com/repos/{owner}/{repo}"

    # Create client without SSL_CERT_FILE env interference
    ssl_context = httpx.create_ssl_context(verify=True, trust_env=False)
    async with httpx.AsyncClient(timeout=15, verify=ssl_context) as client:
        results = {}
        endpoints = {
            "views": f"{base_url}/traffic/views",
            "clones": f"{base_url}/traffic/clones",
            "referrers": f"{base_url}/traffic/popular/referrers",
            "paths": f"{base_url}/traffic/popular/paths",
            "stars": f"{base_url}",
        }
        for key, url in endpoints.items():
            try:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    results[key] = resp.json()
                else:
                    # Log detailed error for debugging
                    error_body = resp.text[:500] if resp.text else "No response body"
                    logger.warning(f"GitHub API {key} failed: HTTP {resp.status_code} - {error_body}")
                    results[key] = {"error": f"HTTP {resp.status_code}", "details": error_body}
            except Exception as exc:
                logger.exception(f"GitHub API {key} exception")
                results[key] = {"error": str(exc)}

    return JSONResponse(results)


@router.get("/api/github/config-debug")
async def github_config_debug(request: Request, user: dict = Depends(_verify_admin_token)):
    """Debug endpoint to verify GitHub settings are loaded (does not expose full token)."""
    token = settings.github_token
    owner = settings.github_repo_owner
    repo = settings.github_repo_name

    return JSONResponse({
        "token_configured": bool(token),
        "token_prefix": token[:10] + "..." if token and len(token) > 10 else "N/A",
        "token_length": len(token) if token else 0,
        "owner": owner,
        "repo": repo,
        "full_repo_path": f"{owner}/{repo}" if owner and repo else "N/A",
        "hint": "If token shows as 'N/A', check your .env file has GITHUB_TOKEN and restart the server.",
    })


@router.get("/api/analytics/recent-events")
async def analytics_recent_events(request: Request, user: dict = Depends(_verify_admin_token), limit: int = 20):
    """Debug endpoint to see recent analytics events (including LLM calls)."""
    try:
        from pymongo import MongoClient
        client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=3000)
        db = client[settings.mongodb_database]
        collection = db["analytics"]

        # Get recent events of all types
        events = list(collection.find(
            {},
            {"_id": 0}
        ).sort("ts", -1).limit(limit))

        # Get counts by event type
        event_counts = list(collection.aggregate([
            {"$group": {"_id": "$event", "count": {"$sum": 1}}}
        ]))

        # Get recent LLM calls specifically
        llm_calls = list(collection.find(
            {"event": "llm_call"},
            {"_id": 0}
        ).sort("ts", -1).limit(10))

        return JSONResponse({
            "event_counts": _bson_safe(event_counts),
            "recent_llm_calls": _bson_safe(llm_calls),
            "recent_events": _bson_safe(events),
            "analytics_queue_size": len(getattr(analytics, '_queue', [])),
        })
    except Exception as exc:
        logger.exception("Failed to fetch recent events")
        return JSONResponse({"error": str(exc)}, status_code=500)


# ═══════════════════════════════════════════════════════════════════════════
# Server metrics API
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/api/server/metrics")
async def server_metrics(request: Request, user: dict = Depends(_verify_admin_token)):
    """Return current server metrics (uptime, pipeline status, etc.)."""
    from src.api.dependencies import get_init_error, get_startup_time, is_ready

    uptime = round(time.time() - get_startup_time(), 1) if get_startup_time() else 0

    return JSONResponse({
        "uptime_seconds": uptime,
        "pipelines_ready": is_ready(),
        "init_error": get_init_error(),
        "environment": settings.environment,
        "sentry_enabled": bool(settings.sentry_dsn),
        "prometheus_enabled": settings.enable_prometheus,
        "analytics_enabled": settings.enable_analytics,
    })


# ═══════════════════════════════════════════════════════════════════════════
# Recent logs API (for initial page load)
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/api/logs/recent")
async def recent_logs(request: Request, user: dict = Depends(_verify_admin_token), since_id: int = -1):
    """Return log entries from the ring buffer.

    If ``since_id`` is provided, only entries with id > since_id are returned,
    enabling efficient incremental HTTP polling as a fallback when WebSocket
    connections are blocked by reverse proxies.
    """
    if since_id >= 0:
        entries = [e for e in _log_buffer if e.get("id", -1) > since_id]
    else:
        entries = list(_log_buffer)
    return JSONResponse(entries)


@router.post("/api/analytics/test-llm-track")
async def test_llm_track(request: Request, user: dict = Depends(_verify_admin_token)):
    """Test endpoint to manually trigger an LLM analytics event."""
    import random
    test_providers = ["gemini", "openai", "claude", "bedrock"]
    test_models = ["gemini-2.5-flash", "gpt-4.1-mini", "claude-3-5-sonnet", "nova-lite"]

    analytics.track_llm_call(
        provider=random.choice(test_providers),
        model=random.choice(test_models),
        agent="test-agent",
        latency_ms=123.45,
        input_tokens=random.randint(100, 500),
        output_tokens=random.randint(50, 200),
        total_tokens=random.randint(150, 700),
        success=True,
    )

    return JSONResponse({
        "status": "ok",
        "message": "Test LLM call tracked. Check /admin/api/analytics/summary or /admin/api/analytics/recent-events to verify.",
        "queue_size": len(getattr(analytics, '_queue', [])),
    })


# ═══════════════════════════════════════════════════════════════════════════
# Dashboard HTML
# ═══════════════════════════════════════════════════════════════════════════

@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """Serve the admin dashboard SPA."""
    html_path = Path(__file__).parent.parent.parent.parent / "admin" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Admin dashboard not found</h1>", status_code=404)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _bson_safe(obj: Any) -> Any:
    """Recursively convert BSON-unfriendly types to JSON-safe equivalents."""
    if isinstance(obj, dict):
        return {str(k): _bson_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_bson_safe(item) for item in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "__str__") and not isinstance(obj, (str, int, float, bool)):
        return str(obj)
    return obj
