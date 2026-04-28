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
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
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
_event_loop: Optional[asyncio.AbstractEventLoop] = None


def _set_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Store the main event loop reference for cross-thread access."""
    global _event_loop
    _event_loop = loop


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

        loop = _event_loop
        if loop is None or loop.is_closed():
            return

        for ws in list(_ws_clients):
            try:
                asyncio.run_coroutine_threadsafe(
                    _send_log_safe(ws, entry),
                    loop
                )
            except RuntimeError:
                pass


async def _send_log_safe(websocket: WebSocket, entry: Dict[str, Any]) -> None:
    """Safely send a log entry to a WebSocket client."""
    try:
        await websocket.send_json(entry)
    except Exception:
        pass


# Install the handler on the root logger so ALL logs are captured
_ws_log_handler = WebSocketLogHandler()
_ws_log_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(_ws_log_handler)
logging.getLogger("xmem").addHandler(_ws_log_handler)
logging.getLogger("src").addHandler(_ws_log_handler)
logging.getLogger("uvicorn").addHandler(_ws_log_handler)
logging.getLogger("boto3").setLevel(logging.INFO)
logging.getLogger("botocore").setLevel(logging.INFO)
logging.getLogger("boto3").addHandler(_ws_log_handler)
logging.getLogger("botocore").addHandler(_ws_log_handler)


@router.websocket("/ws/logs")
async def ws_live_logs(websocket: WebSocket):
    """WebSocket endpoint for live log streaming."""
    await websocket.accept()

    # Capture the running event loop so the log handler can broadcast
    _set_event_loop(asyncio.get_running_loop())

    # Validate auth token from query param
    token = websocket.query_params.get("token", "")
    if token not in _admin_sessions:
        await websocket.close(code=4001, reason="Not authenticated")
        return

    _ws_clients.append(websocket)

    try:
        # Send buffered logs first
        for entry in list(_log_buffer):
            await websocket.send_json(entry)

        # Keep alive — the WebSocketLogHandler.emit() pushes new logs
        # via call_soon_threadsafe. We just need to keep the connection
        # open by waiting for client messages (or disconnect).
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                # Send a lightweight ping to detect broken connections
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


# ═══════════════════════════════════════════════════════════════════════════
# System Logs — journalctl subprocess streamed over SSE (Server-Sent Events)
#
# SSE works over plain HTTP — no WebSocket upgrade needed, so nginx/reverse
# proxies that block WS will NOT break this.
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/api/system-logs/stream")
async def sse_system_logs(request: Request, user: dict = Depends(_verify_admin_token)):
    """Stream `journalctl -u xmem -f` output as Server-Sent Events.

    This is far more reliable than WebSocket because SSE works over regular
    HTTP and is not blocked by reverse proxies.  It captures ALL service
    output (stdout, stderr, crashes, OOM kills, etc.) directly from the
    OS journal.
    """

    async def _journal_stream():
        proc: Optional[asyncio.subprocess.Process] = None
        try:
            journal_cmd = [
                "journalctl", "-u", "xmem", "-f",
                "-n", "200", "--no-pager", "-o", "short-iso",
            ]

            # Try without sudo first
            proc = await asyncio.create_subprocess_exec(
                *journal_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Check if we need sudo
            try:
                first_err = await asyncio.wait_for(proc.stderr.readline(), timeout=2)
                err_text = first_err.decode("utf-8", errors="replace").lower()
                if "permission" in err_text or "access" in err_text or "denied" in err_text:
                    proc.terminate()
                    await proc.wait()
                    proc = await asyncio.create_subprocess_exec(
                        "sudo", *journal_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                    )
            except asyncio.TimeoutError:
                pass  # No error = working fine

            assert proc and proc.stdout
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                try:
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=15)
                except asyncio.TimeoutError:
                    # Send SSE keepalive comment to prevent proxy timeout
                    yield ":keepalive\n\n"
                    continue

                if not line:
                    # journalctl exited — send error event and stop
                    yield f"event: error\ndata: journalctl process exited\n\n"
                    break

                text = line.decode("utf-8", errors="replace").rstrip("\n")
                # SSE format: data: <line>\n\n
                yield f"data: {text}\n\n"

        except Exception as exc:
            logger.error("SSE system logs error: %s", exc)
            yield f"event: error\ndata: {exc}\n\n"
        finally:
            if proc and proc.returncode is None:
                try:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=3)
                except Exception:
                    proc.kill()

    return StreamingResponse(
        _journal_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Tell nginx not to buffer
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Gemini pricing (paid tier, per 1M tokens in USD)
# ═══════════════════════════════════════════════════════════════════════════
COST_TABLE: Dict[str, Dict[str, float]] = {
    "gemini-embedding-001":   {"input": 0.15,  "output": 0.00},
    "gemini-2.5-flash-lite":  {"input": 0.10,  "output": 0.40},
    "gemini-2.5-flash":       {"input": 0.15,  "output": 0.60},
    "gemini-2.0-flash":       {"input": 0.10,  "output": 0.40},
    "gemini-2.0-flash-lite":  {"input": 0.075, "output": 0.30},
}
_PER_M = 1_000_000


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for a given model and token counts."""
    key = model.lower().strip()
    prices = COST_TABLE.get(key)
    if prices is None:
        for prefix, p in COST_TABLE.items():
            if key.startswith(prefix):
                prices = p
                break
    if prices is None:
        return 0.0
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / _PER_M


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

        # Compute cost for each LLM stats row
        for row in llm_stats_24h:
            model_name = (row.get("_id") or {}).get("model", "")
            row["cost_usd"] = round(_estimate_cost(
                model_name,
                row.get("total_input_tokens", 0),
                row.get("total_output_tokens", 0),
            ), 6)

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

        # Total token usage (last 7d)
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

        # Per-model cost breakdown (last 7d)
        cost_by_model_7d = list(collection.aggregate([
            {"$match": {"event": "llm_call", "ts": {"$gte": last_7d}}},
            {"$group": {
                "_id": "$model",
                "total_input": {"$sum": "$input_tokens"},
                "total_output": {"$sum": "$output_tokens"},
                "call_count": {"$sum": 1},
            }},
            {"$sort": {"total_input": -1}},
        ]))

        total_cost_7d = 0.0
        for row in cost_by_model_7d:
            model_name = row.get("_id") or ""
            cost = _estimate_cost(model_name, row.get("total_input", 0), row.get("total_output", 0))
            row["cost_usd"] = round(cost, 6)
            total_cost_7d += cost

        if token_usage_7d:
            token_usage_7d[0]["total_cost_usd"] = round(total_cost_7d, 6)

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
            "cost_by_model_7d": _bson_safe(cost_by_model_7d),
            "cost_table": COST_TABLE,
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
