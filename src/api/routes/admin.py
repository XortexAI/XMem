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
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.config import settings

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
_log_buffer: deque[Dict[str, Any]] = deque(maxlen=500)
_ws_clients: List[WebSocket] = []


class WebSocketLogHandler(logging.Handler):
    """Logging handler that pushes records to connected WebSocket clients."""

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_text:
            entry["exc"] = record.exc_text

        _log_buffer.append(entry)

        # Broadcast to all connected WebSocket clients (fire-and-forget)
        for ws in list(_ws_clients):
            try:
                asyncio.get_event_loop().create_task(ws.send_json(entry))
            except Exception:
                pass


# Install the handler on the root logger so ALL logs are captured
_ws_log_handler = WebSocketLogHandler()
_ws_log_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(_ws_log_handler)
# Also capture xmem-specific loggers
logging.getLogger("xmem").addHandler(_ws_log_handler)
logging.getLogger("src").addHandler(_ws_log_handler)
logging.getLogger("uvicorn").addHandler(_ws_log_handler)


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
        # Send buffered logs first
        for entry in list(_log_buffer):
            await websocket.send_json(entry)

        # Keep alive — wait for disconnect
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({"type": "ping"})
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

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    base_url = f"https://api.github.com/repos/{owner}/{repo}"

    async with httpx.AsyncClient(timeout=15) as client:
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
                    results[key] = {"error": f"HTTP {resp.status_code}"}
            except Exception as exc:
                results[key] = {"error": str(exc)}

    return JSONResponse(results)


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
async def recent_logs(request: Request, user: dict = Depends(_verify_admin_token)):
    """Return the last N log entries from the ring buffer."""
    return JSONResponse(list(_log_buffer))


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
