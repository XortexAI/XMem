"""
/admin/* routes — internal admin dashboard with live logs, analytics, GitHub traffic.

Authentication: simple username/password stored in MongoDB ``admin_users`` collection.
Default credentials are seeded on first boot: admin / admin@123
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import smtplib
import threading
import time
import itertools
from collections import deque
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote, unquote

import httpx
from bson.objectid import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
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
# Scanner analytics endpoints
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/api/scanner/analytics")
async def scanner_analytics(request: Request, user: dict = Depends(_verify_admin_token)):
    """Return scanner analytics from all scanner collections."""
    try:
        from pymongo import MongoClient
        client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=3000)
        db = client[settings.mongodb_database]

        # Collection references
        scan_runs = db["scan_runs"]
        scanner_jobs = db["scanner_jobs"]
        scanner_user_repos = db["scanner_user_repos"]
        scanner_index_visibility = db["scanner_index_visibility"]
        scanner_community_stars = db["scanner_community_stars"]

        # 1. Scan Runs stats
        scan_runs_count = scan_runs.count_documents({})
        scan_runs_latest = list(scan_runs.find({}, {"_id": 0, "org_id": 1, "repo": 1, "last_sha": 1, "last_scanned_at": 1, "status": 1})
                                     .sort("last_scanned_at", -1)
                                     .limit(10))

        # 2. Scanner Jobs stats
        scanner_jobs_count = scanner_jobs.count_documents({})
        jobs_by_status = list(scanner_jobs.aggregate([
            {"$group": {"_id": {"phase1": "$phase1_status", "phase2": "$phase2_status"}, "count": {"$sum": 1}}}
        ]))

        recent_jobs = list(scanner_jobs.find(
            {},
            {"_id": 0, "job_id": 1, "username": 1, "org": 1, "repo": 1, "branch": 1,
             "phase1_status": 1, "phase2_status": 1, "updated_at": 1, "error": 1}
        ).sort("updated_at", -1).limit(20))

        # 3. User Repos stats
        user_repos_count = scanner_user_repos.count_documents({})
        repos_per_user = list(scanner_user_repos.aggregate([
            {"$group": {"_id": "$username", "repo_count": {"$sum": 1}}},
            {"$sort": {"repo_count": -1}},
            {"$limit": 10}
        ]))

        recent_repos = list(scanner_user_repos.find(
            {},
            {"_id": 0, "username": 1, "github_org": 1, "repo": 1, "branch": 1, "last_seen_commit": 1}
        ).sort("_id", -1).limit(20))

        # 4. Index Visibility stats
        visibility_count = scanner_index_visibility.count_documents({})
        visibility_breakdown = list(scanner_index_visibility.aggregate([
            {"$group": {"_id": "$is_visible", "count": {"$sum": 1}}}
        ]))

        # 5. Community Stars stats
        stars_count = scanner_community_stars.count_documents({})
        top_starred_repos = list(scanner_community_stars.aggregate([
            {"$group": {"_id": {"org": "$org_id", "repo": "$repo"}, "star_count": {"$sum": 1}}},
            {"$sort": {"star_count": -1}},
            {"$limit": 10}
        ]))

        # Unique users who starred
        unique_stargazers = scanner_community_stars.distinct("username")

        return JSONResponse({
            "scan_runs": {
                "total": scan_runs_count,
                "latest": _bson_safe(scan_runs_latest),
            },
            "scanner_jobs": {
                "total": scanner_jobs_count,
                "by_status": _bson_safe(jobs_by_status),
                "recent": _bson_safe(recent_jobs),
            },
            "user_repos": {
                "total": user_repos_count,
                "per_user": _bson_safe(repos_per_user),
                "recent": _bson_safe(recent_repos),
            },
            "index_visibility": {
                "total": visibility_count,
                "breakdown": _bson_safe(visibility_breakdown),
            },
            "community_stars": {
                "total_stars": stars_count,
                "unique_users": len(unique_stargazers),
                "top_repos": _bson_safe(top_starred_repos),
            },
        })
    except Exception as exc:
        logger.exception("Scanner analytics failed")
        return JSONResponse({"error": str(exc)}, status_code=500)


# ═══════════════════════════════════════════════════════════════════════════
# Users management endpoints
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/api/users")
async def list_users(request: Request, user: dict = Depends(_verify_admin_token)):
    """Return list of all users with their details and API key counts."""
    try:
        from pymongo import MongoClient
        client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=3000)
        db = client[settings.mongodb_database]
        users_collection = db["users"]
        api_keys_collection = db["api_keys"]

        # Fetch all users
        users = list(users_collection.find({}, {
            "_id": 1,
            "email": 1,
            "name": 1,
            "google_id": 1,
            "picture": 1,
            "username": 1,
            "created_at": 1,
            "last_login": 1,
        }).sort("created_at", -1))

        # Get API key counts per user
        user_ids = [str(u["_id"]) for u in users]
        api_key_counts = {}
        if user_ids:
            pipeline = [
                {"$match": {"user_id": {"$in": user_ids}}},
                {"$group": {"_id": "$user_id", "count": {"$sum": 1}}}
            ]
            for doc in api_keys_collection.aggregate(pipeline):
                api_key_counts[doc["_id"]] = doc["count"]

        # Format response
        formatted_users = []
        for u in users:
            user_id = str(u["_id"])
            formatted_users.append({
                "id": user_id,
                "email": u.get("email", ""),
                "name": u.get("name", ""),
                "google_id": u.get("google_id", ""),
                "picture": u.get("picture", ""),
                "username": u.get("username", None),
                "created_at": u.get("created_at"),
                "last_login": u.get("last_login"),
                "api_key_count": api_key_counts.get(user_id, 0),
            })

        return JSONResponse({
            "users": _bson_safe(formatted_users),
            "total_users": len(formatted_users),
        })
    except Exception as exc:
        logger.exception("Failed to fetch users list")
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/api/users/{user_id}/trail")
async def get_user_trail(
    request: Request,
    user_id: str,
    user: dict = Depends(_verify_admin_token),
    hours: int = 24,
    limit: int = 50
):
    """Return API call trail for a specific user from analytics data.

    Args:
        user_id: The user ID to fetch trail for
        hours: How many hours back to look (default: 24)
        limit: Maximum number of trail entries (default: 50)
    """
    try:
        from pymongo import MongoClient
        client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=3000)
        db = client[settings.mongodb_database]
        users_collection = db["users"]
        analytics_collection = db["analytics"]

        # Fetch user details
        user_doc = users_collection.find_one({"_id": user_id})
        if not user_doc:
            # Try finding by string ID
            from bson.objectid import ObjectId
            try:
                user_doc = users_collection.find_one({"_id": ObjectId(user_id)})
            except Exception:
                pass

        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found")

        user_email = user_doc.get("email", "")
        user_name = user_doc.get("name", "")

        # Calculate time range
        now = datetime.now(timezone.utc)
        since = now - timedelta(hours=hours)

        # Fetch recent API calls for this user
        # Try matching by user_id in various formats
        trail_query = {
            "event": "api_call",
            "ts": {"$gte": since},
            "$or": [
                {"user_id": user_id},
                {"user_id": str(user_doc.get("_id", ""))},
                {"user_id": user_doc.get("email", "")},
            ]
        }

        trail = list(analytics_collection.find(
            trail_query,
            {"_id": 0, "path": 1, "method": 1, "status": 1, "latency_ms": 1, "ts": 1, "user_id": 1}
        ).sort("ts", -1).limit(limit))

        # Get unique paths accessed
        unique_paths = list(set(t.get("path", "") for t in trail if t.get("path")))

        # Get total calls in the period
        total_calls = analytics_collection.count_documents(trail_query)

        # Get calls in last 24h specifically
        trail_24h_query = {
            "event": "api_call",
            "ts": {"$gte": now - timedelta(hours=24)},
            "$or": [
                {"user_id": user_id},
                {"user_id": str(user_doc.get("_id", ""))},
                {"user_id": user_doc.get("email", "")},
            ]
        }
        total_calls_24h = analytics_collection.count_documents(trail_24h_query)

        return JSONResponse({
            "user_id": user_id,
            "user_email": user_email,
            "user_name": user_name,
            "trail": _bson_safe(trail),
            "unique_paths": sorted(unique_paths),
            "total_calls_period": total_calls,
            "total_calls_24h": total_calls_24h,
            "period_hours": hours,
        })
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to fetch user trail")
        return JSONResponse({"error": str(exc)}, status_code=500)


# ═══════════════════════════════════════════════════════════════════════════
# Outreach — GitHub email scraper + email sender with tracking
# ═══════════════════════════════════════════════════════════════════════════

_outreach_db = None
_scrape_threads: Dict[str, threading.Thread] = {}
_scrape_stop_events: Dict[str, threading.Event] = {}
_scrape_queues: Dict[str, deque] = {}  # job_id -> deque of new emails for SSE


def _get_outreach_db():
    global _outreach_db
    if _outreach_db is not None:
        return _outreach_db
    try:
        from pymongo import MongoClient
        client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        _outreach_db = client[settings.mongodb_database]
        return _outreach_db
    except Exception as exc:
        logger.error("Outreach MongoDB connection failed: %s", exc)
        return None


# ── PAT Management ────────────────────────────────────────────────────────

class AddPATRequest(BaseModel):
    token: str
    label: str = ""


@router.post("/api/outreach/pats")
async def add_pat(req: AddPATRequest, user: dict = Depends(_verify_admin_token)):
    db = _get_outreach_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    req.token = req.token.strip()
    coll = db["outreach_pats"]
    if coll.find_one({"token": req.token}):
        raise HTTPException(status_code=400, detail="This PAT already exists")

    remaining, reset_at = 5000, None
    headers = {"Authorization": f"token {req.token}", "Accept": "application/vnd.github.v3+json"}
    try:
        resp = httpx.get("https://api.github.com/user", headers=headers, timeout=15)
        if resp.status_code == 200:
            user_info = resp.json()
            logger.info("[outreach] PAT validated for GitHub user: %s", user_info.get("login"))
        elif resp.status_code == 401:
            raise HTTPException(status_code=400, detail=f"Invalid PAT — GitHub says: {resp.json().get('message', 'Bad credentials')}")
        elif resp.status_code == 403:
            raise HTTPException(status_code=400, detail="PAT forbidden — may be IP-restricted or missing scopes")
        else:
            raise HTTPException(status_code=400, detail=f"GitHub returned HTTP {resp.status_code} on /user check")

        # Also test repo access to confirm the PAT can read public repos
        test_resp = httpx.get(
            "https://api.github.com/repos/torvalds/linux/stargazers?per_page=1",
            headers=headers, timeout=15,
        )
        if test_resp.status_code == 401:
            raise HTTPException(status_code=400, detail="PAT is valid for /user but fails on repo access. Use a Classic token with 'repo' scope, or a Fine-grained token with 'All repositories' read access.")
        if test_resp.status_code == 403:
            raise HTTPException(status_code=400, detail="PAT lacks permission to read repository stargazers. Ensure 'repo' scope (classic) or read access to public repos (fine-grained).")

        rl_resp = httpx.get("https://api.github.com/rate_limit", headers=headers, timeout=10)
        if rl_resp.status_code == 200:
            core = rl_resp.json().get("resources", {}).get("core", {})
            remaining = core.get("remaining", 5000)
            reset_ts = core.get("reset")
            if reset_ts:
                reset_at = datetime.fromtimestamp(reset_ts, tz=timezone.utc)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("[outreach] PAT validation error: %s", exc)
        raise HTTPException(status_code=400, detail=f"Could not validate PAT: {exc}")

    doc = {
        "token": req.token,
        "label": req.label or f"PAT-{req.token[-4:]}",
        "remaining": remaining,
        "reset_at": reset_at,
        "added_at": datetime.now(timezone.utc),
        "active": True,
    }
    result = coll.insert_one(doc)
    doc["_id"] = str(result.inserted_id)
    doc["token"] = doc["token"][:8] + "..." + doc["token"][-4:]
    return JSONResponse(_bson_safe(doc))


@router.get("/api/outreach/pats")
async def list_pats(user: dict = Depends(_verify_admin_token)):
    db = _get_outreach_db()
    if db is None:
        return JSONResponse({"pats": []})
    pats = list(db["outreach_pats"].find().sort("added_at", -1))
    for p in pats:
        p["_id"] = str(p["_id"])
        t = p.get("token", "")
        p["token_masked"] = t[:8] + "..." + t[-4:] if len(t) > 12 else "****"
        p["active"] = p.get("active", True)
        del p["token"]
    return JSONResponse({"pats": _bson_safe(pats)})


@router.delete("/api/outreach/pats/all")
async def delete_all_pats(user: dict = Depends(_verify_admin_token)):
    db = _get_outreach_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    result = db["outreach_pats"].delete_many({})
    return JSONResponse({"status": "ok", "deleted": result.deleted_count})


@router.delete("/api/outreach/pats/{pat_id}")
async def delete_pat(pat_id: str, user: dict = Depends(_verify_admin_token)):
    db = _get_outreach_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    result = db["outreach_pats"].delete_one({"_id": ObjectId(pat_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="PAT not found")
    return JSONResponse({"status": "ok"})


def _get_best_pat(db) -> Optional[Dict]:
    """Pick the PAT with highest remaining rate limit."""
    now = datetime.now(timezone.utc)
    pats = list(db["outreach_pats"].find({"active": True}).sort("remaining", -1))
    for p in pats:
        if p.get("remaining", 0) > 0:
            return p
        if p.get("reset_at") and p["reset_at"] <= now:
            db["outreach_pats"].update_one(
                {"_id": p["_id"]}, {"$set": {"remaining": 5000}}
            )
            p["remaining"] = 5000
            return p
    # All PATs have 0 remaining but no reset_at — force a recheck
    for p in pats:
        if not p.get("reset_at"):
            db["outreach_pats"].update_one(
                {"_id": p["_id"]}, {"$set": {"remaining": 5000}}
            )
            p["remaining"] = 5000
            return p
    return None


def _gh_get(url: str, pat_doc: Optional[Dict], db) -> Optional[httpx.Response]:
    """Make a GitHub API GET with rate-limit tracking and rotation."""
    for attempt in range(5):
        if pat_doc is None:
            pat_doc = _get_best_pat(db)
        if pat_doc is None:
            logger.warning("[outreach] _gh_get: no usable PAT available")
            return None
        try:
            token_val = pat_doc["token"].strip()
            resp = httpx.get(
                url,
                headers={
                    "Authorization": f"token {token_val}",
                    "Accept": "application/vnd.github.v3+json",
                },
                timeout=15,
            )
            rl_remaining = resp.headers.get("x-ratelimit-remaining")
            if rl_remaining is not None:
                db["outreach_pats"].update_one(
                    {"_id": pat_doc["_id"]},
                    {"$set": {"remaining": int(rl_remaining)}},
                )

            if resp.status_code == 401:
                logger.warning("[outreach] PAT %s got 401, marking inactive", pat_doc.get("label"))
                db["outreach_pats"].update_one(
                    {"_id": pat_doc["_id"]},
                    {"$set": {"active": False, "remaining": 0}},
                )
                pat_doc = None
                time.sleep(0.5)
                continue

            if resp.status_code in (403, 429):
                reset_ts = resp.headers.get("x-ratelimit-reset")
                reset_at = datetime.fromtimestamp(int(reset_ts), tz=timezone.utc) if reset_ts else None
                db["outreach_pats"].update_one(
                    {"_id": pat_doc["_id"]},
                    {"$set": {"remaining": 0, "reset_at": reset_at}},
                )
                pat_doc = None
                time.sleep(1)
                continue
            return resp
        except Exception as exc:
            logger.warning("[outreach] _gh_get exception: %s", exc)
            time.sleep(1)
            pat_doc = None
    return None


# ── Scraping Jobs ─────────────────────────────────────────────────────────

class StartJobRequest(BaseModel):
    repo_url: str
    target_email_count: int = 500


def _extract_repo_slug(url: str) -> str:
    url = url.strip().rstrip("/")
    if url.startswith("https://github.com/"):
        url = url[len("https://github.com/"):]
    elif url.startswith("http://github.com/"):
        url = url[len("http://github.com/"):]
    parts = url.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return url


def _get_email_for_user(username: str, pat_doc: Dict, db) -> Optional[str]:
    """3-step email discovery: profile -> push events -> recent repo commit."""
    resp = _gh_get(f"https://api.github.com/users/{username}", pat_doc, db)
    if resp and resp.status_code == 200:
        data = resp.json()
        email = data.get("email")
        if email and "noreply" not in email and email.lower().endswith("@gmail.com"):
            return email

    resp = _gh_get(
        f"https://api.github.com/users/{username}/events/public?per_page=15",
        pat_doc, db,
    )
    if resp and resp.status_code == 200:
        for event in resp.json():
            if event.get("type") == "PushEvent":
                for commit in event.get("payload", {}).get("commits", []):
                    email = commit.get("author", {}).get("email")
                    if email and "noreply" not in email and email.lower().endswith("@gmail.com"):
                        return email

    resp = _gh_get(
        f"https://api.github.com/users/{username}/repos?sort=updated&per_page=1",
        pat_doc, db,
    )
    if resp and resp.status_code == 200:
        repos = resp.json()
        if repos and isinstance(repos, list) and repos:
            repo_name = repos[0].get("name")
            if repo_name:
                c_resp = _gh_get(
                    f"https://api.github.com/repos/{username}/{repo_name}/commits?per_page=1",
                    pat_doc, db,
                )
                if c_resp and c_resp.status_code == 200:
                    commits = c_resp.json()
                    if isinstance(commits, list) and commits:
                        email = commits[0].get("commit", {}).get("author", {}).get("email")
                        if email and "noreply" not in email and email.lower().endswith("@gmail.com"):
                            return email
    return None


def _push_status(queue, msg_type: str, message: str, **extra):
    """Push a status/progress event into the SSE queue."""
    if queue is not None:
        event = {"_type": msg_type, "message": message, **extra}
        queue.append(event)


def _scrape_worker(job_id: str, repo_slug: str, target: int, resume_page: int, resume_index: int):
    """Background thread that scrapes stargazer emails."""
    db = _get_outreach_db()
    if db is None:
        return

    stop_event = _scrape_stop_events.get(job_id)
    queue = _scrape_queues.get(job_id)
    jobs_coll = db["outreach_jobs"]
    emails_coll = db["outreach_emails"]

    def _set_job_error(error_msg: str):
        jobs_coll.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {"status": "error", "error": error_msg, "updated_at": datetime.now(timezone.utc)}},
        )
        _push_status(queue, "error", error_msg)

    try:
        stargazers = []
        page = resume_page
        per_page = 100

        _push_status(queue, "status", f"Fetching stargazers for {repo_slug}...")

        while len(stargazers) < 5000:
            if stop_event and stop_event.is_set():
                break
            pat = _get_best_pat(db)
            if pat is None:
                _push_status(queue, "warning", "All PATs exhausted, waiting 10s for rate limit reset...")
                time.sleep(10)
                pat = _get_best_pat(db)
                if pat is None:
                    _set_job_error("All GitHub PATs have exhausted their rate limits. Add more PATs or wait for reset.")
                    _scrape_threads.pop(job_id, None)
                    _scrape_stop_events.pop(job_id, None)
                    return

            resp = _gh_get(
                f"https://api.github.com/repos/{repo_slug}/stargazers?per_page={per_page}&page={page}",
                pat, db,
            )
            if resp is None:
                active_pats = list(db["outreach_pats"].find({"active": True}))
                if not active_pats:
                    _set_job_error("All PATs have been marked inactive (likely invalid/expired). Delete them and add fresh PATs.")
                else:
                    _set_job_error("GitHub API unreachable after multiple retries. Check network or PAT validity.")
                _scrape_threads.pop(job_id, None)
                _scrape_stop_events.pop(job_id, None)
                return
            if resp.status_code == 404:
                _set_job_error(f"Repository '{repo_slug}' not found on GitHub. Check the URL.")
                _scrape_threads.pop(job_id, None)
                _scrape_stop_events.pop(job_id, None)
                return
            if resp.status_code != 200:
                _set_job_error(f"GitHub API returned HTTP {resp.status_code}: {resp.text[:200]}")
                _scrape_threads.pop(job_id, None)
                _scrape_stop_events.pop(job_id, None)
                return

            users = resp.json()
            if not users:
                break
            for u in users:
                login = u.get("login")
                if login:
                    stargazers.append(login)
            page += 1
            jobs_coll.update_one(
                {"_id": ObjectId(job_id)},
                {"$set": {
                    "total_stargazers_fetched": len(stargazers) + resume_index,
                    "last_stargazer_page": page,
                    "updated_at": datetime.now(timezone.utc),
                }},
            )
            _push_status(queue, "progress", f"Fetched {len(stargazers)} stargazers...",
                         stargazers_fetched=len(stargazers))

        if not stargazers:
            _set_job_error(f"No stargazers found for '{repo_slug}'. The repo may have 0 stars or the URL is wrong.")
            _scrape_threads.pop(job_id, None)
            _scrape_stop_events.pop(job_id, None)
            return

        _push_status(queue, "status",
                     f"Found {len(stargazers)} stargazers. Now scanning for emails (target: {target})...")

        existing_usernames = set(
            doc["username"] for doc in emails_coll.find({"job_id": job_id}, {"username": 1})
        )

        found_count = emails_coll.count_documents({"job_id": job_id})
        scanned_count = 0

        consecutive_no_pat = 0
        for i, username in enumerate(stargazers):
            if stop_event and stop_event.is_set():
                break
            if found_count >= target:
                break
            if username in existing_usernames:
                continue

            pat = _get_best_pat(db)
            if pat is None:
                consecutive_no_pat += 1
                if consecutive_no_pat >= 3:
                    _set_job_error("All PATs are inactive or exhausted during email scanning. Add valid PATs.")
                    _scrape_threads.pop(job_id, None)
                    _scrape_stop_events.pop(job_id, None)
                    return
                _push_status(queue, "warning", "No available PAT, waiting...")
                time.sleep(5)
                continue
            consecutive_no_pat = 0
            email = _get_email_for_user(username, pat, db)
            scanned_count += 1

            jobs_coll.update_one(
                {"_id": ObjectId(job_id)},
                {"$set": {
                    "processed_index": resume_index + i + 1,
                    "updated_at": datetime.now(timezone.utc),
                }},
            )

            if email:
                email_doc = {
                    "job_id": job_id,
                    "username": username,
                    "email": email,
                    "scraped_at": datetime.now(timezone.utc),
                    "sent": False,
                    "sent_at": None,
                    "opened": False,
                    "opened_at": None,
                    "clicked": False,
                    "clicked_at": None,
                }
                emails_coll.insert_one(email_doc)
                email_doc["_id"] = str(email_doc["_id"])
                existing_usernames.add(username)
                found_count += 1

                jobs_coll.update_one(
                    {"_id": ObjectId(job_id)},
                    {"$set": {"emails_found": found_count}},
                )

                if queue is not None:
                    queue.append(email_doc)
            else:
                if scanned_count % 10 == 0:
                    _push_status(queue, "progress",
                                 f"Scanned {scanned_count} users, found {found_count} emails so far...",
                                 scanned=scanned_count, found=found_count)

        final_status = "completed" if found_count >= target else "stopped"
        jobs_coll.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {"status": final_status, "emails_found": found_count,
                       "updated_at": datetime.now(timezone.utc)}},
        )
        _push_status(queue, "done",
                     f"Finished. Found {found_count} emails from {scanned_count} scanned users.",
                     found=found_count, scanned=scanned_count, status=final_status)

    except Exception as exc:
        logger.exception("[outreach] Scrape worker crashed for job %s", job_id)
        _set_job_error(f"Scraper crashed: {exc}")

    _scrape_threads.pop(job_id, None)
    _scrape_stop_events.pop(job_id, None)


@router.post("/api/outreach/jobs/start")
async def start_scrape_job(req: StartJobRequest, user: dict = Depends(_verify_admin_token)):
    db = _get_outreach_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    slug = _extract_repo_slug(req.repo_url)
    if "/" not in slug:
        raise HTTPException(status_code=400, detail="Invalid repo URL")

    active_pats = db["outreach_pats"].count_documents({"active": True})
    if active_pats == 0:
        raise HTTPException(status_code=400, detail="No active GitHub PATs available. Delete old/invalid ones and add fresh PATs first.")

    job = {
        "repo_url": req.repo_url,
        "repo_slug": slug,
        "status": "running",
        "total_stargazers_fetched": 0,
        "last_stargazer_page": 1,
        "processed_index": 0,
        "target_email_count": req.target_email_count,
        "emails_found": 0,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    result = db["outreach_jobs"].insert_one(job)
    job_id = str(result.inserted_id)

    stop_event = threading.Event()
    _scrape_stop_events[job_id] = stop_event
    _scrape_queues[job_id] = deque(maxlen=500)

    t = threading.Thread(
        target=_scrape_worker,
        args=(job_id, slug, req.target_email_count, 1, 0),
        daemon=True,
    )
    _scrape_threads[job_id] = t
    t.start()

    job["_id"] = job_id
    return JSONResponse(_bson_safe(job))


@router.post("/api/outreach/jobs/{job_id}/stop")
async def stop_scrape_job(job_id: str, user: dict = Depends(_verify_admin_token)):
    stop_event = _scrape_stop_events.get(job_id)
    if stop_event:
        stop_event.set()
    db = _get_outreach_db()
    if db is not None:
        db["outreach_jobs"].update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {"status": "stopped", "updated_at": datetime.now(timezone.utc)}},
        )
    return JSONResponse({"status": "ok"})


@router.post("/api/outreach/jobs/{job_id}/resume")
async def resume_scrape_job(job_id: str, user: dict = Depends(_verify_admin_token)):
    db = _get_outreach_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    job = db["outreach_jobs"].find_one({"_id": ObjectId(job_id)})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job_id in _scrape_threads and _scrape_threads[job_id].is_alive():
        raise HTTPException(status_code=400, detail="Job is already running")

    active_pats = db["outreach_pats"].count_documents({"active": True})
    if active_pats == 0:
        raise HTTPException(status_code=400, detail="No active PATs available. Delete old ones and add fresh PATs before resuming.")

    db["outreach_jobs"].update_one(
        {"_id": ObjectId(job_id)},
        {"$set": {"status": "running", "updated_at": datetime.now(timezone.utc)}},
    )

    stop_event = threading.Event()
    _scrape_stop_events[job_id] = stop_event
    _scrape_queues[job_id] = deque(maxlen=500)

    t = threading.Thread(
        target=_scrape_worker,
        args=(
            job_id,
            job["repo_slug"],
            job["target_email_count"],
            job.get("last_stargazer_page", 1),
            job.get("processed_index", 0),
        ),
        daemon=True,
    )
    _scrape_threads[job_id] = t
    t.start()

    return JSONResponse({"status": "ok", "message": "Resumed"})


@router.get("/api/outreach/jobs")
async def list_scrape_jobs(user: dict = Depends(_verify_admin_token)):
    db = _get_outreach_db()
    if db is None:
        return JSONResponse({"jobs": []})
    jobs = list(db["outreach_jobs"].find().sort("created_at", -1))
    for j in jobs:
        j["_id"] = str(j["_id"])
        j["is_running"] = j["_id"] in _scrape_threads and _scrape_threads[j["_id"]].is_alive()
    return JSONResponse({"jobs": _bson_safe(jobs)})


@router.get("/api/outreach/jobs/{job_id}/emails")
async def get_job_emails(job_id: str, user: dict = Depends(_verify_admin_token)):
    db = _get_outreach_db()
    if db is None:
        return JSONResponse({"emails": []})
    emails = list(db["outreach_emails"].find({"job_id": job_id}).sort("scraped_at", -1))
    for e in emails:
        e["_id"] = str(e["_id"])
    return JSONResponse({"emails": _bson_safe(emails)})


@router.get("/api/outreach/jobs/{job_id}/stream")
async def stream_job_emails(job_id: str, request: Request, user: dict = Depends(_verify_admin_token)):
    """SSE endpoint for real-time email discovery updates."""
    queue = _scrape_queues.get(job_id)
    if queue is None:
        _scrape_queues[job_id] = deque(maxlen=500)
        queue = _scrape_queues[job_id]

    async def _event_stream():
        while True:
            if await request.is_disconnected():
                break
            if queue:
                entry = queue.popleft()
                entry_type = entry.get("_type", "email")
                if entry_type in ("done", "error"):
                    yield f"event: {entry_type}\ndata: {json.dumps(_bson_safe(entry))}\n\n"
                    if entry_type == "error":
                        break
                elif entry_type in ("status", "progress", "warning"):
                    yield f"event: status\ndata: {json.dumps(_bson_safe(entry))}\n\n"
                else:
                    yield f"data: {json.dumps(_bson_safe(entry))}\n\n"
            else:
                db = _get_outreach_db()
                if db is not None:
                    job = db["outreach_jobs"].find_one({"_id": ObjectId(job_id)})
                    if job and job.get("status") in ("completed", "stopped", "error"):
                        payload = {"status": job["status"]}
                        if job.get("error"):
                            payload["error"] = job["error"]
                        yield f"event: done\ndata: {json.dumps(payload)}\n\n"
                        break
                yield ":keepalive\n\n"
                await asyncio.sleep(1)

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ── Email Drafts & Sending ────────────────────────────────────────────────

class SaveDraftRequest(BaseModel):
    subject: str
    body_html: str


@router.post("/api/outreach/drafts")
async def save_draft(req: SaveDraftRequest, user: dict = Depends(_verify_admin_token)):
    db = _get_outreach_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    coll = db["outreach_drafts"]
    existing = coll.find_one({}, sort=[("created_at", -1)])
    now = datetime.now(timezone.utc)
    if existing:
        coll.update_one(
            {"_id": existing["_id"]},
            {"$set": {"subject": req.subject, "body_html": req.body_html, "updated_at": now}},
        )
        return JSONResponse({"status": "ok", "id": str(existing["_id"])})
    else:
        doc = {"subject": req.subject, "body_html": req.body_html, "created_at": now, "updated_at": now}
        result = coll.insert_one(doc)
        return JSONResponse({"status": "ok", "id": str(result.inserted_id)})


@router.get("/api/outreach/drafts")
async def get_draft(user: dict = Depends(_verify_admin_token)):
    db = _get_outreach_db()
    if db is None:
        return JSONResponse({"subject": "", "body_html": ""})
    coll = db["outreach_drafts"]
    draft = coll.find_one({}, sort=[("created_at", -1)])
    if draft:
        draft["_id"] = str(draft["_id"])
        return JSONResponse(_bson_safe(draft))
    return JSONResponse({"subject": "", "body_html": ""})


class SendEmailsRequest(BaseModel):
    job_id: str
    email_ids: Optional[List[str]] = None


def _inject_tracking(html_body: str, email_doc_id: str, server_url: str) -> str:
    """Inject tracking pixel and rewrite links for click tracking."""
    base = server_url.rstrip("/")
    pixel = f'<img src="{base}/admin/api/outreach/track/open/{email_doc_id}" width="1" height="1" style="display:none" />'

    def _rewrite_link(match):
        original_url = match.group(1)
        if "/outreach/track/" in original_url:
            return match.group(0)
        encoded = quote(original_url, safe="")
        return f'href="{base}/admin/api/outreach/track/click/{email_doc_id}?url={encoded}"'

    tracked_html = re.sub(r'href="([^"]+)"', _rewrite_link, html_body)
    tracked_html += pixel
    return tracked_html


def _send_single_email(to_email: str, subject: str, html_body: str):
    """Send one email via Gmail SMTP."""
    sender = settings.gmail_sender_email
    app_password = settings.gmail_app_password
    if not app_password:
        raise ValueError("GMAIL_APP_PASSWORD not configured")

    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, app_password)
        server.sendmail(sender, to_email, msg.as_string())


@router.post("/api/outreach/send")
async def send_outreach_emails(req: SendEmailsRequest, user: dict = Depends(_verify_admin_token)):
    db = _get_outreach_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    if not settings.gmail_app_password:
        raise HTTPException(status_code=400, detail="GMAIL_APP_PASSWORD not configured in .env")

    draft = db["outreach_drafts"].find_one({}, sort=[("created_at", -1)])
    if not draft:
        raise HTTPException(status_code=400, detail="No draft saved — write a draft first")

    query = {"job_id": req.job_id, "sent": False}
    if req.email_ids:
        query["_id"] = {"$in": [ObjectId(eid) for eid in req.email_ids]}

    emails = list(db["outreach_emails"].find(query))
    if not emails:
        return JSONResponse({"status": "ok", "sent": 0, "message": "No unsent emails to send"})

    server_url = settings.xmem_server_url
    sent_count = 0
    errors = []

    for email_doc in emails:
        eid = str(email_doc["_id"])
        username = email_doc.get("username", "")
        to_email = email_doc["email"]

        subject = draft["subject"].replace("{{username}}", username)
        body = draft["body_html"].replace("{{username}}", username)
        body = _inject_tracking(body, eid, server_url)

        try:
            _send_single_email(to_email, subject, body)
            db["outreach_emails"].update_one(
                {"_id": email_doc["_id"]},
                {"$set": {"sent": True, "sent_at": datetime.now(timezone.utc)}},
            )
            sent_count += 1
            time.sleep(0.5)
        except Exception as exc:
            errors.append({"email": to_email, "error": str(exc)})
            logger.error("Failed to send email to %s: %s", to_email, exc)

    return JSONResponse({
        "status": "ok",
        "sent": sent_count,
        "errors": errors[:10],
        "total_attempted": len(emails),
    })


# ── Tracking Endpoints (public — no auth) ─────────────────────────────────

TRANSPARENT_1PX_GIF = base64.b64decode(
    "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
)


@router.get("/api/outreach/track/open/{email_id}")
async def track_open(email_id: str):
    """Record email open via tracking pixel. No auth required."""
    try:
        db = _get_outreach_db()
        if db is not None:
            db["outreach_emails"].update_one(
                {"_id": ObjectId(email_id), "opened": False},
                {"$set": {"opened": True, "opened_at": datetime.now(timezone.utc)}},
            )
    except Exception:
        pass
    return Response(
        content=TRANSPARENT_1PX_GIF,
        media_type="image/gif",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@router.get("/api/outreach/track/click/{email_id}")
async def track_click(email_id: str, url: str = ""):
    """Record link click and redirect. No auth required."""
    try:
        db = _get_outreach_db()
        if db is not None:
            db["outreach_emails"].update_one(
                {"_id": ObjectId(email_id), "clicked": False},
                {"$set": {"clicked": True, "clicked_at": datetime.now(timezone.utc)}},
            )
    except Exception:
        pass
    redirect_url = unquote(url) if url else "https://xmem.in"
    return Response(
        status_code=302,
        headers={"Location": redirect_url},
    )


# ── Outreach Analytics ────────────────────────────────────────────────────

@router.get("/api/outreach/analytics/{job_id}")
async def outreach_analytics(job_id: str, user: dict = Depends(_verify_admin_token)):
    db = _get_outreach_db()
    if db is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)

    emails = list(db["outreach_emails"].find({"job_id": job_id}))
    total = len(emails)
    sent = sum(1 for e in emails if e.get("sent"))
    opened = sum(1 for e in emails if e.get("opened"))
    clicked = sum(1 for e in emails if e.get("clicked"))

    return JSONResponse({
        "job_id": job_id,
        "total_emails": total,
        "sent": sent,
        "opened": opened,
        "clicked": clicked,
        "open_rate": round(opened / sent * 100, 1) if sent > 0 else 0,
        "click_rate": round(clicked / sent * 100, 1) if sent > 0 else 0,
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
