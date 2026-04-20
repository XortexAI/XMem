"""
Telemetry endpoint — records clone/install events.

When someone clones the repo and runs `pip install -e .`, the welcome.ts
script fires a POST to /telemetry/clone with hashed auth.  This endpoint
validates the token and writes the user details into the MongoDB "cloners"
collection inside the xmem database.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse

from src.config import settings

logger = logging.getLogger("xmem.api.telemetry")

router = APIRouter(prefix="/telemetry", tags=["telemetry"])

# The expected SHA-256 hash — must match what welcome.ts computes.
# Hash of "xmem-clone-telemetry-key"
EXPECTED_AUTH_HASH = hashlib.sha256(b"xmem-clone-telemetry-key").hexdigest()

# Lazy-init MongoDB connection for the cloners collection
_mongo_client = None
_cloners_collection = None


def _get_cloners_collection():
    """Lazy-connect to MongoDB and return the 'cloners' collection."""
    global _mongo_client, _cloners_collection

    if _cloners_collection is not None:
        return _cloners_collection

    try:
        from pymongo import MongoClient

        _mongo_client = MongoClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=5000,
        )
        # Ping to verify the connection
        _mongo_client.admin.command("ping")

        db = _mongo_client[settings.mongodb_database]  # "xmem"
        _cloners_collection = db["cloners"]

        # Create indexes
        _cloners_collection.create_index("username")
        _cloners_collection.create_index("clonedAt")
        _cloners_collection.create_index(
            [("username", 1), ("hostname", 1)],
            unique=False,
        )

        logger.info("Connected to MongoDB — cloners collection ready.")
        return _cloners_collection
    except Exception as exc:
        logger.error("Failed to connect to MongoDB for cloners: %s", exc)
        return None


@router.post("/clone", summary="Record a repo clone/install event")
async def record_clone(
    request: Request,
    x_clone_auth: Optional[str] = Header(None),
):
    """
    Called by welcome.ts during `pip install -e .`

    Validates the hashed auth token and writes user details
    into the 'cloners' collection in the xmem MongoDB database.
    """

    # ── Auth check ──────────────────────────────────────────────
    if not x_clone_auth or x_clone_auth != EXPECTED_AUTH_HASH:
        return JSONResponse(
            status_code=403,
            content={"error": "Invalid or missing auth token."},
        )

    # ── Parse body ──────────────────────────────────────────────
    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON body."},
        )

    # ── Enrich with server-side timestamp ───────────────────────
    body["serverReceivedAt"] = datetime.utcnow().isoformat()
    body["ip"] = request.client.host if request.client else "unknown"

    # ── Write to MongoDB ────────────────────────────────────────
    collection = _get_cloners_collection()
    if collection is None:
        logger.error("Cloners collection unavailable — skipping write.")
        return JSONResponse(
            status_code=503,
            content={"error": "Database temporarily unavailable."},
        )

    try:
        result = collection.insert_one(body)
        logger.info(
            "Recorded clone event: user=%s, host=%s, id=%s",
            body.get("username", "?"),
            body.get("hostname", "?"),
            result.inserted_id,
        )
        return JSONResponse(
            status_code=201,
            content={"status": "ok", "message": "Clone recorded."},
        )
    except Exception as exc:
        logger.error("Failed to write clone event: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to record clone event."},
        )
