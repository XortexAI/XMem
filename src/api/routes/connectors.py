"""Connector OAuth routes for external knowledge sources."""

from __future__ import annotations

import secrets
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Literal, Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.api.dependencies import require_user
router = APIRouter(prefix="/api/connectors", tags=["Connectors"])

ConnectorId = Literal["notion", "google-drive"]
ConnectorState = Literal["connected", "not_connected", "pending"]

STATE_TTL_MINUTES = 10
MAX_PENDING_STATES = 1000


class ConnectorDefinition(BaseModel):
    id: ConnectorId
    name: str
    description: str
    auth_url: str
    token_url: str
    scopes: List[str]
    docs_url: str


class ConnectorStatusResponse(BaseModel):
    id: ConnectorId
    name: str
    state: ConnectorState
    connected_at: Optional[datetime] = None
    scopes: List[str] = Field(default_factory=list)
    detail: str


class ConnectorListResponse(BaseModel):
    connectors: List[ConnectorStatusResponse]


class ConnectorStartResponse(BaseModel):
    connector_id: ConnectorId
    authorization_url: str
    state: str
    expires_at: datetime


class ConnectorDisconnectResponse(BaseModel):
    connector_id: ConnectorId
    disconnected: bool


class PendingOAuthState(BaseModel):
    connector_id: ConnectorId
    user_id: str
    expires_at: datetime


CONNECTORS: Dict[ConnectorId, ConnectorDefinition] = {
    "notion": ConnectorDefinition(
        id="notion",
        name="Notion",
        description="Sync selected Notion pages and workspace notes into XMem memory.",
        auth_url="https://api.notion.com/v1/oauth/authorize",
        token_url="https://api.notion.com/v1/oauth/token",
        scopes=[],
        docs_url="https://developers.notion.com/docs/authorization",
    ),
    "google-drive": ConnectorDefinition(
        id="google-drive",
        name="Google Drive",
        description="Bring Google Drive docs and files into XMem as searchable memory.",
        auth_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        scopes=[
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/documents.readonly",
        ],
        docs_url="https://developers.google.com/identity/protocols/oauth2",
    ),
}

_pending_states: Dict[str, PendingOAuthState] = {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _client_id(connector_id: ConnectorId) -> Optional[str]:
    if connector_id == "notion":
        return os.getenv("NOTION_CLIENT_ID")
    return os.getenv("GOOGLE_DRIVE_CLIENT_ID") or os.getenv("GOOGLE_CLIENT_ID")


def _redirect_uri(connector_id: ConnectorId) -> str:
    if connector_id == "notion":
        return os.getenv(
            "NOTION_REDIRECT_URI",
            "http://localhost:8000/api/connectors/notion/oauth/callback",
        )
    return os.getenv(
        "GOOGLE_DRIVE_REDIRECT_URI",
        "http://localhost:8000/api/connectors/google-drive/oauth/callback",
    )


def _get_connector(connector_id: str) -> ConnectorDefinition:
    connector = CONNECTORS.get(connector_id)  # type: ignore[arg-type]
    if not connector:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown connector")
    return connector


def _prune_pending_states(now: Optional[datetime] = None) -> None:
    current_time = now or _now()
    expired = [
        key
        for key, pending in _pending_states.items()
        if pending.expires_at <= current_time
    ]
    for key in expired:
        _pending_states.pop(key, None)

    overflow = len(_pending_states) - MAX_PENDING_STATES
    if overflow > 0:
        oldest = sorted(_pending_states.items(), key=lambda item: item[1].expires_at)
        for key, _pending in oldest[:overflow]:
            _pending_states.pop(key, None)


def _status_for(user_id: str, connector: ConnectorDefinition) -> ConnectorStatusResponse:
    return ConnectorStatusResponse(
        id=connector.id,
        name=connector.name,
        state="not_connected",
        scopes=connector.scopes,
        detail="OAuth start is available; token exchange and sync storage are not connected yet.",
    )


def _build_authorization_url(connector: ConnectorDefinition, state: str) -> str:
    client_id = _client_id(connector.id)
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{connector.name} OAuth client ID is not configured",
        )

    params = {
        "client_id": client_id,
        "redirect_uri": _redirect_uri(connector.id),
        "response_type": "code",
        "state": state,
    }
    if connector.id == "google-drive":
        params.update(
            {
                "access_type": "offline",
                "include_granted_scopes": "true",
                "prompt": "consent",
                "scope": " ".join(connector.scopes),
            }
        )
    if connector.id == "notion":
        params["owner"] = "user"

    return f"{connector.auth_url}?{urlencode(params)}"


@router.get("", response_model=ConnectorListResponse)
async def list_connectors(current_user: dict = Depends(require_user)) -> ConnectorListResponse:
    user_id = str(current_user.get("id"))
    return ConnectorListResponse(
        connectors=[_status_for(user_id, connector) for connector in CONNECTORS.values()]
    )


@router.get("/{connector_id}/status", response_model=ConnectorStatusResponse)
async def connector_status(
    connector_id: str,
    current_user: dict = Depends(require_user),
) -> ConnectorStatusResponse:
    connector = _get_connector(connector_id)
    return _status_for(str(current_user.get("id")), connector)


@router.post("/{connector_id}/oauth/start", response_model=ConnectorStartResponse)
async def start_connector_oauth(
    connector_id: str,
    current_user: dict = Depends(require_user),
) -> ConnectorStartResponse:
    connector = _get_connector(connector_id)
    _prune_pending_states()
    state = secrets.token_urlsafe(32)
    expires_at = _now() + timedelta(minutes=STATE_TTL_MINUTES)
    authorization_url = _build_authorization_url(connector, state)
    _pending_states[state] = PendingOAuthState(
        connector_id=connector.id,
        user_id=str(current_user.get("id")),
        expires_at=expires_at,
    )

    return ConnectorStartResponse(
        connector_id=connector.id,
        authorization_url=authorization_url,
        state=state,
        expires_at=expires_at,
    )


@router.get("/{connector_id}/oauth/callback")
async def connector_oauth_callback(
    connector_id: str,
    state: str = Query(..., min_length=1),
    code: Optional[str] = Query(None, min_length=1),
    error: Optional[str] = Query(None, min_length=1),
) -> dict:
    connector = _get_connector(connector_id)
    now = _now()
    _prune_pending_states(now)
    pending = _pending_states.pop(state, None)
    if not pending or pending.connector_id != connector.id or pending.expires_at <= now:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired connector authorization state",
        )
    if error or not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Authorization denied: {error or 'no authorization code received'}",
        )

    # Token exchange, encrypted credential storage, and source ingestion are intentionally
    # separate follow-up steps. Do not mark the connector as connected until those exist.
    return {
        "status": "pending",
        "connector_id": connector.id,
        "detail": f"{connector.name} authorization received; token exchange is not enabled yet.",
    }


@router.post("/{connector_id}/disconnect", response_model=ConnectorDisconnectResponse)
async def disconnect_connector(
    connector_id: str,
    current_user: dict = Depends(require_user),
) -> ConnectorDisconnectResponse:
    connector = _get_connector(connector_id)
    disconnected = False
    return ConnectorDisconnectResponse(connector_id=connector.id, disconnected=disconnected)
