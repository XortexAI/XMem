"""API Key management routes."""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.dependencies import get_current_user
from src.database.api_key_store import APIKeyStore

router = APIRouter(prefix="/api/keys", tags=["API Keys"])

# Initialize API key store
api_key_store = APIKeyStore()


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════════════════════════

class APIKeyCreateRequest(BaseModel):
    """Request model for creating a new API key."""
    name: str = Field(default="Default", description="Name for this API key")


class APIKeyCreateResponse(BaseModel):
    """Response model for newly created API key.

    NOTE: The full API key is only returned once during creation.
    """
    key: str = Field(..., description="The full API key (only shown once)")
    key_id: str = Field(..., description="ID of the API key for reference")
    name: str = Field(..., description="Name of the API key")
    created_at: datetime = Field(..., description="Creation timestamp")


class APIKeyResponse(BaseModel):
    """Response model for API key (without the actual key)."""
    id: str = Field(..., description="API key ID")
    key_prefix: str = Field(..., description="First 8 characters of the key")
    name: str = Field(..., description="Name of the API key")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_used: Optional[datetime] = Field(None, description="Last usage timestamp")
    is_active: bool = Field(..., description="Whether the key is active")


class APIKeyUpdateRequest(BaseModel):
    """Request model for updating an API key."""
    name: str = Field(..., description="New name for the API key")


class APIKeyListResponse(BaseModel):
    """Response model for listing API keys."""
    keys: List[APIKeyResponse]
    total: int


# ═══════════════════════════════════════════════════════════════════════════
# Dependencies
# ═══════════════════════════════════════════════════════════════════════════

async def require_auth(current_user: dict = Depends(get_current_user)) -> dict:
    """Dependency to require authentication.

    Raises HTTPException if user is not authenticated.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


# ═══════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════

@router.get("", response_model=APIKeyListResponse)
async def list_api_keys(
    current_user: dict = Depends(require_auth),
    include_inactive: bool = False,
):
    """List all API keys for the current user.

    Returns metadata about each key but NOT the actual key values.
    """
    keys = api_key_store.get_user_api_keys(
        user_id=current_user["id"],
        include_inactive=include_inactive
    )

    # Convert to response model
    key_responses = [
        APIKeyResponse(
            id=key["id"],
            key_prefix=key.get("key_prefix", "xxxx-xxxx"),
            name=key["name"],
            created_at=key["created_at"],
            last_used=key.get("last_used"),
            is_active=key.get("is_active", True),
        )
        for key in keys
    ]

    return APIKeyListResponse(keys=key_responses, total=len(key_responses))


@router.post("", response_model=APIKeyCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    request: APIKeyCreateRequest,
    current_user: dict = Depends(require_auth),
):
    """Create a new API key.

    WARNING: The full API key is only returned once in this response.
    Make sure to save it securely - it cannot be retrieved again.
    """
    result = api_key_store.create_api_key(
        user_id=current_user["id"],
        name=request.name,
    )

    return APIKeyCreateResponse(
        key=result["key"],
        key_id=result["key_id"],
        name=result["name"],
        created_at=result["created_at"],
    )


@router.patch("/{key_id}", response_model=APIKeyResponse)
async def update_api_key(
    key_id: str,
    request: APIKeyUpdateRequest,
    current_user: dict = Depends(require_auth),
):
    """Update an API key's name."""
    success = api_key_store.update_api_key_name(
        user_id=current_user["id"],
        key_id=key_id,
        new_name=request.name,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found or not owned by you"
        )

    # Get updated key info
    keys = api_key_store.get_user_api_keys(current_user["id"], include_inactive=True)
    updated_key = next((k for k in keys if k["id"] == key_id), None)

    if not updated_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )

    return APIKeyResponse(
        id=updated_key["id"],
        key_prefix=updated_key.get("key_prefix", "xxxx-xxxx"),
        name=updated_key["name"],
        created_at=updated_key["created_at"],
        last_used=updated_key.get("last_used"),
        is_active=updated_key.get("is_active", True),
    )


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: str,
    current_user: dict = Depends(require_auth),
):
    """Revoke (deactivate) an API key.

    Once revoked, the key cannot be used for authentication.
    This action is reversible - you can reactivate a key if needed (not implemented yet).
    """
    success = api_key_store.revoke_api_key(
        user_id=current_user["id"],
        key_id=key_id,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found or not owned by you"
        )

    return None
