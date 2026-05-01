"""Authentication routes for Google OAuth and JWT management."""

import secrets
import string
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from jose import JWTError, jwt
from pydantic import BaseModel

from src.api.dependencies import get_current_user, require_api_key, require_user
from src.config import settings
from src.database.user_store import UserStore
from src.database.api_key_store import APIKeyStore

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Initialize stores
user_store = UserStore()
api_key_store = APIKeyStore()

# ═══════════════════════════════════════════════════════════════════════════
# MCP OAuth Temp Token Store (in-memory with TTL)
# ═══════════════════════════════════════════════════════════════════════════
_mcp_temp_tokens: Dict[str, Dict[str, Any]] = {}
TEMP_TOKEN_PREFIX = "xm-temp-"
TEMP_TOKEN_TTL_MINUTES = 10
TEMP_TOKEN_LENGTH = 32


def _generate_mcp_temp_token() -> str:
    """Generate a temporary token for MCP OAuth flow."""
    alphabet = string.ascii_letters + string.digits
    random_part = "".join(secrets.choice(alphabet) for _ in range(TEMP_TOKEN_LENGTH))
    return f"{TEMP_TOKEN_PREFIX}{random_part}"


def _create_mcp_temp_token(user_id: str) -> str:
    """Create and store a temporary token for the user."""
    token = _generate_mcp_temp_token()
    expires_at = datetime.utcnow() + timedelta(minutes=TEMP_TOKEN_TTL_MINUTES)

    _mcp_temp_tokens[token] = {
        "user_id": user_id,
        "created_at": datetime.utcnow(),
        "expires_at": expires_at,
        "exchanged": False,
    }

    return token


def _get_and_invalidate_mcp_token(token: str) -> Optional[str]:
    """Validate temp token and return user_id if valid, None otherwise."""
    if token not in _mcp_temp_tokens:
        return None

    token_data = _mcp_temp_tokens[token]

    # Check expiry
    if datetime.utcnow() > token_data["expires_at"]:
        del _mcp_temp_tokens[token]
        return None

    # Check if already exchanged
    if token_data["exchanged"]:
        return None

    # Mark as exchanged and return user_id
    user_id = token_data["user_id"]
    del _mcp_temp_tokens[token]  # Single-use token
    return user_id


# ═══════════════════════════════════════════════════════════════════════════
# Standard OAuth 2.0 Store (for ChatGPT UI)
# ═══════════════════════════════════════════════════════════════════════════
_oauth_auth_codes: Dict[str, Dict[str, Any]] = {}

def _generate_auth_code(user_id: str) -> str:
    """Generate a standard OAuth 2.0 authorization code."""
    alphabet = string.ascii_letters + string.digits
    code = "".join(secrets.choice(alphabet) for _ in range(32))
    
    _oauth_auth_codes[code] = {
        "user_id": user_id,
        "expires_at": datetime.utcnow() + timedelta(minutes=10)
    }
    return code

def _get_and_invalidate_auth_code(code: str) -> Optional[str]:
    """Validate auth code and return user_id if valid."""
    if code not in _oauth_auth_codes:
        return None
        
    data = _oauth_auth_codes[code]
    del _oauth_auth_codes[code] # Single-use
    
    if datetime.utcnow() > data["expires_at"]:
        return None
        
    return data["user_id"]


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════════════════════════

class GoogleTokenRequest(BaseModel):
    """Request model for Google OAuth token exchange."""
    credential: str
    client_id: Optional[str] = None


class TokenResponse(BaseModel):
    """Response model for successful authentication."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


class UserResponse(BaseModel):
    """Response model for current user."""
    id: str
    email: str
    name: str
    username: Optional[str] = None
    picture: Optional[str] = None
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

class SetUsernameRequest(BaseModel):
    """Request model for setting a username."""
    username: str

class UsernameCheckResponse(BaseModel):
    """Response model for username availability check."""
    available: bool


class RefreshRequest(BaseModel):
    """Request model for token refresh."""
    refresh_token: str


# ═══════════════════════════════════════════════════════════════════════════
# MCP OAuth Models
# ═══════════════════════════════════════════════════════════════════════════

class MCPTempTokenResponse(BaseModel):
    """Response model for MCP temp token generation."""
    temp_token: str
    expires_in: int  # seconds
    expires_at: datetime


class MCPExchangeRequest(BaseModel):
    """Request model for exchanging temp token for API key."""
    temp_token: str
    client_type: str = "mcp"  # For future extensibility


class MCPExchangeResponse(BaseModel):
    """Response model for successful MCP authentication."""
    status: str = "success"
    api_key: str
    user: dict


class OAuthApproveRequest(BaseModel):
    """Request from frontend to approve OAuth and get a code."""
    client_id: str
    redirect_uri: str

class OAuthApproveResponse(BaseModel):
    """Response with the authorization code."""
    code: str


# ═══════════════════════════════════════════════════════════════════════════
# JWT Utilities
# ═══════════════════════════════════════════════════════════════════════════

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token.

    Args:
        data: Data to encode in the token
        expires_delta: Optional custom expiration time

    Returns:
        JWT token string
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=settings.jwt_expiration_days)

    to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "access"})

    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )

    return encoded_jwt




def verify_google_token(credential: str) -> dict:
    """Verify a Google ID token and return user info.

    Args:
        credential: Google ID token

    Returns:
        Dictionary with user info from Google

    Raises:
        HTTPException: If token is invalid
    """
    try:
        # Verify the token with Google
        idinfo = id_token.verify_oauth2_token(
            credential,
            google_requests.Request(),
            settings.google_client_id,
            clock_skew_in_seconds=60
        )

        # Check issuer
        if idinfo["iss"] not in ["accounts.google.com", "https://accounts.google.com"]:
            raise ValueError("Invalid issuer")

        return {
            "google_id": idinfo["sub"],
            "email": idinfo.get("email"),
            "name": idinfo.get("name"),
            "picture": idinfo.get("picture"),
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid Google token: {str(e)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/google", response_model=TokenResponse)
async def auth_google(request: GoogleTokenRequest):
    """Authenticate with Google OAuth credential.

    This endpoint receives the Google ID token from the frontend,
    verifies it, creates or updates the user, and returns a JWT.
    """
    # Verify the Google token
    google_user = verify_google_token(request.credential)

    # Get or create user in database
    user = user_store.get_or_create_user(
        google_id=google_user["google_id"],
        email=google_user["email"],
        name=google_user["name"],
        picture=google_user.get("picture"),
    )

    # Create JWT token
    access_token = create_access_token(
        data={"sub": str(user["_id"])}
    )

    # Convert user document for response
    user_response = {
        "id": str(user["_id"]),
        "email": user["email"],
        "name": user["name"],
        "username": user.get("username"),
        "picture": user.get("picture"),
        "created_at": user.get("created_at"),
        "last_login": user.get("last_login"),
    }

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.jwt_expiration_days * 24 * 60 * 60,  # Convert to seconds
        user=user_response,
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current authenticated user information."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return UserResponse(**current_user)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshRequest):
    """Refresh an access token.

    Note: This is a simplified implementation. In production, you might
    want to implement refresh tokens separately from access tokens.
    """
    try:
        payload = jwt.decode(
            request.refresh_token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )

        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

        # Get user from database
        user = user_store.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )

        # Create new access token
        new_token = create_access_token(
            data={"sub": str(user["_id"])}
        )

        # Convert user document for response
        user_response = {
            "id": str(user["_id"]),
            "email": user["email"],
            "name": user["name"],
            "username": user.get("username"),
            "picture": user.get("picture"),
            "created_at": user.get("created_at"),
            "last_login": user.get("last_login"),
        }

        return TokenResponse(
            access_token=new_token,
            token_type="bearer",
            expires_in=settings.jwt_expiration_days * 24 * 60 * 60,
            user=user_response,
        )

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

@router.get("/check-username/{username}", response_model=UsernameCheckResponse)
async def check_username(username: str):
    """Check if a username is available."""
    # Basic validation
    if len(username) < 3 or len(username) > 30:
        return UsernameCheckResponse(available=False)
    
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        return UsernameCheckResponse(available=False)
        
    is_available = user_store.is_username_available(username)
    return UsernameCheckResponse(available=is_available)

@router.post("/set-username", response_model=UserResponse)
async def set_username(req: SetUsernameRequest, current_user: dict = Depends(get_current_user)):
    """Set username for the currently authenticated user."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
        
    if current_user.get("username"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already set"
        )
        
    username = req.username
    if len(username) < 3 or len(username) > 30:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username must be between 3 and 30 characters"
        )
        
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username can only contain letters, numbers, underscores, and hyphens"
        )
        
    success = user_store.set_username(current_user["id"], username)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username is not available or could not be set"
        )
        
    # Get updated user
    user = user_store.get_user_by_id(current_user["id"])
    updated_user = dict(user) if user else current_user
    if "_id" in updated_user:
        updated_user["id"] = str(updated_user.pop("_id"))
    elif "id" not in updated_user:
        updated_user["id"] = current_user["id"]
    
    return UserResponse(**updated_user)

@router.get("/verify-key", response_model=UserResponse)
async def verify_key(user: dict = Depends(require_api_key)):
    """Verify an API key and return the associated user information."""
    return UserResponse(**user)


# ═══════════════════════════════════════════════════════════════════════════
# MCP OAuth Routes
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/mcp-token", response_model=MCPTempTokenResponse)
async def generate_mcp_temp_token(current_user: dict = Depends(require_user)):
    """
    Generate a temporary token for MCP OAuth authentication.

    This endpoint is called from the XMem web UI when a user wants to
    connect their account to an MCP client (Claude Desktop, ChatGPT, etc.)
    that doesn't support environment variable configuration.

    The temp token is valid for 10 minutes and can only be exchanged once.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    user_id = str(current_user.get("id"))
    temp_token = _create_mcp_temp_token(user_id)

    return MCPTempTokenResponse(
        temp_token=temp_token,
        expires_in=TEMP_TOKEN_TTL_MINUTES * 60,
        expires_at=_mcp_temp_tokens[temp_token]["expires_at"]
    )


@router.post("/mcp-exchange", response_model=MCPExchangeResponse)
async def exchange_mcp_token(request: MCPExchangeRequest):
    """
    Exchange a temporary MCP token for a permanent API key.

    This endpoint is called by the MCP server to exchange the temporary
    token (provided by the user) for a long-lived API key.

    The temp token is single-use and invalidated after exchange.
    """
    # Validate and consume the temp token
    user_id = _get_and_invalidate_mcp_token(request.temp_token)

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )

    # Get user details
    user = user_store.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Create a new API key for this user
    key_result = api_key_store.create_api_key(
        user_id=user_id,
        name=f"MCP Client - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
    )

    # Prepare user response
    user_response = {
        "id": str(user["_id"]),
        "email": user.get("email"),
        "name": user.get("name"),
        "username": user.get("username"),
    }

    return MCPExchangeResponse(
        status="success",
        api_key=key_result["key"],
        user=user_response
    )


# ═══════════════════════════════════════════════════════════════════════════
# Standard OAuth 2.0 Routes (For ChatGPT UI)
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/oauth/approve", response_model=OAuthApproveResponse)
async def oauth_approve(request: OAuthApproveRequest, current_user: dict = Depends(require_user)):
    """
    Called by the Next.js frontend when the user clicks 'Approve' on the consent screen.
    Generates an authorization code for standard OAuth 2.0 flow.
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    user_id = str(current_user.get("id"))
    code = _generate_auth_code(user_id)
    return OAuthApproveResponse(code=code)


from fastapi import Form
from fastapi.responses import JSONResponse

@router.post("/oauth/token")
async def oauth_token(
    grant_type: str = Form(...),
    code: str = Form(None),
    redirect_uri: str = Form(None),
    client_id: str = Form(None)
):
    """
    Standard OAuth 2.0 token endpoint.
    ChatGPT calls this directly to exchange the authorization code for an access token.
    """
    if grant_type != "authorization_code":
        return JSONResponse(status_code=400, content={"error": "unsupported_grant_type"})
        
    if not code:
        return JSONResponse(status_code=400, content={"error": "invalid_request", "error_description": "code is required"})
        
    user_id = _get_and_invalidate_auth_code(code)
    if not user_id:
        return JSONResponse(status_code=400, content={"error": "invalid_grant", "error_description": "Invalid or expired authorization code"})
        
    # Generate a permanent API key acting as the access token
    key_result = api_key_store.create_api_key(
        user_id=user_id,
        name=f"OAuth Client ({client_id or 'Unknown'}) - {datetime.utcnow().strftime('%Y-%m-%d')}"
    )
    
    return {
        "access_token": key_result["key"],
        "token_type": "Bearer",
        "expires_in": 31536000, # 1 year
        "scope": "all"
    }
