"""Authentication routes for Google OAuth and JWT management."""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from jose import JWTError, jwt
from pydantic import BaseModel

from src.api.dependencies import get_current_user
from src.config import settings
from src.database.user_store import UserStore

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Initialize user store
user_store = UserStore()


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
            settings.google_client_id
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
