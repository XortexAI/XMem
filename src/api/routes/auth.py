"""Authentication routes for Google OAuth and JWT management."""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from jose import JWTError, jwt
from pydantic import BaseModel

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
    picture: Optional[str] = None
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None


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


async def get_current_user(token: str = Depends(lambda: None)) -> Optional[dict]:
    """Validate JWT token and return current user.

    This is used as a FastAPI dependency.
    """
    if not token:
        return None

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )

        user_id: str = payload.get("sub")
        if user_id is None:
            return None

        # Get fresh user data from database
        user = user_store.get_user_by_id(user_id)
        if not user:
            return None

        # Convert ObjectId to string for JSON serialization
        user["id"] = str(user.pop("_id"))

        return user

    except JWTError:
        return None


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
