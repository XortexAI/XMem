"""Pydantic models for database entities."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class User(BaseModel):
    """User model for MongoDB users collection."""

    id: Optional[str] = Field(None, alias="_id")
    email: str = Field(..., description="User's email address")
    name: str = Field(..., description="User's display name")
    google_id: str = Field(..., description="Google OAuth ID")
    picture: Optional[str] = Field(None, description="Google profile picture URL")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        json_encoders = {ObjectId: str}


class APIKey(BaseModel):
    """API Key model for MongoDB api_keys collection."""

    id: Optional[str] = Field(None, alias="_id")
    user_id: str = Field(..., description="ID of the user who owns this key")
    key_hash: str = Field(..., description="SHA-256 hash of the API key")
    key_prefix: str = Field(..., description="First 8 characters of the key for display")
    name: str = Field(default="Default", description="User-defined name for this key")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = Field(None, description="Last time the key was used")
    is_active: bool = Field(default=True, description="Whether the key is active")

    class Config:
        populate_by_name = True
        json_encoders = {ObjectId: str}
