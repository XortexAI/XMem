"""Pydantic models for database entities."""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from bson import ObjectId


class TeamRole(str, Enum):
    """Team member roles with hierarchical permissions."""
    MANAGER = "manager"           # Full access, team management
    STAFF_ENGINEER = "staff_engineer"  # Can annotate, view all, mentor
    SDE2 = "sde2"                 # Can annotate, view team annotations
    INTERN = "intern"             # Can annotate, view assigned scope


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
    username: Optional[str] = Field(None, description="Unique username for XMem services")
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


class Project(BaseModel):
    """Enterprise project model for team-based code annotations."""

    id: Optional[str] = Field(None, alias="_id")
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    org_id: str = Field(..., description="GitHub organization ID")
    repo: str = Field(..., description="Repository name")
    created_by: str = Field(..., description="User ID who created the project")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True, description="Whether the project is active")
    annotation_count: int = Field(default=0, description="Number of annotations in project")

    class Config:
        populate_by_name = True
        json_encoders = {ObjectId: str}


class TeamMember(BaseModel):
    """Team member association for enterprise projects."""

    id: Optional[str] = Field(None, alias="_id")
    project_id: str = Field(..., description="Project ID")
    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username for display")
    email: Optional[str] = Field(None, description="User email")
    role: TeamRole = Field(default=TeamRole.INTERN, description="Team role")
    added_by: str = Field(..., description="User ID who added this member")
    added_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True, description="Whether the member is active")

    class Config:
        populate_by_name = True
        json_encoders = {ObjectId: str}
