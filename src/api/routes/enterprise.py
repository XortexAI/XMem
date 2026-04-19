"""
/v1/enterprise/* routes — Enterprise team annotations API.

Provides endpoints for:
  - Project management (create, list, update, delete)
  - Team member management (add, remove, update roles)
  - Annotation CRUD (create, search, update, delete)
  - Team chat with annotation-aware retrieval

All routes require authentication via Bearer token.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.api.dependencies import require_api_key
from src.config import settings
from src.database.models import TeamRole
from src.database.project_store import ProjectStore
from src.database.user_store import UserStore
from src.storage.team_annotation_store import TeamAnnotationStore

logger = logging.getLogger("xmem.api.routes.enterprise")

router = APIRouter(prefix="/v1/enterprise", tags=["enterprise"])

# ---------------------------------------------------------------------------
# Singleton stores
# ---------------------------------------------------------------------------

_project_store: Optional[ProjectStore] = None
_annotation_store: Optional[TeamAnnotationStore] = None
_user_store: Optional[UserStore] = None


def _get_project_store() -> ProjectStore:
    global _project_store
    if _project_store is None:
        _project_store = ProjectStore(
            uri=settings.mongodb_uri,
            database=settings.mongodb_database,
        )
    return _project_store


def _get_annotation_store() -> TeamAnnotationStore:
    global _annotation_store
    if _annotation_store is None:
        _annotation_store = TeamAnnotationStore(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            dimension=settings.pinecone_dimension,
        )
    return _annotation_store


def _get_user_store() -> UserStore:
    global _user_store
    if _user_store is None:
        _user_store = UserStore(
            uri=settings.mongodb_uri,
            database=settings.mongodb_database,
        )
    return _user_store


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class CreateProjectRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    org_id: str = Field(..., min_length=1)
    repo: str = Field(..., min_length=1)


class UpdateProjectRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    is_active: Optional[bool] = None


class AddTeamMemberRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    username: str = Field(..., min_length=1)
    email: Optional[str] = None
    role: str = Field(..., pattern="^(manager|staff_engineer|sde2|intern)$")


class UpdateTeamMemberRoleRequest(BaseModel):
    role: str = Field(..., pattern="^(manager|staff_engineer|sde2|intern)$")


class CreateAnnotationRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)
    annotation_type: str = Field(
        default="explanation",
        pattern="^(bug_report|fix|explanation|warning|feature_idea)$"
    )
    file_path: Optional[str] = None
    symbol_name: Optional[str] = None
    severity: Optional[str] = Field(None, pattern="^(low|medium|high|critical)$")


class UpdateAnnotationRequest(BaseModel):
    content: Optional[str] = Field(None, min_length=1, max_length=10000)
    status: Optional[str] = Field(None, pattern="^(active|resolved|outdated)$")
    severity: Optional[str] = Field(None, pattern="^(low|medium|high|critical)$")


class SearchAnnotationsRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    file_path: Optional[str] = None
    symbol_name: Optional[str] = None


class TeamChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    file_path: Optional[str] = None
    symbol_name: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=50)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize a MongoDB document for JSON response."""
    if doc is None:
        return None
    result = dict(doc)
    if "_id" in result:
        result["id"] = str(result.pop("_id"))
    # Convert datetime to ISO string
    for key in ["created_at", "updated_at", "added_at", "last_login"]:
        if key in result and hasattr(result[key], "isoformat"):
            result[key] = result[key].isoformat()
    return result


# ---------------------------------------------------------------------------
# User lookup routes
# ---------------------------------------------------------------------------

@router.get("/users/lookup", summary="Lookup user by username")
async def lookup_user(
    username: str,
    user: dict = Depends(require_api_key),
) -> JSONResponse:
    """Lookup a user by username to validate before adding to team."""
    store = _get_user_store()

    # Search for user by username
    found_user = store.get_user_by_username(username)

    if not found_user:
        # Also try searching by email or partial match
        # For now, return not found
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")

    return JSONResponse(
        content={
            "status": "ok",
            "user": {
                "id": str(found_user.get("_id")),
                "username": found_user.get("username"),
                "email": found_user.get("email"),
                "name": found_user.get("name"),
                "picture": found_user.get("picture"),
            },
        }
    )


@router.get("/users/search", summary="Search users by query")
async def search_users(
    q: str,
    limit: int = 10,
    user: dict = Depends(require_api_key),
) -> JSONResponse:
    """Search users by username or email prefix."""
    store = _get_user_store()

    # Get all users and filter (in production, you'd want a proper search index)
    # This is a simple implementation for now
    users_collection = store.users
    if users_collection is None:
        # In-memory fallback
        return JSONResponse(content={"status": "ok", "users": []})

    try:
        # Search by username or email (case-insensitive)
        query_filter = {
            "$or": [
                {"username": {"$regex": q, "$options": "i"}},
                {"email": {"$regex": q, "$options": "i"}},
                {"name": {"$regex": q, "$options": "i"}},
            ]
        }
        cursor = users_collection.find(query_filter).limit(limit)
        users = []
        for doc in cursor:
            users.append({
                "id": str(doc.get("_id")),
                "username": doc.get("username"),
                "email": doc.get("email"),
                "name": doc.get("name"),
                "picture": doc.get("picture"),
            })

        return JSONResponse(content={"status": "ok", "users": users})
    except Exception as e:
        logger.error(f"Error searching users: {e}")
        return JSONResponse(content={"status": "ok", "users": []})


# ---------------------------------------------------------------------------
# Project routes
# ---------------------------------------------------------------------------

@router.post("/projects", summary="Create a new enterprise project")
async def create_project(
    req: CreateProjectRequest,
    user: dict = Depends(require_api_key),
) -> JSONResponse:
    """Create a new enterprise project."""
    store = _get_project_store()
    user_id = user.get("id") or user.get("google_id")

    project = store.create_project(
        name=req.name,
        org_id=req.org_id,
        repo=req.repo,
        created_by=user_id,
        description=req.description,
    )

    if project is None:
        raise HTTPException(status_code=500, detail="Failed to create project")

    # Add creator as manager
    store.add_team_member(
        project_id=str(project["_id"]),
        user_id=user_id,
        username=user.get("username") or user.get("name") or user_id,
        email=user.get("email"),
        role=TeamRole.MANAGER,
        added_by=user_id,
    )

    return JSONResponse(
        status_code=201,
        content={
            "status": "ok",
            "project": _serialize_doc(project),
        }
    )


@router.get("/projects", summary="List all projects for the current user")
async def list_projects(
    user: dict = Depends(require_api_key),
) -> JSONResponse:
    """List all projects where the user is a member or creator."""
    store = _get_project_store()
    user_id = user.get("id") or user.get("google_id")

    projects = store.list_projects_for_user(user_id)

    return JSONResponse(
        content={
            "status": "ok",
            "projects": [_serialize_doc(p) for p in projects],
        }
    )


@router.get("/projects/{project_id}", summary="Get project details")
async def get_project(
    project_id: str,
    user: dict = Depends(require_api_key),
) -> JSONResponse:
    """Get project details."""
    store = _get_project_store()
    user_id = user.get("id") or user.get("google_id")

    # Check access
    if not store.check_user_has_access(project_id, user_id):
        raise HTTPException(status_code=403, detail="Access denied")

    project = store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get team members
    members = store.list_team_members(project_id)

    return JSONResponse(
        content={
            "status": "ok",
            "project": _serialize_doc(project),
            "team_members": [_serialize_doc(m) for m in members],
        }
    )


@router.patch("/projects/{project_id}", summary="Update project")
async def update_project(
    project_id: str,
    req: UpdateProjectRequest,
    user: dict = Depends(require_api_key),
) -> JSONResponse:
    """Update project details."""
    store = _get_project_store()
    user_id = user.get("id") or user.get("google_id")

    # Check permissions (only manager or creator)
    if not store.check_user_can_manage_team(project_id, user_id):
        if not store.check_user_is_project_creator(project_id, user_id):
            raise HTTPException(status_code=403, detail="Only managers can update projects")

    updates = {}
    if req.name is not None:
        updates["name"] = req.name
    if req.description is not None:
        updates["description"] = req.description
    if req.is_active is not None:
        updates["is_active"] = req.is_active

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    success = store.update_project(project_id, updates)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update project")

    return JSONResponse(
        content={
            "status": "ok",
            "message": "Project updated",
        }
    )


@router.delete("/projects/{project_id}", summary="Delete project")
async def delete_project(
    project_id: str,
    user: dict = Depends(require_api_key),
) -> JSONResponse:
    """Delete a project and all its data."""
    store = _get_project_store()
    user_id = user.get("id") or user.get("google_id")

    # Only creator can delete
    if not store.check_user_is_project_creator(project_id, user_id):
        raise HTTPException(status_code=403, detail="Only the project creator can delete")

    # Clear annotations first
    ann_store = _get_annotation_store()
    try:
        ann_store.clear_project_annotations(project_id)
    except Exception as e:
        logger.warning(f"Failed to clear annotations for project {project_id}: {e}")

    # Delete project
    success = store.delete_project(project_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete project")

    return JSONResponse(
        content={
            "status": "ok",
            "message": "Project deleted",
        }
    )


# ---------------------------------------------------------------------------
# Team member routes
# ---------------------------------------------------------------------------

@router.post("/projects/{project_id}/team", summary="Add team member")
async def add_team_member(
    project_id: str,
    req: AddTeamMemberRequest,
    user: dict = Depends(require_api_key),
) -> JSONResponse:
    """Add a team member to a project."""
    store = _get_project_store()
    user_id = user.get("id") or user.get("google_id")

    # Check permissions
    if not store.check_user_can_manage_team(project_id, user_id):
        raise HTTPException(status_code=403, detail="Only managers can add team members")

    # Validate role
    try:
        role = TeamRole(req.role)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role: {req.role}")

    member = store.add_team_member(
        project_id=project_id,
        user_id=req.user_id,
        username=req.username,
        email=req.email,
        role=role,
        added_by=user_id,
    )

    if member is None:
        raise HTTPException(status_code=500, detail="Failed to add team member")

    return JSONResponse(
        status_code=201,
        content={
            "status": "ok",
            "member": _serialize_doc(member),
        }
    )


@router.get("/projects/{project_id}/team", summary="List team members")
async def list_team_members(
    project_id: str,
    user: dict = Depends(require_api_key),
) -> JSONResponse:
    """List all team members for a project."""
    store = _get_project_store()
    user_id = user.get("id") or user.get("google_id")

    # Check access
    if not store.check_user_has_access(project_id, user_id):
        raise HTTPException(status_code=403, detail="Access denied")

    members = store.list_team_members(project_id)

    return JSONResponse(
        content={
            "status": "ok",
            "members": [_serialize_doc(m) for m in members],
        }
    )


@router.patch("/projects/{project_id}/team/{member_user_id}", summary="Update team member role")
async def update_team_member_role(
    project_id: str,
    member_user_id: str,
    req: UpdateTeamMemberRoleRequest,
    user: dict = Depends(require_api_key),
) -> JSONResponse:
    """Update a team member's role."""
    store = _get_project_store()
    user_id = user.get("id") or user.get("google_id")

    # Check permissions - managers and staff engineers can edit roles
    if not store.check_user_can_edit_team_member(project_id, user_id, member_user_id):
        raise HTTPException(
            status_code=403,
            detail="Only managers and staff engineers can update roles, and only of members below their level"
        )

    # Can't change own role
    if member_user_id == user_id:
        raise HTTPException(status_code=400, detail="Cannot change your own role")

    # Validate the new role is valid
    try:
        role = TeamRole(req.role)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role: {req.role}")

    # Check that the user isn't trying to assign a role higher than their own
    user_role = store.get_user_role_in_project(project_id, user_id)
    if user_role is None:
        raise HTTPException(status_code=403, detail="Access denied")

    role_hierarchy = {
        TeamRole.MANAGER: 4,
        TeamRole.STAFF_ENGINEER: 3,
        TeamRole.SDE2: 2,
        TeamRole.INTERN: 1,
    }
    user_level = role_hierarchy.get(user_role, 0)
    requested_level = role_hierarchy.get(role, 0)

    if requested_level >= user_level:
        raise HTTPException(
            status_code=403,
            detail=f"Cannot assign role equal to or higher than your own"
        )

    member = store.update_team_member_role(project_id, member_user_id, role)
    if member is None:
        raise HTTPException(status_code=404, detail="Team member not found")

    return JSONResponse(
        content={
            "status": "ok",
            "member": _serialize_doc(member),
        }
    )


@router.delete("/projects/{project_id}/team/{member_user_id}", summary="Remove team member")
async def remove_team_member(
    project_id: str,
    member_user_id: str,
    user: dict = Depends(require_api_key),
) -> JSONResponse:
    """Remove a team member from a project."""
    store = _get_project_store()
    user_id = user.get("id") or user.get("google_id")

    # Check permissions
    if not store.check_user_can_manage_team(project_id, user_id):
        raise HTTPException(status_code=403, detail="Only managers can remove team members")

    # Can't remove self
    if member_user_id == user_id:
        raise HTTPException(status_code=400, detail="Cannot remove yourself. Delete project instead.")

    success = store.remove_team_member(project_id, member_user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Team member not found")

    return JSONResponse(
        content={
            "status": "ok",
            "message": "Team member removed",
        }
    )


# ---------------------------------------------------------------------------
# Annotation routes
# ---------------------------------------------------------------------------

@router.post("/projects/{project_id}/annotations", summary="Create annotation")
async def create_annotation(
    project_id: str,
    req: CreateAnnotationRequest,
    user: dict = Depends(require_api_key),
) -> JSONResponse:
    """Create a new team annotation."""
    project_store = _get_project_store()
    ann_store = _get_annotation_store()

    user_id = user.get("id") or user.get("google_id")

    # Check permissions (all team members can annotate)
    if not project_store.check_user_can_annotate(project_id, user_id):
        raise HTTPException(status_code=403, detail="Access denied")

    # Get project for org_id/repo
    project = project_store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get user's role
    role = project_store.get_user_role_in_project(project_id, user_id)
    role_str = role.value if role else "intern"

    annotation_id = ann_store.create_annotation(
        project_id=project_id,
        content=req.content,
        author_id=user_id,
        author_name=user.get("username") or user.get("name") or user_id,
        author_role=role_str,
        org_id=project["org_id"],
        repo=project["repo"],
        annotation_type=req.annotation_type,
        file_path=req.file_path,
        symbol_name=req.symbol_name,
        severity=req.severity,
    )

    # Increment project annotation count
    project_store.increment_annotation_count(project_id)

    return JSONResponse(
        status_code=201,
        content={
            "status": "ok",
            "annotation_id": annotation_id,
        }
    )


@router.post("/projects/{project_id}/annotations/search", summary="Search annotations")
async def search_annotations(
    project_id: str,
    req: SearchAnnotationsRequest,
    user: dict = Depends(require_api_key),
) -> JSONResponse:
    """Search annotations by semantic similarity."""
    project_store = _get_project_store()
    ann_store = _get_annotation_store()

    user_id = user.get("id") or user.get("google_id")

    # Check access
    if not project_store.check_user_has_access(project_id, user_id):
        raise HTTPException(status_code=403, detail="Access denied")

    # Build filters
    filters = {}
    if req.file_path:
        filters["file_path"] = req.file_path
    if req.symbol_name:
        filters["symbol_name"] = req.symbol_name

    results = await ann_store.search_annotations(
        project_id=project_id,
        query=req.query,
        top_k=req.top_k,
        filters=filters if filters else None,
    )

    return JSONResponse(
        content={
            "status": "ok",
            "annotations": [
                {
                    "id": r.id,
                    "content": r.content,
                    "score": r.score,
                    **r.metadata,
                }
                for r in results
            ],
        }
    )


@router.get("/projects/{project_id}/annotations/file", summary="Get annotations for file")
async def get_annotations_for_file(
    project_id: str,
    file_path: str,
    user: dict = Depends(require_api_key),
) -> JSONResponse:
    """Get all annotations targeting a specific file."""
    project_store = _get_project_store()
    ann_store = _get_annotation_store()

    user_id = user.get("id") or user.get("google_id")

    # Check access
    if not project_store.check_user_has_access(project_id, user_id):
        raise HTTPException(status_code=403, detail="Access denied")

    results = ann_store.get_annotations_for_file(project_id, file_path)

    return JSONResponse(
        content={
            "status": "ok",
            "annotations": [
                {
                    "id": r.id,
                    "content": r.content,
                    **r.metadata,
                }
                for r in results
            ],
        }
    )


@router.get("/projects/{project_id}/annotations/symbol", summary="Get annotations for symbol")
async def get_annotations_for_symbol(
    project_id: str,
    symbol_name: str,
    file_path: Optional[str] = None,
    user: dict = Depends(require_api_key),
) -> JSONResponse:
    """Get all annotations targeting a specific symbol."""
    project_store = _get_project_store()
    ann_store = _get_annotation_store()

    user_id = user.get("id") or user.get("google_id")

    # Check access
    if not project_store.check_user_has_access(project_id, user_id):
        raise HTTPException(status_code=403, detail="Access denied")

    results = ann_store.get_annotations_for_symbol(
        project_id=project_id,
        symbol_name=symbol_name,
        file_path=file_path,
    )

    return JSONResponse(
        content={
            "status": "ok",
            "annotations": [
                {
                    "id": r.id,
                    "content": r.content,
                    **r.metadata,
                }
                for r in results
            ],
        }
    )


@router.get("/projects/{project_id}/annotations/{annotation_id}", summary="Get annotation")
async def get_annotation(
    project_id: str,
    annotation_id: str,
    user: dict = Depends(require_api_key),
) -> JSONResponse:
    """Get a specific annotation by ID."""
    project_store = _get_project_store()
    ann_store = _get_annotation_store()

    user_id = user.get("id") or user.get("google_id")

    # Check access
    if not project_store.check_user_has_access(project_id, user_id):
        raise HTTPException(status_code=403, detail="Access denied")

    annotation = ann_store.get_annotation(project_id, annotation_id)
    if annotation is None:
        raise HTTPException(status_code=404, detail="Annotation not found")

    return JSONResponse(
        content={
            "status": "ok",
            "annotation": annotation,
        }
    )


@router.patch("/projects/{project_id}/annotations/{annotation_id}", summary="Update annotation")
async def update_annotation(
    project_id: str,
    annotation_id: str,
    req: UpdateAnnotationRequest,
    user: dict = Depends(require_api_key),
) -> JSONResponse:
    """Update an annotation."""
    project_store = _get_project_store()
    ann_store = _get_annotation_store()

    user_id = user.get("id") or user.get("google_id")

    # Check access
    if not project_store.check_user_has_access(project_id, user_id):
        raise HTTPException(status_code=403, detail="Access denied")

    # Get existing annotation to check authorship
    existing = ann_store.get_annotation(project_id, annotation_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Annotation not found")

    # Only author or manager can update
    is_author = existing.get("metadata", {}).get("author_id") == user_id
    is_manager = project_store.check_user_can_manage_team(project_id, user_id)

    if not is_author and not is_manager:
        raise HTTPException(status_code=403, detail="Only the author or managers can update")

    success = ann_store.update_annotation(
        project_id=project_id,
        annotation_id=annotation_id,
        content=req.content,
        status=req.status,
        severity=req.severity,
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to update annotation")

    return JSONResponse(
        content={
            "status": "ok",
            "message": "Annotation updated",
        }
    )


@router.delete("/projects/{project_id}/annotations/{annotation_id}", summary="Delete annotation")
async def delete_annotation(
    project_id: str,
    annotation_id: str,
    user: dict = Depends(require_api_key),
) -> JSONResponse:
    """Delete an annotation."""
    project_store = _get_project_store()
    ann_store = _get_annotation_store()

    user_id = user.get("id") or user.get("google_id")

    # Check access
    if not project_store.check_user_has_access(project_id, user_id):
        raise HTTPException(status_code=403, detail="Access denied")

    # Get existing annotation to check authorship
    existing = ann_store.get_annotation(project_id, annotation_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Annotation not found")

    # Only author or manager can delete
    is_author = existing.get("metadata", {}).get("author_id") == user_id
    is_manager = project_store.check_user_can_manage_team(project_id, user_id)

    if not is_author and not is_manager:
        raise HTTPException(status_code=403, detail="Only the author or managers can delete")

    success = ann_store.delete_annotation(project_id, annotation_id)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete annotation")

    # Decrement project annotation count
    project_store.increment_annotation_count(project_id, increment=-1)

    return JSONResponse(
        content={
            "status": "ok",
            "message": "Annotation deleted",
        }
    )


# ---------------------------------------------------------------------------
# Team chat route
# ---------------------------------------------------------------------------

@router.post("/projects/{project_id}/chat", summary="Team chat with annotation context")
async def team_chat(
    project_id: str,
    req: TeamChatRequest,
    user: dict = Depends(require_api_key),
) -> StreamingResponse:
    """Chat with the codebase, including team annotations as context."""
    from src.api.dependencies import get_code_pipeline
    from src.pipelines.code_retrieval import CodeRetrievalPipeline

    project_store = _get_project_store()
    ann_store = _get_annotation_store()

    user_id = user.get("id") or user.get("google_id")
    username = user.get("username") or user.get("name") or user_id

    # Check access
    if not project_store.check_user_has_access(project_id, user_id):
        raise HTTPException(status_code=403, detail="Access denied")

    # Get project for org_id/repo
    project = project_store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get relevant annotations
    relevant_annotations = await ann_store.search_relevant_for_query(
        project_id=project_id,
        query=req.query,
        file_path=req.file_path,
        symbol_name=req.symbol_name,
        top_k=5,
    )

    # Get the code pipeline with project_id for annotation retrieval
    pipeline = get_code_pipeline(
        org_id=project["org_id"],
        repo=project["repo"],
        project_id=project_id,
    )

    # Modify the query to include annotation context if relevant annotations found
    enhanced_query = req.query
    if relevant_annotations:
        ann_context = "\n\nTeam Knowledge:\n"
        for ann in relevant_annotations:
            ann_context += f"- [{ann.get('annotation_type')}] {ann.get('content')[:200]}... "
            ann_context += f"(by {ann.get('author_name')}, {ann.get('author_role')})\n"
        enhanced_query = req.query + ann_context

    # Stream the response
    async def generate():
        # First yield the relevant annotations as context
        if relevant_annotations:
            yield f'{{"type": "annotations", "annotations": {jsonable_encoder(relevant_annotations)}}}\n'

        # Then stream the chat response
        async for chunk in pipeline.run_stream(
            query=enhanced_query,
            user_id=username,
            repo=project["repo"],
            top_k=req.top_k,
        ):
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
    )
