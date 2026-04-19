"""Project and team management store for MongoDB."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.config import settings
from src.database.models import Project, TeamMember, TeamRole

logger = logging.getLogger("xmem.database.project_store")


class ProjectStore:
    """MongoDB-backed storage for enterprise projects and team management."""

    def __init__(
        self,
        uri: str = None,
        database: str = None,
    ) -> None:
        self._uri = uri or settings.mongodb_uri
        self._database = database or settings.mongodb_database
        self._client = None
        self._db = None
        self.projects = None
        self.team_members = None
        self._connected = False
        self._in_memory = False

        # In-memory fallback for when MongoDB is unavailable
        self._in_memory_projects: Dict[str, Dict[str, Any]] = {}
        self._in_memory_team_members: Dict[str, List[Dict[str, Any]]] = {}

        # Try to connect, but don't fail if MongoDB is unavailable
        self._try_connect()

    def _try_connect(self) -> None:
        """Attempt to connect to MongoDB, fall back to in-memory if unavailable."""
        try:
            from pymongo import MongoClient, ASCENDING
            from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

            self._client = MongoClient(self._uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self._client.admin.command('ping')
            self._db = self._client[self._database]
            self.projects = self._db["projects"]
            self.team_members = self._db["team_members"]
            self._connected = True
            self._in_memory = False
            logger.info("Connected to MongoDB for project storage")
            self._ensure_indexes()
        except Exception as e:
            logger.warning(f"MongoDB connection failed, using in-memory storage: {e}")
            self._connected = False
            self._in_memory = True

    def _ensure_indexes(self) -> None:
        """Create necessary indexes for the collections."""
        if not self._connected:
            return
        try:
            from pymongo import ASCENDING
            # Project indexes
            self.projects.create_index([("created_by", ASCENDING)])
            self.projects.create_index([("org_id", ASCENDING), ("repo", ASCENDING)])
            self.projects.create_index([("is_active", ASCENDING)])

            # Team member indexes
            self.team_members.create_index([("project_id", ASCENDING), ("user_id", ASCENDING)], unique=True)
            self.team_members.create_index([("project_id", ASCENDING)])
            self.team_members.create_index([("user_id", ASCENDING)])
            self.team_members.create_index([("role", ASCENDING)])
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")

    # ======================================================================
    # Project CRUD
    # ======================================================================

    def create_project(
        self,
        name: str,
        org_id: str,
        repo: str,
        created_by: str,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new enterprise project."""
        now = datetime.utcnow()
        project_doc = {
            "name": name,
            "description": description,
            "org_id": org_id,
            "repo": repo,
            "created_by": created_by,
            "created_at": now,
            "updated_at": now,
            "is_active": True,
            "annotation_count": 0,
        }

        if self._in_memory:
            import uuid
            project_id = str(uuid.uuid4())
            project_doc["_id"] = project_id
            self._in_memory_projects[project_id] = project_doc
            logger.info(f"Created project (memory): {name}")
            return project_doc

        try:
            result = self.projects.insert_one(project_doc)
            project_doc["_id"] = result.inserted_id
            logger.info(f"Created project: {name} (id={result.inserted_id})")
            return project_doc
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            return None

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get a project by ID."""
        if self._in_memory:
            return self._in_memory_projects.get(project_id)

        try:
            from bson import ObjectId
            return self.projects.find_one({"_id": ObjectId(project_id)})
        except Exception as e:
            logger.error(f"Failed to get project: {e}")
            return None

    def list_projects_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """List all projects where user is a member or creator."""
        if self._in_memory:
            # Get projects created by user
            created = [
                p for p in self._in_memory_projects.values()
                if p.get("created_by") == user_id
            ]
            # Get projects where user is a team member
            member_projects = []
            for project_id, members in self._in_memory_team_members.items():
                for m in members:
                    if m.get("user_id") == user_id and m.get("is_active", True):
                        project = self._in_memory_projects.get(project_id)
                        if project:
                            member_projects.append(project)
            # Combine and deduplicate
            seen = set()
            result = []
            for p in created + member_projects:
                pid = p.get("_id")
                if pid not in seen:
                    seen.add(pid)
                    result.append(p)
            return result

        try:
            from bson import ObjectId
            # Find projects created by user
            created = list(self.projects.find({"created_by": user_id}))

            # Find projects where user is a team member
            member_project_ids = [
                m["project_id"] for m in self.team_members.find({
                    "user_id": user_id,
                    "is_active": True
                })
            ]
            member_projects = list(self.projects.find({
                "_id": {"$in": [ObjectId(pid) for pid in member_project_ids]}
            }))

            # Combine and deduplicate
            seen = set()
            result = []
            for p in created + member_projects:
                pid = str(p.get("_id"))
                if pid not in seen:
                    seen.add(pid)
                    result.append(p)
            return result
        except Exception as e:
            logger.error(f"Failed to list projects: {e}")
            return []

    def update_project(
        self,
        project_id: str,
        updates: Dict[str, Any],
    ) -> bool:
        """Update project fields."""
        updates["updated_at"] = datetime.utcnow()

        if self._in_memory:
            project = self._in_memory_projects.get(project_id)
            if project:
                project.update(updates)
                return True
            return False

        try:
            from bson import ObjectId
            result = self.projects.update_one(
                {"_id": ObjectId(project_id)},
                {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update project: {e}")
            return False

    def increment_annotation_count(self, project_id: str, increment: int = 1) -> bool:
        """Increment the annotation count for a project."""
        if self._in_memory:
            project = self._in_memory_projects.get(project_id)
            if project:
                project["annotation_count"] = project.get("annotation_count", 0) + increment
                project["updated_at"] = datetime.utcnow()
                return True
            return False

        try:
            from bson import ObjectId
            result = self.projects.update_one(
                {"_id": ObjectId(project_id)},
                {"$inc": {"annotation_count": increment}, "$set": {"updated_at": datetime.utcnow()}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to increment annotation count: {e}")
            return False

    def delete_project(self, project_id: str) -> bool:
        """Delete a project and all its team members."""
        if self._in_memory:
            if project_id in self._in_memory_projects:
                del self._in_memory_projects[project_id]
                self._in_memory_team_members.pop(project_id, None)
                return True
            return False

        try:
            from bson import ObjectId
            # Delete project
            result = self.projects.delete_one({"_id": ObjectId(project_id)})
            # Delete all team members
            self.team_members.delete_many({"project_id": project_id})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to delete project: {e}")
            return False

    # ======================================================================
    # Team Member CRUD
    # ======================================================================

    def add_team_member(
        self,
        project_id: str,
        user_id: str,
        username: str,
        email: Optional[str],
        role: TeamRole,
        added_by: str,
    ) -> Dict[str, Any]:
        """Add a team member to a project."""
        member_doc = {
            "project_id": project_id,
            "user_id": user_id,
            "username": username,
            "email": email,
            "role": role.value,
            "added_by": added_by,
            "added_at": datetime.utcnow(),
            "is_active": True,
        }

        if self._in_memory:
            import uuid
            member_id = str(uuid.uuid4())
            member_doc["_id"] = member_id
            if project_id not in self._in_memory_team_members:
                self._in_memory_team_members[project_id] = []
            # Check if already exists
            existing = next(
                (m for m in self._in_memory_team_members[project_id] if m["user_id"] == user_id),
                None
            )
            if existing:
                existing.update(member_doc)
            else:
                self._in_memory_team_members[project_id].append(member_doc)
            logger.info(f"Added team member (memory): {username} to project {project_id}")
            return member_doc

        try:
            from pymongo.errors import DuplicateKeyError
            result = self.team_members.insert_one(member_doc)
            member_doc["_id"] = result.inserted_id
            logger.info(f"Added team member: {username} to project {project_id}")
            return member_doc
        except DuplicateKeyError:
            logger.warning(f"Team member already exists: {user_id} in project {project_id}")
            # Update existing
            return self.update_team_member_role(project_id, user_id, role)
        except Exception as e:
            logger.error(f"Failed to add team member: {e}")
            return None

    def get_team_member(self, project_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific team member."""
        if self._in_memory:
            members = self._in_memory_team_members.get(project_id, [])
            return next((m for m in members if m["user_id"] == user_id), None)

        try:
            return self.team_members.find_one({
                "project_id": project_id,
                "user_id": user_id,
            })
        except Exception as e:
            logger.error(f"Failed to get team member: {e}")
            return None

    def list_team_members(self, project_id: str) -> List[Dict[str, Any]]:
        """List all active team members for a project."""
        if self._in_memory:
            members = self._in_memory_team_members.get(project_id, [])
            return [m for m in members if m.get("is_active", True)]

        try:
            return list(self.team_members.find({
                "project_id": project_id,
                "is_active": True,
            }))
        except Exception as e:
            logger.error(f"Failed to list team members: {e}")
            return []

    def update_team_member_role(
        self,
        project_id: str,
        user_id: str,
        role: TeamRole,
    ) -> Dict[str, Any]:
        """Update a team member's role."""
        if self._in_memory:
            members = self._in_memory_team_members.get(project_id, [])
            member = next((m for m in members if m["user_id"] == user_id), None)
            if member:
                member["role"] = role.value
                return member
            return None

        try:
            result = self.team_members.find_one_and_update(
                {"project_id": project_id, "user_id": user_id},
                {"$set": {"role": role.value}},
                return_document=True,
            )
            return result
        except Exception as e:
            logger.error(f"Failed to update team member role: {e}")
            return None

    def remove_team_member(self, project_id: str, user_id: str) -> bool:
        """Remove (deactivate) a team member."""
        if self._in_memory:
            members = self._in_memory_team_members.get(project_id, [])
            member = next((m for m in members if m["user_id"] == user_id), None)
            if member:
                member["is_active"] = False
                return True
            return False

        try:
            result = self.team_members.update_one(
                {"project_id": project_id, "user_id": user_id},
                {"$set": {"is_active": False}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to remove team member: {e}")
            return False

    def get_user_role_in_project(self, project_id: str, user_id: str) -> Optional[TeamRole]:
        """Get the role of a user in a project."""
        member = self.get_team_member(project_id, user_id)
        if member and member.get("is_active", True):
            role_str = member.get("role", "intern")
            try:
                return TeamRole(role_str)
            except ValueError:
                return TeamRole.INTERN
        return None

    def check_user_can_annotate(self, project_id: str, user_id: str) -> bool:
        """Check if user can create annotations (all roles except INTERN)."""
        role = self.get_user_role_in_project(project_id, user_id)
        if role is None:
            return False
        # All roles except INTERN can create annotations
        return role != TeamRole.INTERN

    def check_user_can_manage_team(self, project_id: str, user_id: str) -> bool:
        """Check if user can manage team (manager only)."""
        role = self.get_user_role_in_project(project_id, user_id)
        return role == TeamRole.MANAGER

    def check_user_can_edit_team_member(
        self, project_id: str, user_id: str, target_user_id: str
    ) -> bool:
        """
        Check if user can edit another team member's role.
        - Managers can edit anyone
        - Staff Engineers can edit SDE2 and Intern
        - SDE2 and Intern cannot edit anyone
        """
        # Can't edit yourself
        if user_id == target_user_id:
            return False

        user_role = self.get_user_role_in_project(project_id, user_id)
        target_role = self.get_user_role_in_project(project_id, target_user_id)

        if target_role is None:
            return False

        # Manager can edit anyone
        if user_role == TeamRole.MANAGER:
            return True

        # Staff Engineer can edit SDE2 (2) and Intern (1)
        if user_role == TeamRole.STAFF_ENGINEER:
            role_hierarchy = {
                TeamRole.MANAGER: 4,
                TeamRole.STAFF_ENGINEER: 3,
                TeamRole.SDE2: 2,
                TeamRole.INTERN: 1,
            }
            user_level = role_hierarchy.get(user_role, 0)
            target_level = role_hierarchy.get(target_role, 0)
            return user_level > target_level

        return False

    def check_user_is_project_creator(self, project_id: str, user_id: str) -> bool:
        """Check if user created the project."""
        project = self.get_project(project_id)
        if project:
            return project.get("created_by") == user_id
        return False

    def check_user_has_access(self, project_id: str, user_id: str) -> bool:
        """Check if user has any access to the project."""
        # Check if creator
        if self.check_user_is_project_creator(project_id, user_id):
            return True
        # Check if team member
        role = self.get_user_role_in_project(project_id, user_id)
        return role is not None

    def close(self) -> None:
        """Close the MongoDB connection."""
        if self._client:
            self._client.close()
