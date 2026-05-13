from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.database import api_key_store as api_key_store_module
from src.database.api_key_store import APIKeyStore, _in_memory_api_keys
from src.database.project_store import ProjectStore
from src.database.models import TeamRole
from src.database.user_store import UserStore, _in_memory_users


def _force_api_key_memory(self):
    self._connected = False
    self._in_memory = True
    self.api_keys = None


class _FailingApiKeyCollection:
    def insert_one(self, _doc):
        raise RuntimeError("database unavailable")


def _force_api_key_connected_with_write_failure(self):
    self._connected = True
    self._in_memory = False
    self.api_keys = _FailingApiKeyCollection()


def _force_user_memory(self):
    self._connected = False
    self._in_memory = True
    self.users = None


def _force_project_memory(self):
    self._connected = False
    self._in_memory = True


def test_api_key_store_creates_validates_updates_and_revokes_in_memory(monkeypatch):
    _in_memory_api_keys.clear()
    monkeypatch.setattr(APIKeyStore, "_try_connect", _force_api_key_memory)
    store = APIKeyStore()

    created = store.create_api_key("user-1", name="CI")
    assert created["key"].startswith("xmem_")

    validated = store.validate_api_key(created["key"])
    assert validated["user_id"] == "user-1"
    assert "key_hash" not in validated

    scoped = store.create_api_key(
        "user-1",
        name="Project",
        scopes=["memory:read"],
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        org_id="org-1",
        project_id="proj-1",
    )
    scoped_doc = store.validate_api_key(scoped["key"])
    assert scoped_doc["scopes"] == ["memory:read"]
    assert scoped_doc["org_id"] == "org-1"
    assert scoped_doc["project_id"] == "proj-1"

    expired = store.create_api_key(
        "user-1",
        name="Expired",
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    assert store.validate_api_key(expired["key"]) is None

    assert store.update_api_key_name("user-1", created["key_id"], "Local Dev")
    listed = next(
        key for key in store.get_user_api_keys("user-1") if key["id"] == created["key_id"]
    )
    assert listed["name"] == "Local Dev"

    assert store.revoke_api_key("user-1", created["key_id"])
    assert store.validate_api_key(created["key"]) is None


def test_api_key_store_refuses_write_fallback_in_remote_production(monkeypatch):
    monkeypatch.setattr(APIKeyStore, "_try_connect", _force_api_key_connected_with_write_failure)
    monkeypatch.setattr(api_key_store_module.settings, "environment", "production")
    monkeypatch.setattr(
        api_key_store_module.settings,
        "mongodb_uri",
        "mongodb://mongo.example:27017",
    )

    store = APIKeyStore()
    with pytest.raises(RuntimeError, match="Durable MongoDB storage is required"):
        store.create_api_key("user-1")


def test_user_store_get_or_create_and_username_helpers_in_memory(monkeypatch):
    _in_memory_users.clear()
    monkeypatch.setattr(UserStore, "_try_connect", _force_user_memory)
    store = UserStore()

    user = store.get_or_create_user(
        google_id="google-1",
        email="alice@example.com",
        name="Alice",
    )
    assert user["email"] == "alice@example.com"
    assert store.get_or_create_user("google-1", "ignored@example.com", "Alice")["_id"] == "google-1"

    assert store.set_username("google-1", "alice_dev")
    assert store.get_user_by_username("alice_dev")["_id"] == "google-1"
    assert not store.is_username_available("alice_dev")
    assert not store.set_username("google-1", "x")


def test_project_store_team_permissions_in_memory(monkeypatch):
    monkeypatch.setattr(ProjectStore, "_try_connect", _force_project_memory)
    store = ProjectStore()

    project = store.create_project(
        name="Payments",
        org_id="acme",
        repo="payments",
        created_by="manager-1",
    )
    project_id = project["_id"]

    store.add_team_member(project_id, "manager-1", "manager", "m@example.com", TeamRole.MANAGER, "manager-1")
    store.add_team_member(project_id, "intern-1", "intern", "i@example.com", TeamRole.INTERN, "manager-1")
    store.add_team_member(project_id, "sde-1", "sde", "s@example.com", TeamRole.SDE2, "manager-1")

    assert store.check_user_has_access(project_id, "intern-1")
    assert not store.check_user_can_annotate(project_id, "intern-1")
    assert store.check_user_can_annotate(project_id, "sde-1")
    assert store.check_user_can_manage_team(project_id, "manager-1")
    assert store.check_user_can_edit_team_member(project_id, "manager-1", "intern-1")

    assert store.increment_annotation_count(project_id, 2)
    assert store.get_project(project_id)["annotation_count"] == 2
