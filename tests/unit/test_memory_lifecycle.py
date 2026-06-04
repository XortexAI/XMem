from __future__ import annotations

from src.storage.local import SQLiteVectorStore
from src.storage.memory_lifecycle import (
    CONTENT_HASH_KEY,
    FORGET_REASON_KEY,
    FORGOTTEN_AT_KEY,
    IS_CURRENT_KEY,
    PARENT_MEMORY_ID_KEY,
    VERSION_KEY,
    compute_memory_hash,
)


def _store(tmp_path):
    return SQLiteVectorStore(
        path=str(tmp_path / "vectors.sqlite3"),
        namespace="test",
        dimension=3,
    )


def test_sqlite_add_reuses_current_memory_with_same_normalized_hash(tmp_path):
    store = _store(tmp_path)

    first_ids = store.add(
        ["Remember that Alice likes XMem."],
        [[1.0, 0.0, 0.0]],
        ids=["memory-1"],
        metadata=[{"user_id": "alice"}],
    )
    duplicate_ids = store.add(
        [" remember   THAT alice likes xmem. "],
        [[0.0, 1.0, 0.0]],
        ids=["memory-duplicate"],
        metadata=[{"user_id": "alice"}],
    )

    assert first_ids == ["memory-1"]
    assert duplicate_ids == ["memory-1"]
    assert store.search_by_metadata({"user_id": "alice"}, top_k=10)[0].id == "memory-1"

    stored = store.get(["memory-1"])[0]
    assert stored["metadata"][CONTENT_HASH_KEY] == compute_memory_hash("Remember that Alice likes XMem.")
    assert stored["metadata"][VERSION_KEY] == 1
    assert stored["metadata"][IS_CURRENT_KEY] is True


def test_sqlite_hash_dedup_is_scoped_by_user_id(tmp_path):
    store = _store(tmp_path)

    alice_ids = store.add(
        ["Shared wording."],
        [[1.0, 0.0, 0.0]],
        ids=["alice-memory"],
        metadata=[{"user_id": "alice"}],
    )
    bob_ids = store.add(
        ["shared wording."],
        [[0.0, 1.0, 0.0]],
        ids=["bob-memory"],
        metadata=[{"user_id": "bob"}],
    )

    assert alice_ids == ["alice-memory"]
    assert bob_ids == ["bob-memory"]
    assert [r.id for r in store.search_by_metadata({"user_id": "alice"}, top_k=10)] == ["alice-memory"]
    assert [r.id for r in store.search_by_metadata({"user_id": "bob"}, top_k=10)] == ["bob-memory"]


def test_sqlite_update_rejects_current_hash_collision(tmp_path):
    store = _store(tmp_path)
    store.add(
        ["Alice works at XMem."],
        [[1.0, 0.0, 0.0]],
        ids=["profile-1"],
        metadata=[{"user_id": "alice"}],
    )
    store.add(
        ["Alice works at XortexAI."],
        [[0.0, 1.0, 0.0]],
        ids=["profile-2"],
        metadata=[{"user_id": "alice"}],
    )

    assert store.update("profile-1", text="Alice works at XortexAI.") is False

    visible = store.search_by_metadata({"user_id": "alice"}, top_k=10)
    assert {result.id for result in visible} == {"profile-1", "profile-2"}
    assert store.get(["profile-1"])[0]["content"] == "Alice works at XMem."


def test_lifecycle_fields_cannot_be_overridden_by_caller_metadata(tmp_path):
    store = _store(tmp_path)

    ids = store.add(
        ["Visible memory."],
        [[1.0, 0.0, 0.0]],
        ids=["visible-1"],
        metadata=[{
            "user_id": "alice",
            CONTENT_HASH_KEY: "caller-hash",
            IS_CURRENT_KEY: False,
            FORGOTTEN_AT_KEY: "2024-01-01T00:00:00+00:00",
        }],
    )

    assert ids == ["visible-1"]
    stored = store.get(["visible-1"])[0]
    assert stored["metadata"][CONTENT_HASH_KEY] == compute_memory_hash("Visible memory.")
    assert stored["metadata"][IS_CURRENT_KEY] is True
    assert stored["metadata"][FORGOTTEN_AT_KEY] is None
    assert store.search_by_metadata({"user_id": "alice"}, top_k=10)[0].id == "visible-1"


def test_sqlite_add_version_supersedes_parent_but_keeps_history(tmp_path):
    store = _store(tmp_path)
    store.add(
        ["Alice works at XMem."],
        [[1.0, 0.0, 0.0]],
        ids=["profile-1"],
        metadata=[{"user_id": "alice", "domain": "profile"}],
    )

    version_id = store.add_version(
        "profile-1",
        "Alice works at XortexAI.",
        [0.0, 1.0, 0.0],
        id="profile-2",
        metadata={"user_id": "alice", "domain": "profile"},
    )

    assert version_id == "profile-2"
    parent = store.get(["profile-1"])[0]
    child = store.get(["profile-2"])[0]
    assert parent["metadata"][IS_CURRENT_KEY] is False
    assert child["metadata"][PARENT_MEMORY_ID_KEY] == "profile-1"
    assert child["metadata"][VERSION_KEY] == 2

    visible = store.search_by_metadata({"user_id": "alice"}, top_k=10)
    assert [result.id for result in visible] == ["profile-2"]


def test_sqlite_add_version_duplicate_supersedes_parent(tmp_path):
    store = _store(tmp_path)
    store.add(
        ["Alice works at XMem."],
        [[1.0, 0.0, 0.0]],
        ids=["profile-1"],
        metadata=[{"user_id": "alice", "domain": "profile"}],
    )
    store.add(
        ["Alice works at XortexAI."],
        [[0.0, 1.0, 0.0]],
        ids=["profile-existing"],
        metadata=[{"user_id": "alice", "domain": "profile"}],
    )

    version_id = store.add_version(
        "profile-1",
        "Alice works at XortexAI.",
        [0.0, 0.0, 1.0],
        id="profile-2",
        metadata={"user_id": "alice", "domain": "profile"},
    )

    assert version_id == "profile-existing"
    assert store.get(["profile-1"])[0]["metadata"][IS_CURRENT_KEY] is False
    visible = store.search_by_metadata({"user_id": "alice"}, top_k=10)
    assert [result.id for result in visible] == ["profile-existing"]


def test_sqlite_add_version_rejects_forgotten_parent(tmp_path):
    store = _store(tmp_path)
    store.add(
        ["Alice revoked this memory."],
        [[1.0, 0.0, 0.0]],
        ids=["profile-1"],
        metadata=[{"user_id": "alice", "domain": "profile"}],
    )
    store.forget(["profile-1"], reason="user requested deletion")

    version_id = store.add_version(
        "profile-1",
        "Alice revoked this memory but changed.",
        [0.0, 1.0, 0.0],
        id="profile-2",
        metadata={"user_id": "alice", "domain": "profile"},
    )

    assert version_id is None
    assert store.get(["profile-2"]) == []
    assert store.search_by_metadata({"user_id": "alice"}, top_k=10) == []


def test_sqlite_add_version_rejects_superseded_parent(tmp_path):
    store = _store(tmp_path)
    store.add(
        ["Alice works at XMem."],
        [[1.0, 0.0, 0.0]],
        ids=["profile-1"],
        metadata=[{"user_id": "alice", "domain": "profile"}],
    )
    store.add_version(
        "profile-1",
        "Alice works at XortexAI.",
        [0.0, 1.0, 0.0],
        id="profile-2",
        metadata={"user_id": "alice", "domain": "profile"},
    )

    version_id = store.add_version(
        "profile-1",
        "Alice works somewhere else.",
        [0.0, 0.0, 1.0],
        id="profile-3",
        metadata={"user_id": "alice", "domain": "profile"},
    )

    assert version_id is None
    assert store.get(["profile-3"]) == []
    visible = store.search_by_metadata({"user_id": "alice"}, top_k=10)
    assert [result.id for result in visible] == ["profile-2"]


def test_sqlite_forget_soft_deletes_memory_from_retrieval(tmp_path):
    store = _store(tmp_path)
    store.add(
        ["Alice's temporary preference."],
        [[1.0, 0.0, 0.0]],
        ids=["temp-1"],
        metadata=[{"user_id": "alice"}],
    )

    assert store.forget(["temp-1"], reason="user requested deletion") is True

    assert store.search_by_metadata({"user_id": "alice"}, top_k=10) == []
    assert store.search([1.0, 0.0, 0.0], top_k=10) == []

    stored = store.get(["temp-1"])[0]
    assert stored["metadata"][IS_CURRENT_KEY] is False
    assert stored["metadata"][FORGOTTEN_AT_KEY]
    assert stored["metadata"][FORGET_REASON_KEY] == "user requested deletion"
