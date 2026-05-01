import importlib.util
import os
import sys
import types
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("NEO4J_PASSWORD", "test-neo4j-password")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")

config_module = types.ModuleType("src.config")
config_module.settings = types.SimpleNamespace(
    mongodb_uri="mongodb://localhost:27017",
    mongodb_database="xmem-test",
)
sys.modules.setdefault("src.config", config_module)

API_KEY_STORE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "database" / "api_key_store.py"
)
api_key_store_spec = importlib.util.spec_from_file_location(
    "api_key_store_under_test",
    API_KEY_STORE_PATH,
)
api_key_store_module = importlib.util.module_from_spec(api_key_store_spec)
api_key_store_spec.loader.exec_module(api_key_store_module)
APIKeyStore = api_key_store_module.APIKeyStore


class FakeUpdateResult:
    def __init__(self, modified_count=1):
        self.modified_count = modified_count


class FakeAPIKeysCollection:
    def __init__(self, doc):
        self.doc = doc
        self.find_one_calls = 0
        self.update_one_calls = []

    def find_one(self, query):
        self.find_one_calls += 1
        if (
            self.doc
            and self.doc.get("key_hash") == query.get("key_hash")
            and self.doc.get("is_active") == query.get("is_active")
        ):
            return dict(self.doc)
        return None

    def update_one(self, query, update):
        self.update_one_calls.append((query, update))
        if self.doc and self.doc.get("_id") == query.get("_id"):
            self.doc.update(update.get("$set", {}))
            return FakeUpdateResult()
        return FakeUpdateResult(modified_count=0)


def build_store(collection):
    store = object.__new__(APIKeyStore)
    store._in_memory = False
    store._connected = True
    store.api_keys = collection
    store._validation_cache = {}
    return store


def key_doc_for(store, key, **overrides):
    doc = {
        "_id": "key-1",
        "user_id": "user-1",
        "key_hash": store._hash_key(key),
        "key_prefix": key[:8],
        "name": "Primary",
        "created_at": datetime(2026, 1, 1),
        "last_used": None,
        "is_active": True,
    }
    doc.update(overrides)
    return doc


def test_validate_api_key_caches_active_database_key(monkeypatch):
    monkeypatch.setattr(api_key_store_module, "VALIDATION_CACHE_TTL_SECONDS", 60)
    store = build_store(None)
    key = "xmem_test-key"
    collection = FakeAPIKeysCollection(key_doc_for(store, key))
    store.api_keys = collection

    first = store.validate_api_key(key)
    second = store.validate_api_key(key)

    assert collection.find_one_calls == 1
    assert len(collection.update_one_calls) == 1
    assert first["id"] == "key-1"
    assert second["id"] == "key-1"
    assert "key_hash" not in first
    assert "key_hash" not in second

    second["name"] = "mutated caller copy"
    third = store.validate_api_key(key)
    assert third["name"] == "Primary"


def test_validate_api_key_requeries_after_cache_expiry(monkeypatch):
    monkeypatch.setattr(api_key_store_module, "VALIDATION_CACHE_TTL_SECONDS", -1)
    store = build_store(None)
    key = "xmem_test-key"
    collection = FakeAPIKeysCollection(key_doc_for(store, key))
    store.api_keys = collection

    store.validate_api_key(key)
    store.validate_api_key(key)

    assert collection.find_one_calls == 2
    assert len(collection.update_one_calls) == 2


def test_validate_api_key_does_not_cache_missing_or_inactive_key(monkeypatch):
    monkeypatch.setattr(api_key_store_module, "VALIDATION_CACHE_TTL_SECONDS", 60)
    store = build_store(None)
    key = "xmem_test-key"
    collection = FakeAPIKeysCollection(key_doc_for(store, key, is_active=False))
    store.api_keys = collection

    assert store.validate_api_key(key) is None
    assert store.validate_api_key(key) is None

    assert collection.find_one_calls == 2
    assert collection.update_one_calls == []
