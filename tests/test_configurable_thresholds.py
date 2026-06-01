import sys
import types

# Mock the neo4j package to allow execution without global environment installation
neo4j_mock = types.ModuleType("neo4j")
neo4j_mock.GraphDatabase = object
sys.modules.setdefault("neo4j", neo4j_mock)

import pytest
from src.config import settings
from src.agents.judge import _has_summary_judge_candidates, _deterministic_summary_add
from src.storage.base import SearchResult
from src.graph.neo4j_client import Neo4jClient


def test_summary_judge_respects_custom_settings(monkeypatch):
    # Match with a score of 0.35
    matches = {
        "test item": [
            SearchResult(id="1", content="similar text", score=0.35, metadata={})
        ]
    }

    # Default is 0.4, so score of 0.35 should NOT match
    monkeypatch.setattr(settings, "summary_judge_similarity_threshold", 0.4)
    assert not _has_summary_judge_candidates(matches)

    # If we configure it to 0.3, a score of 0.35 SHOULD match
    monkeypatch.setattr(settings, "summary_judge_similarity_threshold", 0.3)
    assert _has_summary_judge_candidates(matches)

    # Test deterministic addition reason string includes threshold
    monkeypatch.setattr(settings, "summary_judge_similarity_threshold", 0.55)
    result = _deterministic_summary_add(["new summary"])
    assert len(result.operations) == 1
    assert "0.55" in result.operations[0].reason


def test_neo4j_client_respects_custom_settings(monkeypatch):
    monkeypatch.setattr(settings, "temporal_search_similarity_threshold", 0.45)

    # Instantiate Neo4jClient without real connections
    client = Neo4jClient(uri="bolt://localhost:7687", username="neo4j", password="password")

    # Mock embedding function
    client._embedding_fn = lambda text: [0.1, 0.2]

    # Mock session & driver behaviour so query does not hit a real Neo4j server
    class MockSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def run(self, query, **params):
            # Verify that the similarity_threshold parameter passed to the query
            # is indeed custom loaded from settings.temporal_search_similarity_threshold
            assert params["similarity_threshold"] == 0.45
            return []

    monkeypatch.setattr(client, "_session", lambda: MockSession())

    # Trigger search without specifying similarity_threshold explicitly
    client.search_events_by_embedding(user_id="user-1", query_text="yesterday I did VOS")


def test_settings_threshold_boundaries():
    from pydantic import ValidationError
    from src.config.settings import Settings

    # Test valid thresholds
    s = Settings(
        neo4j_password="test",
        summary_judge_similarity_threshold=0.5,
        temporal_search_similarity_threshold=0.1
    )
    assert s.summary_judge_similarity_threshold == 0.5
    assert s.temporal_search_similarity_threshold == 0.1

    # Test out of bounds summary threshold < 0
    with pytest.raises(ValidationError):
        Settings(
            neo4j_password="test",
            summary_judge_similarity_threshold=-0.1,
        )

    # Test out of bounds summary threshold > 1
    with pytest.raises(ValidationError):
        Settings(
            neo4j_password="test",
            summary_judge_similarity_threshold=1.1,
        )

    # Test out of bounds temporal threshold < -1
    with pytest.raises(ValidationError):
        Settings(
            neo4j_password="test",
            temporal_search_similarity_threshold=-1.1,
        )

    # Test out of bounds temporal threshold > 1
    with pytest.raises(ValidationError):
        Settings(
            neo4j_password="test",
            temporal_search_similarity_threshold=1.1,
        )

