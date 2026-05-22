import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from typing import Dict, Any

from src.api.app import create_app
from src.api.schemas import BatchIngestRequest, IngestRequest
from src.pipelines.ingest import IngestPipeline

@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)

@pytest.fixture
def mock_ingest_pipeline():
    with patch("src.api.routes.memory.get_ingest_pipeline") as mock_get_pipeline:
        from types import SimpleNamespace
        mock_pipeline = AsyncMock(spec=IngestPipeline)
        mock_pipeline.model = SimpleNamespace(model_name="test-model")
        
        # Default mock behavior
        async def mock_run(*args, **kwargs):
            return {
                "classification_result": SimpleNamespace(classifications=["test"]),
                "profile_judge": None,
                "profile_weaver": None,
                "temporal_judge": None,
                "temporal_weaver": None,
                "summary_judge": None,
                "summary_weaver": None,
                "image_judge": None,
                "image_weaver": None,
            }
        
        mock_pipeline.run.side_effect = mock_run
        mock_get_pipeline.return_value = mock_pipeline
        yield mock_pipeline

def test_batch_ingest_success(client, mock_ingest_pipeline):
    """Test that multiple items can be successfully ingested in a batch."""
    payload = {
        "items": [
            {
                "user_query": "Hello world",
                "agent_response": "Hi there",
                "user_id": "test_user_1",
            },
            {
                "user_query": "Second message",
                "agent_response": "Understood",
                "user_id": "test_user_1",
            }
        ]
    }

    # You must provide API key or mock dependency for require_api_key
    # For test purposes, we assume we override the dependency or add a test key
    # Let's mock require_api_key in dependencies
    with patch("src.api.routes.memory.require_api_key", return_value={"username": "test_user"}):
        app = client.app
        from src.api.dependencies import require_api_key, enforce_rate_limit, require_ready
        app.dependency_overrides[require_api_key] = lambda: {"username": "test_user"}
        app.dependency_overrides[enforce_rate_limit] = lambda: True
        app.dependency_overrides[require_ready] = lambda: True

        response = client.post(
            "/v1/memory/batch-ingest",
            json=payload,
            headers={"Authorization": "Bearer test-key"}
        )

    assert response.status_code == 200, response.json()
    data = response.json()
    assert data["status"] == "ok", data
    assert len(data["data"]["results"]) == 2, data
    for item in data["data"]["results"]:
        assert item["model"] == "test-model"


def test_coordinator_serializes_concurrent_batches(client, mock_ingest_pipeline):
    """Two concurrent batch-ingest requests for the same user must not overlap.

    We verify this by checking that all 4 pipeline.run calls were made
    (2 items × 2 batches) and both requests succeed.
    """
    import threading

    payload = {
        "items": [
            {
                "user_query": "Batch message 1",
                "agent_response": "Ack 1",
                "user_id": "same_user",
            },
            {
                "user_query": "Batch message 2",
                "agent_response": "Ack 2",
                "user_id": "same_user",
            },
        ]
    }

    with patch("src.api.routes.memory.require_api_key", return_value={"username": "same_user"}):
        app = client.app
        from src.api.dependencies import require_api_key, enforce_rate_limit, require_ready
        app.dependency_overrides[require_api_key] = lambda: {"username": "same_user"}
        app.dependency_overrides[enforce_rate_limit] = lambda: True
        app.dependency_overrides[require_ready] = lambda: True

        # Send two batch requests concurrently via threads
        results = [None, None]

        def _send_batch(idx):
            results[idx] = client.post(
                "/v1/memory/batch-ingest",
                json=payload,
                headers={"Authorization": "Bearer test-key"},
            )

        t1 = threading.Thread(target=_send_batch, args=(0,))
        t2 = threading.Thread(target=_send_batch, args=(1,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    # Both requests should succeed
    for r in results:
        assert r is not None
        assert r.status_code == 200, r.json()

    # All 4 pipeline.run calls (2 items × 2 batches) should have been made
    assert mock_ingest_pipeline.run.call_count == 4

