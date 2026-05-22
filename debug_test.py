import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.api.app import create_app

app = create_app()
client = TestClient(app)

with patch("src.api.routes.memory.require_api_key", return_value={"username": "test_user"}):
    from src.api.dependencies import require_api_key, enforce_rate_limit, require_ready
    app.dependency_overrides[require_api_key] = lambda: {"username": "test_user"}
    app.dependency_overrides[enforce_rate_limit] = lambda: True
    app.dependency_overrides[require_ready] = lambda: True

    payload = {
        "items": [
            {
                "user_query": "Hello world",
                "agent_response": "Hi there",
                "user_id": "test_user_1",
            }
        ]
    }

    try:
        response = client.post(
            "/v1/memory/batch-ingest",
            json=payload,
            headers={"Authorization": "Bearer test-key"}
        )
        print("Status code:", response.status_code)
        import json
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print("Exception:", e)
