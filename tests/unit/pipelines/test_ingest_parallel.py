
import asyncio
import time
import sys
import unittest
from unittest.mock import MagicMock, AsyncMock

# Mock dependencies before importing anything from src
sys.modules['pinecone'] = MagicMock()
sys.modules['neo4j'] = MagicMock()
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()
sys.modules['google.genai.types'] = MagicMock()
sys.modules['langchain_core'] = MagicMock()
# Mock langchain_core packages
sys.modules['langchain_core.language_models'] = MagicMock()
sys.modules['langchain_core.messages'] = MagicMock()
sys.modules['langchain_google_genai'] = MagicMock()
sys.modules['langchain_anthropic'] = MagicMock()
sys.modules['langchain_openai'] = MagicMock()
sys.modules['langgraph'] = MagicMock()
sys.modules['langgraph.graph'] = MagicMock()
sys.modules['langgraph.types'] = MagicMock()
sys.modules['pydantic'] = MagicMock()
sys.modules['pydantic_settings'] = MagicMock()
sys.modules['dotenv'] = MagicMock()

# Mock internal heavy modules
sys.modules['src.storage.pinecone'] = MagicMock()
sys.modules['src.graph.neo4j_client'] = MagicMock()
sys.modules['src.models'] = MagicMock()

# Mock agents
sys.modules['src.agents.classifier'] = MagicMock()
sys.modules['src.agents.image'] = MagicMock()
sys.modules['src.agents.judge'] = MagicMock()
sys.modules['src.agents.profiler'] = MagicMock()
sys.modules['src.agents.summarizer'] = MagicMock()
sys.modules['src.agents.temporal'] = MagicMock()
sys.modules['src.agents.verification'] = MagicMock()
sys.modules['src.pipelines.weaver'] = MagicMock()

# Mock TypedDict for IngestState
from typing import TypedDict  # noqa: E402
sys.modules['typing_extensions'] = MagicMock()
sys.modules['typing_extensions'].TypedDict = TypedDict

# Mock settings
settings_mock = MagicMock()
settings_mock.pinecone_api_key = "dummy"
settings_mock.neo4j_password = "dummy"
settings_mock.gemini_api_key = "dummy"
settings_mock.embedding_model = "dummy"
settings_mock.pinecone_dimension = 1536
sys.modules['src.config'] = MagicMock()
sys.modules['src.config'].settings = settings_mock

# Now import IngestPipeline
from src.pipelines.ingest import IngestPipeline  # noqa: E402

class TestIngestParallel(unittest.TestCase):
    def test_extract_profile_parallel(self):
        """Test that _node_extract_profile runs queries in parallel."""
        asyncio.run(self._test_extract_profile_parallel())

    async def _test_extract_profile_parallel(self):
        # Setup pipeline
        pipeline = IngestPipeline()

        # Configure mock profiler to be slow
        delay = 0.1
        async def slow_arun(state):
            await asyncio.sleep(delay)
            res = MagicMock()
            res.is_empty = False
            res.facts = [MagicMock()]
            res.facts[0].model_dump.return_value = {"content": "fact"}
            return res

        pipeline.profiler.arun = AsyncMock(side_effect=slow_arun)
        pipeline.judge.arun = AsyncMock(return_value=MagicMock())
        pipeline.weaver.execute = AsyncMock(return_value=MagicMock())

        # 5 queries
        num_queries = 5
        state = {
            "profile_queries": [f"query_{i}" for i in range(num_queries)],
            "user_id": "test_user",
        }

        start_time = time.perf_counter()
        await pipeline._node_extract_profile(state)
        duration = time.perf_counter() - start_time

        # If sequential, it would take ~0.5s. If parallel, ~0.1s.
        # We verify it's less than 0.25s (allowing for some overhead)
        print(f"\nDuration: {duration:.4f}s")
        self.assertLess(duration, delay * num_queries * 0.8, "Execution was not parallel enough")

    def test_extract_temporal_parallel(self):
        """Test that _node_extract_temporal runs queries in parallel."""
        asyncio.run(self._test_extract_temporal_parallel())

    async def _test_extract_temporal_parallel(self):
        # Setup pipeline
        pipeline = IngestPipeline()

        # Configure mock temporal agent
        delay = 0.1
        async def slow_arun(state):
            await asyncio.sleep(delay)
            res = MagicMock()
            res.is_empty = False
            res.event = MagicMock()
            res.event.date = "2023-01-01"
            res.event.event_name = "test"
            res.event.desc = "desc"
            res.event.year = "2023"
            res.event.time = "12:00"
            res.event.date_expression = "today"
            return res

        pipeline.temporal.arun = AsyncMock(side_effect=slow_arun)
        pipeline.judge.arun = AsyncMock(return_value=MagicMock())
        pipeline.weaver.execute = AsyncMock(return_value=MagicMock())

        # 5 queries
        num_queries = 5
        state = {
            "temporal_queries": [f"query_{i}" for i in range(num_queries)],
            "user_id": "test_user",
            "session_datetime": "2023-01-01T12:00:00"
        }

        start_time = time.perf_counter()
        await pipeline._node_extract_temporal(state)
        duration = time.perf_counter() - start_time

        print(f"\nDuration: {duration:.4f}s")
        self.assertLess(duration, delay * num_queries * 0.8, "Execution was not parallel enough")

if __name__ == "__main__":
    unittest.main()
