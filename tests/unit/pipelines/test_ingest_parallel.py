import unittest
from unittest.mock import MagicMock, AsyncMock
import asyncio
import time
import sys

# Mock dependencies before import
sys.modules["google"] = MagicMock()
sys.modules["google.genai"] = MagicMock()
sys.modules["google.genai.types"] = MagicMock()

sys.modules["langgraph"] = MagicMock()
sys.modules["langgraph.graph"] = MagicMock()
sys.modules["langgraph.types"] = MagicMock()

sys.modules["src.storage.pinecone"] = MagicMock()
sys.modules["src.graph.neo4j_client"] = MagicMock()
sys.modules["src.pipelines.weaver"] = MagicMock()
sys.modules["src.agents.classifier"] = MagicMock()
sys.modules["src.agents.image"] = MagicMock()
sys.modules["src.agents.judge"] = MagicMock()
sys.modules["src.agents.profiler"] = MagicMock()
sys.modules["src.agents.summarizer"] = MagicMock()
sys.modules["src.agents.temporal"] = MagicMock()
sys.modules["src.models"] = MagicMock()
sys.modules["src.graph.schema"] = MagicMock()
sys.modules["src.schemas.classification"] = MagicMock()
sys.modules["src.schemas.events"] = MagicMock()
sys.modules["src.schemas.image"] = MagicMock()
sys.modules["src.schemas.judge"] = MagicMock()
sys.modules["src.schemas.profile"] = MagicMock()
sys.modules["src.schemas.summary"] = MagicMock()
sys.modules["src.schemas.weaver"] = MagicMock()
sys.modules["src.storage.base"] = MagicMock()
sys.modules["dotenv"] = MagicMock()
sys.modules["src.pipelines.retrieval"] = MagicMock()
sys.modules["src.schemas.retrieval"] = MagicMock()
sys.modules["typing_extensions"] = MagicMock()

# Mock config
settings_mock = MagicMock()
settings_mock.pinecone_api_key = "test"
settings_mock.pinecone_index_name = "test"
settings_mock.pinecone_dimension = 1536
settings_mock.pinecone_metric = "cosine"
settings_mock.pinecone_cloud = "aws"
settings_mock.pinecone_region = "us-east-1"
settings_mock.pinecone_namespace = "test"
settings_mock.gemini_api_key = "test"
settings_mock.embedding_model = "test"

sys.modules["src.config"] = MagicMock()
sys.modules["src.config"].settings = settings_mock

# Now import the class under test
from src.pipelines.ingest import IngestPipeline

class TestIngestParallel(unittest.TestCase):
    def setUp(self):
        self.pipeline = IngestPipeline()

        # Mock agents
        self.pipeline.temporal = MagicMock()
        self.pipeline.profiler = MagicMock()
        self.pipeline.judge = MagicMock()
        self.pipeline.weaver = MagicMock()

        # Mock downstream calls
        self.pipeline.judge.arun = AsyncMock(return_value=MagicMock())
        self.pipeline.weaver.execute = AsyncMock(return_value=MagicMock())

    def test_parallel_temporal_extraction(self):
        async def mock_arun(state):
            await asyncio.sleep(0.1)
            result = MagicMock()
            result.is_empty = False
            result.event = MagicMock()
            return result

        self.pipeline.temporal.arun = AsyncMock(side_effect=mock_arun)

        queries = [f"query_{i}" for i in range(10)]
        state = {
            "temporal_queries": queries,
            "user_id": "test_user",
            "session_datetime": "2023-10-27T10:00:00"
        }

        print("Running temporal extraction test...")
        start_time = time.time()
        asyncio.run(self.pipeline._node_extract_temporal(state))
        end_time = time.time()

        duration = end_time - start_time
        print(f"Temporal execution took {duration:.4f}s")

        # Current sequential implementation takes ~1.0s
        # Parallel implementation (sem=5) should take ~0.2s + overhead
        # If the test fails now, it confirms sequential behavior (regression test fails against baseline if strict, or we can use it to verify fix)
        # I'll set assertion to be < 0.6s to pass ONLY if optimized.
        self.assertLess(duration, 0.6, f"Execution took {duration}s, expected parallel execution")

    def test_parallel_profile_extraction(self):
        async def mock_arun(state):
            await asyncio.sleep(0.1)
            result = MagicMock()
            result.is_empty = False
            result.facts = [MagicMock()]
            return result

        self.pipeline.profiler.arun = AsyncMock(side_effect=mock_arun)

        queries = [f"query_{i}" for i in range(10)]
        state = {
            "profile_queries": queries,
            "user_id": "test_user",
        }

        print("Running profile extraction test...")
        start_time = time.time()
        asyncio.run(self.pipeline._node_extract_profile(state))
        end_time = time.time()

        duration = end_time - start_time
        print(f"Profile execution took {duration:.4f}s")
        self.assertLess(duration, 0.6, f"Execution took {duration}s, expected parallel execution")

if __name__ == "__main__":
    unittest.main()
