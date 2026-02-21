import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import types
import logging

@pytest.fixture(scope="module")
def mock_pipeline():
    # Setup mocks
    mock_neo4j = MagicMock()
    mock_pinecone = MagicMock()
    mock_models = MagicMock()

    mock_config = types.ModuleType('src.config')
    mock_settings = MagicMock()
    # Mock settings required by IngestPipeline.__init__
    mock_settings.pinecone_api_key = "dummy"
    mock_settings.pinecone_index_name = "dummy"
    mock_settings.pinecone_dimension = 384
    mock_settings.pinecone_metric = "cosine"
    mock_settings.pinecone_cloud = "aws"
    mock_settings.pinecone_region = "us-east-1"
    mock_settings.pinecone_namespace = "default"
    mock_settings.neo4j_uri = "bolt://localhost:7687"
    mock_settings.neo4j_username = "neo4j"
    mock_settings.neo4j_password = "dummy"
    mock_settings.embedding_model = "all-MiniLM-L6-v2"
    mock_settings.classifier_model = None
    mock_settings.profiler_model = None
    mock_settings.temporal_model = None
    mock_settings.summarizer_model = None
    mock_settings.judge_model = None

    mock_config.settings = mock_settings
    mock_config.get_logger = MagicMock(return_value=logging.getLogger('mock_logger'))

    mock_constants = MagicMock()
    mock_constants.LLM_TAB_SEPARATOR = ' | '

    # We use patch.dict to safely mock sys.modules for this test function scope
    with patch.dict(sys.modules, {
        'src.graph.neo4j_client': mock_neo4j,
        'src.storage.pinecone': mock_pinecone,
        'src.models': mock_models,
        'src.config': mock_config,
        'src.config.constants': mock_constants,
    }):
        # Mock get_model before import
        mock_models.get_model.return_value = MagicMock()
        mock_models.get_vision_model.return_value = MagicMock()

        # Import the module inside the patch context
        # We need to ensure it's reloaded or not cached if it was imported before
        if 'src.pipelines.ingest' in sys.modules:
            del sys.modules['src.pipelines.ingest']

        from src.pipelines.ingest import IngestPipeline, IngestState

        # Yield IngestState class and a factory for pipeline
        # We can't yield a single pipeline instance easily if we want to reset mocks between tests,
        # but since we are mocking methods on the instance, we can just yield the class or a factory.
        # Actually, creating the pipeline instance here is fine, we just need to reset mocks in tests.

        pipeline = IngestPipeline(
            vector_store=MagicMock(),
            neo4j_client=MagicMock()
        )
        # Mock agents
        pipeline.profiler = MagicMock()
        pipeline.temporal = MagicMock()
        pipeline.judge = MagicMock()
        pipeline.weaver = MagicMock()

        # Mock async methods
        pipeline.profiler.arun = AsyncMock()
        pipeline.temporal.arun = AsyncMock()
        pipeline.judge.arun = AsyncMock()
        pipeline.weaver.execute = AsyncMock()

        yield pipeline, IngestState

@pytest.mark.asyncio
async def test_node_extract_profile_parallel(mock_pipeline):
    pipeline, IngestState = mock_pipeline

    # Reset mocks
    pipeline.profiler.arun.reset_mock(side_effect=True, return_value=True)
    pipeline.judge.arun.reset_mock(side_effect=True, return_value=True)
    pipeline.weaver.execute.reset_mock(side_effect=True, return_value=True)

    # Setup
    queries = ["q1", "q2"]
    state = IngestState(profile_queries=queries, user_id="test_user")

    # Mock profiler results
    fact1 = MagicMock()
    fact1.model_dump.return_value = {"topic": "t1"}
    res1 = MagicMock()
    res1.is_empty = False
    res1.facts = [fact1]

    fact2 = MagicMock()
    fact2.model_dump.return_value = {"topic": "t2"}
    res2 = MagicMock()
    res2.is_empty = False
    res2.facts = [fact2]

    pipeline.profiler.arun.side_effect = [res1, res2]

    pipeline.judge.arun.return_value = MagicMock()
    pipeline.weaver.execute.return_value = MagicMock()

    # Execute
    await pipeline._node_extract_profile(state)

    # Verify
    assert pipeline.profiler.arun.call_count == 2
    # Ensure judge was called with combined items
    call_args = pipeline.judge.arun.call_args[0][0]
    assert len(call_args["new_items"]) == 2
    assert call_args["new_items"][0] == {"topic": "t1"}
    assert call_args["new_items"][1] == {"topic": "t2"}

@pytest.mark.asyncio
async def test_node_extract_temporal_parallel(mock_pipeline):
    pipeline, IngestState = mock_pipeline

    # Reset mocks
    pipeline.temporal.arun.reset_mock(side_effect=True, return_value=True)
    pipeline.judge.arun.reset_mock(side_effect=True, return_value=True)
    pipeline.weaver.execute.reset_mock(side_effect=True, return_value=True)

    # Setup
    queries = ["q1", "q2"]
    state = IngestState(temporal_queries=queries, user_id="test_user", session_datetime="now")

    # Mock temporal results
    event1 = MagicMock()
    event1.date = "2023-01-01"
    event1.event_name = "e1"
    res1 = MagicMock()
    res1.is_empty = False
    res1.event = event1

    event2 = MagicMock()
    event2.date = "2023-01-02"
    event2.event_name = "e2"
    res2 = MagicMock()
    res2.is_empty = False
    res2.event = event2

    pipeline.temporal.arun.side_effect = [res1, res2]

    pipeline.judge.arun.return_value = MagicMock()
    pipeline.weaver.execute.return_value = MagicMock()

    # Execute
    await pipeline._node_extract_temporal(state)

    # Verify
    assert pipeline.temporal.arun.call_count == 2
    # Ensure judge was called with combined items
    call_args = pipeline.judge.arun.call_args[0][0]
    assert len(call_args["new_items"]) == 2
    assert call_args["new_items"][0]["event_name"] == "e1"
    assert call_args["new_items"][1]["event_name"] == "e2"
