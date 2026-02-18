import sys
import asyncio
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

# ----------------------------------------------------------------------------
# 1. Setup Mocks BEFORE importing the module under test
# ----------------------------------------------------------------------------

# Mock src.config and settings
mock_settings = MagicMock()
mock_settings.pinecone_api_key = "test-key"
mock_settings.pinecone_index_name = "test-index"
mock_settings.pinecone_dimension = 1536
mock_settings.pinecone_metric = "cosine"
mock_settings.pinecone_cloud = "aws"
mock_settings.pinecone_region = "us-east-1"
mock_settings.pinecone_namespace = "test-namespace"

mock_config = MagicMock()
mock_config.settings = mock_settings
mock_config.get_logger = MagicMock(return_value=MagicMock())

# Mock pinecone module
mock_pinecone_module = MagicMock()
mock_pinecone_client = MagicMock()
mock_index = MagicMock()
mock_pinecone_client.Index.return_value = mock_index
mock_pinecone_module.Pinecone.return_value = mock_pinecone_client
mock_pinecone_module.ServerlessSpec = MagicMock()

# Patch sys.modules
with patch.dict("sys.modules", {
    "src.config": mock_config,
    "pinecone": mock_pinecone_module,
}):
    # Import the module under test
    import src.storage.pinecone as pinecone_store_module
    from src.storage.pinecone import PineconeVectorStore, SearchResult

# ----------------------------------------------------------------------------
# 2. Test Class
# ----------------------------------------------------------------------------

class TestPineconeVectorStore(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # We need to ensure PINECONE_AVAILABLE is True.
        # Since we mocked pinecone module during import, it should be True.
        # The imports in the module under test are already pointing to our mocks.

        self.store = PineconeVectorStore(api_key="test-key", create_if_not_exists=False)

    async def test_search_by_text_calls_embed_text(self):
        # Mock embed_text from src.pipelines.ingest

        mock_embedding = [0.1] * 1536
        mock_embed_text = MagicMock(return_value=mock_embedding)

        # Mock store.search
        self.store.search = MagicMock(return_value=[
            SearchResult(id="1", content="test", score=0.9, metadata={})
        ])

        # Patch the module where embed_text is imported FROM
        mock_ingest = MagicMock()
        mock_ingest.embed_text = mock_embed_text

        with patch.dict("sys.modules", {"src.pipelines.ingest": mock_ingest}):
            results = await self.store.search_by_text("query", top_k=5)

            # Verify embed_text was called
            mock_embed_text.assert_called_once_with("query")

            # Verify search was called with the embedding
            self.store.search.assert_called_once_with(
                query_embedding=mock_embedding,
                top_k=5,
                filters=None
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].id, "1")

    async def test_search_by_text_offloads_to_thread(self):
        """
        Verify that embed_text is offloaded to a thread using asyncio.to_thread.
        """
        mock_embedding = [0.1] * 1536
        mock_embed_text = MagicMock(return_value=mock_embedding)

        self.store.search = MagicMock(return_value=[])

        mock_ingest = MagicMock()
        mock_ingest.embed_text = mock_embed_text

        with patch.dict("sys.modules", {"src.pipelines.ingest": mock_ingest}):
            with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
                # Mock return value of to_thread to be the embedding
                mock_to_thread.return_value = mock_embedding

                await self.store.search_by_text("query")

                # Check if to_thread was called with embed_text and the query
                mock_to_thread.assert_called_once_with(mock_embed_text, "query")
