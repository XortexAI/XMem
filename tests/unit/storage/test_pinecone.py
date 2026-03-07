
import sys
from unittest.mock import MagicMock, patch
import pytest

# Mock pinecone module
sys.modules["pinecone"] = MagicMock()

# Mock pydantic
sys.modules["pydantic"] = MagicMock()
sys.modules["pydantic_settings"] = MagicMock()

# Mock src.config and src.config.settings
mock_settings = MagicMock()
mock_settings.pinecone_api_key = "dummy"
mock_settings.pinecone_index_name = "test-index"
mock_settings.pinecone_dimension = 10
mock_settings.pinecone_metric = "cosine"
mock_settings.pinecone_cloud = "aws"
mock_settings.pinecone_region = "us-east-1"
mock_settings.pinecone_namespace = "default"

mock_config = MagicMock()
mock_config.settings = mock_settings
mock_config.get_logger = lambda name: MagicMock()

sys.modules["src.config"] = mock_config
sys.modules["src.config.settings"] = mock_settings

# Mock src.utils.retry
sys.modules["src.utils.retry"] = MagicMock()
sys.modules["src.utils.retry"].with_retry = lambda config: lambda f: f

# Now import PineconeVectorStore
# We need to make sure src.storage.base can be imported if it doesn't depend on pydantic
# src/storage/base.py imports dataclasses (std lib), abc (std lib), typing (std lib), enum (std lib)
# and ..config (mocked), ..utils.exceptions

from src.storage.pinecone import PineconeVectorStore

class TestPineconeVectorStore:
    def test_add_metadata_handling(self):
        # Arrange
        store = PineconeVectorStore(api_key="test-key", create_if_not_exists=False)
        store._index = MagicMock()

        texts = ["text1", "text2"]
        embeddings = [[0.1]*10, [0.2]*10]

        # Act
        store.add(texts=texts, embeddings=embeddings)

        # Assert
        store._index.upsert.assert_called()
        call_args = store._index.upsert.call_args
        # Depending on how upsert is called (kwargs vs args)
        if 'vectors' in call_args.kwargs:
            vectors = call_args.kwargs['vectors']
        else:
            vectors = call_args.args[0]

        assert len(vectors) == 2
        assert vectors[0]['metadata']['content'] == "text1"
        assert vectors[1]['metadata']['content'] == "text2"

        # Ensure modifying one metadata dict doesn't affect others
        # (This is the crucial safety check for shared empty dict optimization)
        vectors[0]['metadata']['extra'] = 'val'
        assert 'extra' not in vectors[1]['metadata']

    def test_add_explicit_metadata(self):
        # Arrange
        store = PineconeVectorStore(api_key="test-key", create_if_not_exists=False)
        store._index = MagicMock()

        texts = ["text1"]
        embeddings = [[0.1]*10]
        metadata = [{"source": "test"}]

        # Act
        store.add(texts=texts, embeddings=embeddings, metadata=metadata)

        # Assert
        call_args = store._index.upsert.call_args
        if 'vectors' in call_args.kwargs:
            vectors = call_args.kwargs['vectors']
        else:
            vectors = call_args.args[0]

        assert vectors[0]['metadata']['source'] == "test"
        assert vectors[0]['metadata']['content'] == "text1"
