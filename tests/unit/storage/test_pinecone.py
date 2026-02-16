
import sys
import unittest
from unittest.mock import MagicMock, patch
import importlib

class TestPineconeVectorStore(unittest.TestCase):
    def setUp(self):
        # Create mocks for dependencies
        self.pydantic_mock = MagicMock()
        self.pydantic_settings_mock = MagicMock()
        self.pinecone_mock = MagicMock()

        # Create config mock
        self.config_mock = MagicMock()
        self.settings_mock = MagicMock()
        self.settings_mock.pinecone_api_key = "test-key"
        self.settings_mock.pinecone_index_name = "test-index"
        self.settings_mock.pinecone_namespace = "test-ns"
        self.settings_mock.pinecone_dimension = 128
        self.settings_mock.pinecone_metric = "cosine"
        self.settings_mock.pinecone_cloud = "aws"
        self.settings_mock.pinecone_region = "us-east-1"
        self.config_mock.settings = self.settings_mock
        self.config_mock.get_logger = MagicMock()

        # Setup the patcher for sys.modules
        self.modules_patcher = patch.dict(sys.modules, {
            "pydantic": self.pydantic_mock,
            "pydantic_settings": self.pydantic_settings_mock,
            "pinecone": self.pinecone_mock,
            "src.config": self.config_mock,
            "src.config.settings": MagicMock(),
        })
        self.modules_patcher.start()

        # Reload or import the module under test
        # We enforce reload to ensure it uses the mocked dependencies
        if "src.storage.pinecone" in sys.modules:
            import src.storage.pinecone
            importlib.reload(src.storage.pinecone)
        else:
            import src.storage.pinecone

        self.module = src.storage.pinecone
        self.PineconeVectorStore = self.module.PineconeVectorStore

        # Instantiate store
        self.store = self.PineconeVectorStore(
            api_key="test-key",
            index_name="test-index",
            create_if_not_exists=False
        )

    def tearDown(self):
        self.modules_patcher.stop()
        # Clean up the module from sys.modules to avoid polluting other tests
        # with a module that has references to stopped mocks
        if "src.storage.pinecone" in sys.modules:
            del sys.modules["src.storage.pinecone"]

    def test_build_filter_empty(self):
        """Test _build_filter with empty input."""
        result = self.store._build_filter({})
        self.assertIsNone(result)

        result = self.store._build_filter(None)
        self.assertIsNone(result)

    def test_build_filter_single(self):
        """Test _build_filter with single key-value pair."""
        filters = {"category": "news"}
        expected = {"category": {"$eq": "news"}}

        result = self.store._build_filter(filters)
        self.assertEqual(result, expected)

    def test_build_filter_multiple(self):
        """Test _build_filter with multiple key-value pairs."""
        filters = {"category": "news", "year": 2023}

        result = self.store._build_filter(filters)

        self.assertIn("$and", result)
        self.assertEqual(len(result["$and"]), 2)

        conditions = result["$and"]
        self.assertTrue(
            {"category": {"$eq": "news"}} in conditions
        )
        self.assertTrue(
            {"year": {"$eq": 2023}} in conditions
        )

if __name__ == "__main__":
    unittest.main()
