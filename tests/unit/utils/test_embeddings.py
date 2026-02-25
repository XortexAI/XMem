import unittest
from unittest.mock import MagicMock, patch
import sys
import importlib

class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        # Create mocks
        self.mock_google = MagicMock()
        self.mock_genai = MagicMock()
        self.mock_types = MagicMock()
        # Ensure imports work: import google.genai -> accessing mock_google.genai
        self.mock_google.genai = self.mock_genai
        self.mock_genai.types = self.mock_types

        # Configure genai.Client
        self.mock_client_cls = self.mock_genai.Client

        self.mock_settings = MagicMock()
        self.mock_settings.gemini_api_key = "fake_key"
        self.mock_settings.embedding_model = "fake_model"
        self.mock_settings.pinecone_dimension = 123

        # Patch sys.modules
        self.modules_patcher = patch.dict("sys.modules", {
            "google": self.mock_google,
            "google.genai": self.mock_genai,
            "google.genai.types": self.mock_types,
            "src.config": MagicMock(settings=self.mock_settings),
        })
        self.modules_patcher.start()

        # Import or reload src.utils.embeddings
        if "src.utils.embeddings" in sys.modules:
            import src.utils.embeddings
            importlib.reload(src.utils.embeddings)
        else:
            import src.utils.embeddings

        self.module = sys.modules["src.utils.embeddings"]
        # Ensure _embedding_client is None
        self.module._embedding_client = None

    def tearDown(self):
        self.modules_patcher.stop()

    def test_get_embedding_client_initialization(self):
        # When
        client = self.module.get_embedding_client()

        # Then
        self.mock_client_cls.assert_called_once_with(api_key="fake_key")
        self.assertEqual(client, self.mock_client_cls.return_value)

        # Call again
        client2 = self.module.get_embedding_client()
        self.mock_client_cls.assert_called_once() # Should be called only once
        self.assertEqual(client, client2)

    def test_embed_text(self):
        # Setup
        mock_client_instance = self.mock_client_cls.return_value

        mock_result = MagicMock()
        mock_embedding_obj = MagicMock()
        mock_embedding_obj.values = [0.1, 0.2, 0.3]
        mock_result.embeddings = [mock_embedding_obj]

        mock_client_instance.models.embed_content.return_value = mock_result

        # When
        result = self.module.embed_text("hello world")

        # Then
        self.assertEqual(result, [0.1, 0.2, 0.3])
        mock_client_instance.models.embed_content.assert_called_once()

        _, kwargs = mock_client_instance.models.embed_content.call_args
        self.assertEqual(kwargs['model'], "fake_model")
        self.assertEqual(kwargs['contents'], "hello world")
        # Ensure config is passed
        self.assertIn('config', kwargs)

if __name__ == "__main__":
    unittest.main()
