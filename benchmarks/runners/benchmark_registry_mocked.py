import sys
import timeit
import time
from unittest.mock import MagicMock, patch
import types
import os

# 1. Mock external dependencies
langchain_core = types.ModuleType("langchain_core")
sys.modules["langchain_core"] = langchain_core
langchain_core_lm = types.ModuleType("langchain_core.language_models")
sys.modules["langchain_core.language_models"] = langchain_core_lm

class BaseChatModel:
    pass
langchain_core_lm.BaseChatModel = BaseChatModel

# Mock langchain_openai
langchain_openai = types.ModuleType("langchain_openai")
sys.modules["langchain_openai"] = langchain_openai
class ChatOpenAI(BaseChatModel):
    def __init__(self, **kwargs):
        # Simulate initialization overhead (e.g. config parsing, client setup)
        # 0.1ms overhead
        start = time.perf_counter()
        while time.perf_counter() - start < 0.0001:
            pass
langchain_openai.ChatOpenAI = ChatOpenAI

# Mock langchain_google_genai
langchain_google_genai = types.ModuleType("langchain_google_genai")
sys.modules["langchain_google_genai"] = langchain_google_genai
class ChatGoogleGenerativeAI(BaseChatModel):
    def __init__(self, **kwargs):
         # Simulate initialization overhead
        start = time.perf_counter()
        while time.perf_counter() - start < 0.0001:
            pass
langchain_google_genai.ChatGoogleGenerativeAI = ChatGoogleGenerativeAI

# 2. Mock src.config.settings
class Settings:
    gemini_api_key = "fake_key"
    gemini_model = "gemini-pro"
    claude_api_key = "fake_key"
    claude_model = "claude-3-opus"
    openai_api_key = "fake_key"
    openai_model = "gpt-4"
    temperature = 0.5
    fallback_order = ["openai", "gemini"]

src_config = types.ModuleType("src.config")
sys.modules["src.config"] = src_config
src_config.settings = Settings()

# 3. Import src.models.registry
src_models_base = types.ModuleType("src.models.base")
sys.modules["src.models.base"] = src_models_base
src_models_base.Provider = str

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.registry import get_model

def benchmark():
    # Warmup
    print("Warming up...")
    get_model("openai")

    start = timeit.default_timer()
    N = 1000
    for _ in range(N):
        get_model("openai")
    end = timeit.default_timer()

    avg_time_ms = (end - start) / N * 1000
    print(f"Time for {N} calls: {end - start:.4f} s")
    print(f"Average time per call: {avg_time_ms:.4f} ms")

if __name__ == "__main__":
    benchmark()
