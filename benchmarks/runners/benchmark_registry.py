import os
import timeit
import sys
from unittest.mock import patch

# Mock environment variables to satisfy Pydantic validation
os.environ["GEMINI_API_KEY"] = "fake_gemini_key"
os.environ["CLAUDE_API_KEY"] = "fake_claude_key"
os.environ["OPENAI_API_KEY"] = "fake_openai_key"
os.environ["PINECONE_API_KEY"] = "fake_pinecone_key"
os.environ["NEO4J_PASSWORD"] = "fake_neo4j_password"

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.registry import get_model

def benchmark_get_model():
    # Warmup
    get_model("openai")

    start_time = timeit.default_timer()
    iterations = 10000
    for _ in range(iterations):
        get_model("openai")
    end_time = timeit.default_timer()

    total_time = end_time - start_time
    print(f"Total time for {iterations} calls: {total_time:.4f} seconds")
    print(f"Average time per call: {total_time/iterations*1000:.4f} ms")

if __name__ == "__main__":
    benchmark_get_model()
