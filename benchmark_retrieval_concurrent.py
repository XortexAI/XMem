import asyncio
import time
import sys
from unittest.mock import AsyncMock, MagicMock

class MockPackage(MagicMock):
    __path__ = []

    def __getattr__(self, name):
        if name == "__spec__":
            return None
        return super().__getattr__(name)

class MockToolMessage:
    def __init__(self, content, tool_call_id):
        self.content = content
        self.tool_call_id = tool_call_id

mock_messages = MockPackage()
mock_messages.ToolMessage = MockToolMessage
sys.modules['langchain_core.messages'] = mock_messages

sys.modules['langgraph'] = MockPackage()
sys.modules['langgraph.graph'] = MockPackage()
sys.modules['langgraph.types'] = MockPackage()
sys.modules['langchain_core'] = MockPackage()
sys.modules['langchain_core.prompts'] = MockPackage()
sys.modules['langchain_core.language_models'] = MockPackage()
sys.modules['langchain_openai'] = MockPackage()
sys.modules['neo4j'] = MockPackage()
sys.modules['pinecone'] = MockPackage()
sys.modules['openai'] = MockPackage()
sys.modules['langchain_google_genai'] = MockPackage()
sys.modules['langchain_anthropic'] = MockPackage()
sys.modules['pydantic'] = MockPackage()
sys.modules['pydantic_settings'] = MockPackage()
sys.modules['fastapi'] = MockPackage()
sys.modules['langchain_community'] = MockPackage()
sys.modules['langchain_community.document_loaders'] = MockPackage()
sys.modules['numpy'] = MockPackage()
sys.modules['motor'] = MockPackage()
sys.modules['motor.motor_asyncio'] = MockPackage()
sys.modules['google'] = MockPackage()
sys.modules['google.genai'] = MockPackage()
sys.modules['google.genai.types'] = MockPackage()
sys.modules['dotenv'] = MockPackage()

import src.pipelines.retrieval
from src.schemas.retrieval import RetrievalResult, SourceRecord

class DummyPipelineConcurrent(src.pipelines.retrieval.RetrievalPipeline):
    def __init__(self):
        self.model = MagicMock()
        self.model_with_tools = MagicMock()
        self.neo4j = MagicMock()
        self.vector_store = MagicMock()

    async def _execute_tool(self, tool_name, tool_args, user_id, top_k):
        # concurrent execute
        await asyncio.sleep(0.5)
        return [SourceRecord(domain="dummy", content="dummy", score=1.0, metadata={})]

    async def run_concurrent(self, query: str, user_id: str, top_k: int = 20) -> RetrievalResult:
        ai_response = MagicMock()
        ai_response.tool_calls = [
            {"name": "searchprofile", "args": {"topic": "work"}, "id": "1"},
            {"name": "searchtemporal", "args": {"query": "recent events"}, "id": "2"},
            {"name": "searchsummary", "args": {"query": "summary"}, "id": "3"}
        ]

        sources = []
        tool_messages = []

        if ai_response.tool_calls:
            called_tools = set()

            async def _process_tool_call(tc):
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc["id"]

                records = await self._execute_tool(
                    tool_name, tool_args, user_id, top_k,
                )

                tool_result_text = self._format_tool_results(records)
                tool_msg = MockToolMessage(content=tool_result_text, tool_call_id=tool_id)

                return tool_name, records, tool_msg

            results = await asyncio.gather(*(_process_tool_call(tc) for tc in ai_response.tool_calls))

            for tool_name, records, tool_msg in results:
                sources.extend(records)
                tool_messages.append(tool_msg)
                called_tools.add(tool_name.lower().replace("_", ""))

        return None

async def main():
    pipeline = DummyPipelineConcurrent()

    print("Testing Concurrent Execution...")
    start_time = time.time()
    await pipeline.run_concurrent(query="hello", user_id="user1")
    duration = time.time() - start_time

    print(f"Concurrent Execution took {duration:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
