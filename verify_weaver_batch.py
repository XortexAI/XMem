import asyncio
import time
from unittest.mock import MagicMock
from src.pipelines.weaver import Weaver, _extract_structured_metadata
from src.schemas.judge import JudgeResult, JudgeDomain, Operation, OperationType

class DummyVectorStore:
    def add(self, texts, embeddings, metadata):
        time.sleep(0.5) # Simulate network IO
        return [f"id_{i}" for i in range(len(texts))]

    def delete(self, ids):
        time.sleep(0.5) # Simulate network IO
        return True

def slow_embed(text: str):
    time.sleep(0.2) # Simulate API call delay
    if "FAIL" in text:
        raise Exception("Simulated embedding failure")
    return [0.1, 0.2, 0.3]


async def main():
    store = DummyVectorStore()
    weaver = Weaver(vector_store=store, embed_fn=slow_embed)

    # 5 ADD operations (1 fails) and 2 DELETE operations
    operations = [
        Operation(type=OperationType.ADD, content="Valid doc 1"),
        Operation(type=OperationType.ADD, content="Valid doc 2"),
        Operation(type=OperationType.ADD, content="FAIL doc"),
        Operation(type=OperationType.ADD, content="Valid doc 3"),
        Operation(type=OperationType.ADD, content="Valid doc 4"),
        Operation(type=OperationType.DELETE, embedding_id="id_to_delete_1"),
        Operation(type=OperationType.DELETE, embedding_id="id_to_delete_2"),
    ]

    judge_result = JudgeResult(
        confidence=0.9,
        operations=operations
    )

    start_time = time.time()
    res = await weaver.execute(judge_result, JudgeDomain.PROFILE, "user_123")
    end_time = time.time()

    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print(f"Total operations expected: 7")
    print(f"Operations executed: {res.total}")
    print(f"Succeeded: {res.succeeded}")
    print(f"Failed: {res.failed}")

    # We expect 5 adds + 2 deletes
    # 1 fail doc should fail.
    # With concurrency:
    # 5 embeds run in parallel (max ~0.2s)
    # 1 store add runs (~0.5s)
    # 1 store delete runs (~0.5s)
    # Total time should be roughly ~1.2s instead of > 2s (sequential: 5*0.2 + 0.5 + 0.5 = 2.0)

if __name__ == "__main__":
    asyncio.run(main())
