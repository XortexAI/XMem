import asyncio
import time

class MockJudgeResult:
    def __init__(self, n_ops=10):
        self.is_empty = False
        self.has_writes = True
        self.operations = [{"type": "ADD", "content": f"test {i}"} for i in range(n_ops)]

class MockWeaver:
    async def _execute_one(self, op, domain, user_id):
        await asyncio.sleep(0.1) # Simulate DB op
        return f"Executed {op['content']}"

async def main_sequential():
    weaver = MockWeaver()
    judge_result = MockJudgeResult()
    start = time.time()
    result = []
    for op in judge_result.operations:
        executed = await weaver._execute_one(op, "domain", "user_id")
        result.append(executed)
    print(f"Sequential took: {time.time() - start:.2f}s")

async def main_concurrent():
    weaver = MockWeaver()
    judge_result = MockJudgeResult()
    start = time.time()

    # Run concurrently using asyncio.gather
    tasks = [weaver._execute_one(op, "domain", "user_id") for op in judge_result.operations]
    result = await asyncio.gather(*tasks)

    print(f"Concurrent took: {time.time() - start:.2f}s")

if __name__ == "__main__":
    asyncio.run(main_sequential())
    asyncio.run(main_concurrent())
