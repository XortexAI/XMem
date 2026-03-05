import asyncio
import logging
logging.basicConfig(level=logging.INFO)
from src.pipelines.code_retrieval import CodeRetrievalPipeline

async def main():
    pipeline = CodeRetrievalPipeline(org_id="Xmem", repos=["xmem-main"])
    print("\n--- Test 1: Finding a symbol ---")
    res1 = await pipeline.run("give full code of src/agents/judge.py")
    print(res1.answer)

if __name__ == "__main__":
    asyncio.run(main())
