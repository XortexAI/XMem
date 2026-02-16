"""
Interactive profiler test — send queries and see extracted profile facts.

Usage:
    PYTHONPATH=. python tests/unit/agents/test_profiler.py
    PYTHONPATH=. python tests/unit/agents/test_profiler.py --provider gemini
"""

import asyncio
import sys

from src.models import get_model
from src.agents.profiler import ProfilerAgent


async def main():
    provider = None
    if "--provider" in sys.argv:
        idx = sys.argv.index("--provider")
        provider = sys.argv[idx + 1]

    model = get_model(provider=provider)
    agent = ProfilerAgent(model=model)

    model_name = getattr(model, "model", getattr(model, "model_name", "unknown"))
    print(f"\n  Profiler Agent ready  (provider: {type(model).__name__}, model: {model_name})")
    print(f"  Type a query and press Enter. Type 'q' to quit.\n")

    while True:
        try:
            query = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.lower() in ("q", "quit", "exit"):
            break
        if not query:
            continue

        # Simulate the input coming from the classifier
        result = await agent.arun({"classifier_output": query})

        if result.is_empty:
            print("   (no profile facts extracted)\n")
        else:
            for idx, fact in enumerate(result.facts, 1):
                print(f"   {idx}. [{fact.topic}/{fact.sub_topic}]  {fact.memo}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
