"""
Interactive classifier test — send queries and see live classifications.

Usage:
    PYTHONPATH=. python tests/unit/agents/test_classifier.py
    PYTHONPATH=. python tests/unit/agents/test_classifier.py --provider gemini
"""

import asyncio
import sys

from src.models import get_model
from src.agents.classifier import ClassifierAgent


async def main():
    provider = None
    if "--provider" in sys.argv:
        idx = sys.argv.index("--provider")
        provider = sys.argv[idx + 1]

    model = get_model(provider=provider)
    agent = ClassifierAgent(model=model)

    print(f"\n  Classifier Agent ready  (model: {model.__class__.__name__})")
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

        result = await agent.arun({"user_query": query})

        if not result.classifications:
            print("   (no classifications — trivial/skip)\n")
        else:
            for i, c in enumerate(result.classifications, 1):
                print(f"   {i}. [{c['source']}]  {c['query']}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
