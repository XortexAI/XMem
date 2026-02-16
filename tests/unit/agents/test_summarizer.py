"""
Interactive test for the SummarizerAgent.

Usage:
    PYTHONPATH=. python3 tests/unit/agents/test_summarizer.py
    PYTHONPATH=. python3 tests/unit/agents/test_summarizer.py --provider gemini
"""

import asyncio
import sys

from src.models import get_model
from src.agents.summarizer import SummarizerAgent


async def main():
    provider = None
    if "--provider" in sys.argv:
        idx = sys.argv.index("--provider")
        provider = sys.argv[idx + 1]

    model = get_model(provider=provider)
    agent = SummarizerAgent(model=model)

    model_name = getattr(model, "model", getattr(model, "model_name", "unknown"))
    print(f"\n  Summarizer Agent ready  (provider: {type(model).__name__}, model: {model_name})")
    print("  Enter a user query, then an agent response. Type 'q' to quit.\n")

    while True:
        try:
            print("  [User Query]")
            user_query = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_query.lower() in ("q", "quit", "exit"):
            break
        if not user_query:
            continue

        try:
            print("  [Agent Response]")
            agent_response = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if agent_response.lower() in ("q", "quit", "exit"):
            break

        result = await agent.arun({
            "user_query": user_query,
            "agent_response": agent_response,
        })

        if result.is_empty:
            print("   (no memorable facts — trivial/skip)\n")
        else:
            print(f"\n{result.summary}\n")


if __name__ == "__main__":
    asyncio.run(main())
