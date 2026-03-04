"""
Interactive temporal test — send queries and see extracted events.

Usage:
    PYTHONPATH=. python3 tests/unit/agents/test_temporal.py
    PYTHONPATH=. python3 tests/unit/agents/test_temporal.py --provider gemini
"""

import asyncio
import sys

from src.models import get_model
from src.agents.temporal import TemporalAgent


async def main():
    provider = None
    if "--provider" in sys.argv:
        idx = sys.argv.index("--provider")
        provider = sys.argv[idx + 1]

    model = get_model(provider=provider)
    agent = TemporalAgent(model=model)

    model_name = getattr(model, "model", getattr(model, "model_name", "unknown"))
    print(f"\n  Temporal Agent ready  (provider: {type(model).__name__}, model: {model_name})")
    print(f"  Type a query and press Enter. Type 'q' to quit.")
    print(f"  Optionally set context date with:  date: 4:04 pm on 20 January, 2023\n")

    context_date = ""

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

        # Allow setting context date inline
        if query.lower().startswith("date:"):
            context_date = query[5:].strip()
            print(f"   Context date set to: {context_date}\n")
            continue

        state = {"classifier_output": query}
        if context_date:
            state["session_datetime"] = context_date

        result = await agent.arun(state)

        if result.is_empty:
            print("   (no event extracted)\n")
        else:
            for i, e in enumerate(result.events, 1):
                if len(result.events) > 1:
                    print(f"   --- Event {i} ---")
                print(f"   Date:       {e.date}")
                print(f"   Event:      {e.event_name or 'N/A'}")
                print(f"   Year:       {e.year or 'N/A'}")
                print(f"   Desc:       {e.desc or 'N/A'}")
                print(f"   Time:       {e.time or 'N/A'}")
                print(f"   Expression: {e.date_expression or 'N/A'}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
