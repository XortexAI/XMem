"""
Interactive test for the ImageAgent.

Unlike other agents, the ImageAgent requires a **vision-capable** model.
This test uses ``get_vision_model()`` instead of ``get_model()``.

Usage:
    PYTHONPATH=. python tests/unit/agents/test_image.py
    PYTHONPATH=. python tests/unit/agents/test_image.py --provider gemini

Provide an image URL when prompted (or paste a base64 data-URI).
"""

import asyncio
import sys

from src.models import get_vision_model
from src.agents.image import ImageAgent


async def main():
    provider = None
    if "--provider" in sys.argv:
        idx = sys.argv.index("--provider")
        provider = sys.argv[idx + 1]

    # Use vision model — NOT the regular get_model()
    model = get_vision_model(provider=provider)
    agent = ImageAgent(model=model)

    model_name = getattr(model, "model", getattr(model, "model_name", "unknown"))
    print(f"\n  Image Agent ready  (provider: {type(model).__name__}, model: {model_name})")
    print("  This agent requires a vision-capable model for image analysis.")
    print("  Enter an image URL, then an optional text query. Type 'q' to quit.\n")

    while True:
        # --- Image URL ---
        try:
            print("  [Image URL or base64 data-URI]")
            image_url = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if image_url.lower() in ("q", "quit", "exit"):
            break

        # --- Optional text query ---
        try:
            print("  [Text query (optional, press Enter to skip)]")
            query = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.lower() in ("q", "quit", "exit"):
            break

        # Build state matching pipeline expectations
        state = {
            "classifier_output": query,
            "image_url": image_url,
        }

        result = await agent.arun(state)

        if result.is_empty:
            print("   (no observations extracted)\n")
        else:
            print(f"\n  Description: {result.description}")
            for idx, obs in enumerate(result.observations, 1):
                conf = obs.confidence or "N/A"
                print(f"   {idx}. [{obs.category}] {obs.description}  (confidence: {conf})")
            print()


if __name__ == "__main__":
    asyncio.run(main())
