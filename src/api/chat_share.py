from __future__ import annotations

from typing import Any


def scrape_failure_message(result: dict[str, Any]) -> str:
    provider = result.get("provider") or "unknown"

    if provider in {"chatgpt", "claude", "gemini"}:
        display_name = {
            "chatgpt": "ChatGPT",
            "claude": "Claude",
            "gemini": "Gemini",
        }[provider]
        return (
            f"Could not extract messages from this {display_name} share link. "
            "Make sure the link is public, still exists, and has not expired."
        )

    return (
        "Failed to extract messages from the provided link. "
        "Supported public share links are ChatGPT, Claude, and Gemini."
    )
