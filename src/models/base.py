"""
Base types for the models module.
"""

from typing import Literal

Provider = Literal[
    "gemini",
    "claude",
    "openai",
    "deepseek",
    "groq",
    "mimo",
    "openrouter",
    "bedrock",
    "ollama",
]
