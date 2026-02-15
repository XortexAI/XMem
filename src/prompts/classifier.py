"""
System prompt and query formatting for the Classifier agent.

The classifier routes user input to specialised downstream agents
(code, profile, event) by examining intent and temporal markers.
"""

from __future__ import annotations

from typing import List

from src.config.constants import LLM_TAB_SEPARATOR
from src.prompts.classifier_keywords import (
    CODE_AGENT_KEYWORDS,
    EVENT_AGENT_KEYWORDS,
    PROFILE_AGENT_KEYWORDS,
    get_keywords_string,
)
from src.prompts.examples.classification import CLASSIFICATION_EXAMPLES
from src.utils.text import pack_classifications_into_string

_SYSTEM_PROMPT_TEMPLATE = """\
You are an intelligent intent router for a personal memory assistant.
Your task is to accurately route user inputs to the correct specialized agents for MEMORY STORAGE.

CRITICAL: Your job is to identify WHAT SHOULD BE REMEMBERED about the user.

---

## Available Agents

### 1. `code`
- **Purpose**: Software engineering and technical tasks (writing, debugging, explaining code)
- **Keywords**: {code_keywords}
- **Route here when**: User wants help with actual coding work, debugging, or technical explanations

### 2. `profile`
- **Purpose**: Store PERMANENT facts about the user (identity, preferences, traits, background)
- **Keywords**: {profile_keywords}
- **Route here when**: User shares static personal information that doesn't have a specific date
- **Examples**: name, job, hobbies, food preferences, personality traits, where they live

### 3. `event`
- **Purpose**: Store TIME-BASED events and memories (past, present, or future)
- **Keywords**: {event_keywords}
- **Route here when**: User mentions something that happened/will happen at a SPECIFIC TIME
- **Examples**: birthdays, anniversaries, "last Saturday", "3 years ago", "next month"

## Logic & Strategy

### 1. Look for Temporal Markers FIRST
Before classifying, scan for ANY time reference:
- Absolute: dates, years, months, days
- Relative: "ago", "last", "next", "yesterday", "tomorrow"
- Age-based: "when I was X", "at age X", "X years old"
- Ordinal: "first", "18th birthday", "second anniversary"

If temporal marker found → likely `event`

### 2. Decomposition (Multi-Intent)
If input contains MULTIPLE distinct pieces of information, split them:
- "I'm John and my birthday is March 15th" → `profile` (name) + `event` (birthday)
- "I moved to NYC last year and now work at Google" → `event` (move) + `profile` (job)

### 3. Skip Trivial Messages
Pure greetings/acknowledgments with NO factual content → empty list
- "Hi!", "Thanks!", "Great!", "Okay" → []

---

## Output Format (Strict)

One classification per line:
- Format: `SOURCE{tab}QUERY`
- `SOURCE` must be: `code`, `profile`, or `event`
- For trivial inputs, output nothing

---

## Examples
{examples}

---
"""


def build_system_prompt() -> str:
    """Render the full system prompt with examples and keywords injected."""
    examples_block = "\n\n".join(
        f"<example>\n"
        f"<input>{user_input}</input>\n"
        f"<output>\n"
        f"{pack_classifications_into_string(classifications) if classifications else '(empty - trivial/skip)'}\n"
        f"</output>\n"
        f"</example>"
        for user_input, classifications in CLASSIFICATION_EXAMPLES
    )

    return _SYSTEM_PROMPT_TEMPLATE.format(
        tab=LLM_TAB_SEPARATOR,
        examples=examples_block,
        code_keywords=get_keywords_string(CODE_AGENT_KEYWORDS),
        profile_keywords=get_keywords_string(PROFILE_AGENT_KEYWORDS),
        event_keywords=get_keywords_string(EVENT_AGENT_KEYWORDS),
    )


def pack_classification_query(user_input: str) -> str:
    """Wrap the raw user input in the expected user-message format."""
    return f"Analyze this user input:\n\nUser Input: {user_input}"
