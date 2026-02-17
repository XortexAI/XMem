"""
Retrieval agent examples — teaches the LLM which tools to call for
different types of queries.

Each example is a tuple of:
    (query, available_profiles, expected_tool_calls)

where expected_tool_calls is a list of dicts with:
    tool:  "search_profile" | "search_temporal" | "search_summary"
    args:  the arguments for that tool call
    why:   brief explanation (for the prompt)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


RETRIEVAL_EXAMPLES: List[Tuple[str, str, List[Dict[str, Any]]]] = [

    # ── Profile lookups ───────────────────────────────────────────────

    (
        "What food does the user like?",
        "  - interest / foods  (current: \"pizza\")\n"
        "  - work / company  (current: \"Google\")",
        [
            {
                "tool": "search_profile",
                "args": {"topic": "interest", "sub_topic": "foods"},
                "why": "Question asks about food → exact match on interest/foods",
            },
        ],
    ),

    (
        "Where does the user work?",
        "  - interest / foods  (current: \"pizza\")\n"
        "  - work / company  (current: \"Google\")\n"
        "  - work / title  (current: \"Senior Engineer\")",
        [
            {
                "tool": "search_profile",
                "args": {"topic": "work", "sub_topic": "company"},
                "why": "Question asks about workplace → work/company",
            },
        ],
    ),

    (
        "What is the user's job title and company?",
        "  - work / company  (current: \"Google\")\n"
        "  - work / title  (current: \"Senior Engineer\")",
        [
            {
                "tool": "search_profile",
                "args": {"topic": "work", "sub_topic": "company"},
                "why": "Need company info",
            },
            {
                "tool": "search_profile",
                "args": {"topic": "work", "sub_topic": "title"},
                "why": "Need job title info",
            },
        ],
    ),

    # ── Temporal lookups ──────────────────────────────────────────────

    (
        "When is my dentist appointment?",
        "  - interest / foods  (current: \"pizza\")",
        [
            {
                "tool": "search_temporal",
                "args": {"query": "dentist appointment"},
                "why": "Date/scheduling question → search events",
            },
        ],
    ),

    (
        "When is my birthday?",
        "  - personal / name  (current: \"Alice\")",
        [
            {
                "tool": "search_temporal",
                "args": {"query": "birthday"},
                "why": "'When' question about a recurring event → temporal",
            },
        ],
    ),

    (
        "What events do I have coming up?",
        "  - work / company  (current: \"Google\")",
        [
            {
                "tool": "search_temporal",
                "args": {"query": "upcoming events"},
                "why": "Broad events question → temporal search",
            },
        ],
    ),

    # ── Summary / general lookups ─────────────────────────────────────

    (
        "What do you know about me?",
        "  - interest / foods  (current: \"pizza\")\n"
        "  - work / company  (current: \"Google\")",
        [
            {
                "tool": "search_summary",
                "args": {"query": "what do you know about the user"},
                "why": "Broad question with no specific domain → summary",
            },
        ],
    ),

    (
        "What happened in our last conversation?",
        "  - personal / name  (current: \"Alice\")",
        [
            {
                "tool": "search_summary",
                "args": {"query": "last conversation"},
                "why": "General recall question → summary search",
            },
        ],
    ),

    # ── Multi-tool queries ────────────────────────────────────────────

    (
        "Where do I work and when is my birthday?",
        "  - work / company  (current: \"Google\")\n"
        "  - work / title  (current: \"Senior Engineer\")",
        [
            {
                "tool": "search_profile",
                "args": {"topic": "work", "sub_topic": "company"},
                "why": "First part: workplace → profile lookup",
            },
            {
                "tool": "search_temporal",
                "args": {"query": "birthday"},
                "why": "Second part: birthday date → temporal search",
            },
        ],
    ),

    (
        "Tell me about my hobbies and any upcoming events",
        "  - interest / hobbies  (current: \"hiking\")\n"
        "  - personal / name  (current: \"Bob\")",
        [
            {
                "tool": "search_profile",
                "args": {"topic": "interest", "sub_topic": "hobbies"},
                "why": "Hobbies question → profile",
            },
            {
                "tool": "search_temporal",
                "args": {"query": "upcoming events"},
                "why": "Events question → temporal",
            },
        ],
    ),

    (
        "What do I like to eat and when did I start my current job?",
        "  - interest / foods  (current: \"sushi\")\n"
        "  - work / company  (current: \"Meta\")",
        [
            {
                "tool": "search_profile",
                "args": {"topic": "interest", "sub_topic": "foods"},
                "why": "Food preference → profile",
            },
            {
                "tool": "search_temporal",
                "args": {"query": "started current job"},
                "why": "When question → temporal search for job start event",
            },
        ],
    ),

    # ── No profile match → fallback to summary ───────────────────────

    (
        "Does the user have any pets?",
        "  - work / company  (current: \"Google\")\n"
        "  - interest / foods  (current: \"pizza\")",
        [
            {
                "tool": "search_summary",
                "args": {"query": "pets"},
                "why": "No pet-related profile exists → try summary",
            },
        ],
    ),

    (
        "What programming languages does the user know?",
        "  - work / company  (current: \"Google\")",
        [
            {
                "tool": "search_summary",
                "args": {"query": "programming languages"},
                "why": "No language-related profile → search summaries",
            },
        ],
    ),
]
