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
        "  - interest / foods\n"
        "  - work / company",
        [
            {
                "tool": "search_profile",
                "args": {"topic": "interest"},
                "why": "Question asks about food → falls under 'interest' topic",
            },
        ],
    ),

    (
        "Where does the user work?",
        "  - interest / foods\n"
        "  - work / company\n"
        "  - work / title",
        [
            {
                "tool": "search_profile",
                "args": {"topic": "work"},
                "why": "Question asks about workplace → 'work' topic returns company + title",
            },
        ],
    ),

    # ── Temporal lookups ──────────────────────────────────────────────

    (
        "When is my dentist appointment?",
        "  - interest / foods",
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
        "  - personal / name",
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
        "  - work / company",
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
        "  - interest / foods\n"
        "  - work / company",
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
        "  - personal / name",
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
        "  - work / company\n"
        "  - work / title",
        [
            {
                "tool": "search_profile",
                "args": {"topic": "work"},
                "why": "First part: workplace → 'work' topic",
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
        "  - interest / hobbies\n"
        "  - personal / name",
        [
            {
                "tool": "search_profile",
                "args": {"topic": "interest"},
                "why": "Hobbies fall under 'interest' topic",
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
        "  - interest / foods\n"
        "  - work / company",
        [
            {
                "tool": "search_profile",
                "args": {"topic": "interest"},
                "why": "Food preference → 'interest' topic",
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
        "  - work / company\n"
        "  - interest / foods",
        [
            {
                "tool": "search_summary",
                "args": {"query": "pets"},
                "why": "No pet-related topic exists → try summary",
            },
        ],
    ),

    (
        "What programming languages does the user know?",
        "  - work / company",
        [
            {
                "tool": "search_summary",
                "args": {"query": "programming languages"},
                "why": "No language-related topic → search summaries",
            },
        ],
    ),
]
