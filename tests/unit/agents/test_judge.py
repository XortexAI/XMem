"""
Interactive judge test with hardcoded existing facts.

Usage:
    PYTHONPATH=. python3 tests/unit/agents/test_judge.py
    PYTHONPATH=. python3 tests/unit/agents/test_judge.py --provider gemini

Existing facts are injected via a mock so you can verify the judge's
ADD / UPDATE / DELETE / NOOP decisions against known data.
"""

import asyncio
import sys
from typing import Any, Dict, List, Optional

from src.models import get_model
from src.agents.judge import JudgeAgent, _format_similar_block
from src.storage.base import SearchResult


# ── Hardcoded "existing" facts (what's already in the store) ──────────────

EXISTING_PROFILE = [
    SearchResult(id="prof-001", content="work / company = Works at Microsoft", score=0.0, metadata={}),
    SearchResult(id="prof-002", content="food / preference = Loves pizza", score=0.0, metadata={}),
    SearchResult(id="prof-003", content="basic_info / name = Alice", score=0.0, metadata={}),
    SearchResult(id="prof-004", content="food / diet = User loves steak", score=0.0, metadata={}),
    SearchResult(id="prof-005", content="hobby / sport = Plays tennis on weekends", score=0.0, metadata={}),
]

EXISTING_TEMPORAL = [
    SearchResult(id="evt-001", content="03-15 | Birthday | User's birthday", score=0.0, metadata={}),
    SearchResult(id="evt-002", content="01-10 | Dentist | Scheduled dentist visit", score=0.0, metadata={}),
    SearchResult(id="evt-003", content="12-25 | Christmas | Family dinner at grandma's", score=0.0, metadata={}),
]

EXISTING_SUMMARY = [
    SearchResult(id="sum-001", content="User is a software engineer at Microsoft", score=0.0, metadata={}),
    SearchResult(id="sum-002", content="User enjoys hiking on weekends", score=0.0, metadata={}),
    SearchResult(id="sum-003", content="User lives in NYC", score=0.0, metadata={}),
]


# ── New items to judge against the existing facts ─────────────────────────

NEW_PROFILE_ITEMS = [
    {"topic": "work", "sub_topic": "company", "memo": "Now at Google"},
    {"topic": "food", "sub_topic": "preference", "memo": "Loves pizza"},
    {"topic": "food", "sub_topic": "diet", "memo": "User is now vegetarian"},
    {"topic": "travel", "sub_topic": "dream_dest", "memo": "Wants to visit Japan"},
]

NEW_TEMPORAL_ITEMS = [
    {"date": "03-15", "event_name": "Birthday", "desc": "User's birthday"},
    {"date": "02-10", "event_name": "Dentist", "desc": "Rescheduled dentist visit"},
    {"date": "06-01", "event_name": "Marathon", "desc": "Running first marathon"},
]

NEW_SUMMARY_ITEMS = [
    "User is a software engineer at Google",
    "User enjoys hiking on weekends",
    "User moved from NYC to San Francisco",
    "User adopted a cat named Luna",
]

DOMAIN_NEW_ITEMS = {
    "profile": NEW_PROFILE_ITEMS,
    "temporal": NEW_TEMPORAL_ITEMS,
    "summary": NEW_SUMMARY_ITEMS,
}

DOMAIN_EXISTING = {
    "profile": EXISTING_PROFILE,
    "temporal": EXISTING_TEMPORAL,
    "summary": EXISTING_SUMMARY,
}


# ── Keyword matcher to simulate similarity search ─────────────────────────

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "or", "s",
    "she", "that", "the", "to", "was", "were", "will", "with",
    "user", "user's",
}


def _meaningful_words(text: str) -> set:
    return {w for w in text.lower().split() if w not in STOP_WORDS}


def _find_best_match(
    query: str,
    existing: List[SearchResult],
    threshold: float = 0.2,
) -> List[SearchResult]:
    q_words = _meaningful_words(query)
    best: Optional[SearchResult] = None
    best_overlap = 0.0

    for record in existing:
        r_words = _meaningful_words(record.content)
        if not r_words:
            continue
        overlap = len(q_words & r_words) / len(q_words | r_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best = record

    if best and best_overlap >= threshold:
        return [SearchResult(
            id=best.id,
            content=best.content,
            score=round(best_overlap, 2),
            metadata=best.metadata,
        )]
    return []


# ── Monkey-patch: override _fetch_similar to use hardcoded data ───────────

async def _mock_fetch_similar(
    self,
    items_strings: List[str],
    new_items: list,
    user_id: str,
    domain: Any,
) -> Dict[str, List[SearchResult]]:
    existing = DOMAIN_EXISTING.get(domain.value, [])
    matches: Dict[str, List[SearchResult]] = {}
    for item_str in items_strings:
        matches[item_str] = _find_best_match(item_str, existing)
    return matches


def _print_existing(domain: str) -> None:
    existing = DOMAIN_EXISTING.get(domain, [])
    print(f"\n   Existing {domain} facts in store:")
    for r in existing:
        print(f"     [{r.id}]  {r.content}")
    print()


def _print_new(domain: str) -> None:
    items = DOMAIN_NEW_ITEMS.get(domain, [])
    print(f"   New {domain} items to judge:")
    for i, item in enumerate(items, 1):
        if isinstance(item, dict):
            display = " | ".join(str(v) for v in item.values() if v)
        else:
            display = str(item)
        print(f"     {i}. {display}")
    print()


# ── Expected results (for reference while testing) ────────────────────────

EXPECTED = {
    "profile": [
        "1. UPDATE  work / company → Google  (was Microsoft, prof-001)",
        "2. NOOP    food / preference → pizza  (exact dup, prof-002)",
        "3. UPDATE  food / diet → vegetarian  (contradicts steak, prof-004)",
        "4. ADD     travel / dream_dest → Japan  (brand new)",
    ],
    "temporal": [
        "1. NOOP    Birthday 03-15  (exact dup, evt-001)",
        "2. DELETE  Dentist old date 01-10  (evt-002) + ADD new date 02-10",
        "3. ADD     Marathon 06-01  (brand new)",
    ],
    "summary": [
        "1. UPDATE  engineer at Google  (was Microsoft, sum-001)",
        "2. NOOP    hiking on weekends  (exact dup, sum-002)",
        "3. UPDATE  moved NYC→SF  (outdates 'lives in NYC', sum-003)",
        "4. ADD     cat named Luna  (brand new)",
    ],
}


async def main():
    provider = None
    if "--provider" in sys.argv:
        idx = sys.argv.index("--provider")
        provider = sys.argv[idx + 1]

    model = get_model(provider=provider)
    agent = JudgeAgent(model=model, vector_store=None, top_k=1)

    # Patch to use hardcoded existing facts
    agent._fetch_similar = _mock_fetch_similar.__get__(agent, JudgeAgent)

    model_name = getattr(model, "model", getattr(model, "model_name", "unknown"))
    print(f"\n  Judge Agent ready  (model: {model_name})")
    print(f"  Using hardcoded existing facts for similarity matching.\n")
    print(f"  Commands:")
    print(f"    profile  — judge profile items")
    print(f"    temporal — judge temporal items")
    print(f"    summary  — judge summary items")
    print(f"    q        — quit\n")

    while True:
        try:
            cmd = input(">> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd in ("q", "quit", "exit"):
            break
        if cmd not in DOMAIN_NEW_ITEMS:
            print(f"   Unknown command. Choose: profile, temporal, summary, q\n")
            continue

        _print_existing(cmd)
        _print_new(cmd)

        items = DOMAIN_NEW_ITEMS[cmd]
        print(f"   → Judging {len(items)} item(s)...\n")

        result = await agent.arun({
            "domain": cmd,
            "new_items": items,
            "user_id": "test_user",
        })

        print(f"   ┌─ Judge Output ─────────────────────────────────────")
        if result.is_empty:
            print(f"   │  (no operations)")
        else:
            for i, op in enumerate(result.operations, 1):
                preview = (op.content[:55] + "...") if len(op.content) > 55 else op.content
                print(f"   │  {i}. [{op.type.value:6s}]  {preview}")
                if op.reason:
                    print(f"   │     reason: {op.reason}")
                if op.embedding_id:
                    print(f"   │     target: {op.embedding_id}")
            print(f"   │  Confidence: {result.confidence:.2f}")
        print(f"   └────────────────────────────────────────────────────\n")

        print(f"   Expected:")
        for line in EXPECTED.get(cmd, []):
            print(f"     {line}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
