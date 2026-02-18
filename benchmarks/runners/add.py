"""
XMEM Benchmark - ADD Phase (Refactored)

Processes LoCoMo conversations by treating each speaker's messages independently.
No agent_response pairing -- each message is processed as a standalone user_query.

Flow:
1. Collect ALL messages from speaker_a across all sessions
2. Process them through the ingest pipeline
3. Collect ALL messages from speaker_b across all sessions
4. Process them through the ingest pipeline

This matches real-world usage where each user's messages are their own memories.

Usage: python benchmarks/runners/add.py
"""
import os
import sys
import json
import time
import asyncio
from typing import List, Dict, Any

# Setup paths
runners_dir = os.path.dirname(os.path.abspath(__file__))
benchmarks_dir = os.path.dirname(runners_dir)
xmem_root = os.path.dirname(benchmarks_dir)
sys.path.insert(0, xmem_root)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(xmem_root, ".env"))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("xmem.benchmark.add")


# ─── helpers ────────────────────────────────────────────────────────────────

def get_all_sessions(conversation: Dict) -> List[tuple]:
    """
    Get all sessions from a conversation in order.

    Returns:
        List of (session_key, session_messages, session_datetime) tuples.
    """
    sessions = []
    session_num = 1
    while True:
        key = f"session_{session_num}"
        dt_key = f"session_{session_num}_date_time"
        if key not in conversation:
            break
        messages = conversation[key]
        session_datetime = conversation.get(dt_key, "")
        if messages:
            sessions.append((key, messages, session_datetime))
        session_num += 1
    return sessions


def collect_speaker_messages(
    sessions: List[tuple],
    speaker_name: str,
) -> List[Dict[str, Any]]:
    """
    Collect all messages from a specific speaker across all sessions.
    
    Returns:
        List of message dicts with keys: text, dia_id, session_key, session_datetime
    """
    messages = []
    for session_key, session_messages, session_datetime in sessions:
        for msg in session_messages:
            if msg.get("speaker") == speaker_name and msg.get("text"):
                messages.append({
                    "text": msg["text"],
                    "dia_id": msg.get("dia_id", ""),
                    "session_key": session_key,
                    "session_datetime": session_datetime,
                    "img_url": msg.get("img_url", []),
                })
    return messages


def estim_tokens(texts: List[str]) -> float:
    """Rough token estimate (4 chars ≈ 1 token)."""
    return sum(len(t) for t in texts) / 4


# ─── core processing ───────────────────────────────────────────────────────

async def process_speaker_messages(
    messages: List[Dict[str, Any]],
    user_id: str,
    speaker_name: str,
    pipeline,
) -> Dict[str, Any]:
    """
    Process all messages for one speaker through the ingest pipeline.
    
    Each message is processed as a standalone user_query (no agent_response).
    """
    facts_stored = 0
    events_stored = 0
    all_facts: List[str] = []
    errors: List[str] = []
    
    total = len(messages)
    logger.info(f"Processing {total} messages for {speaker_name} (user_id={user_id})")
    
    for i, msg in enumerate(messages):
        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"  Progress: {i + 1}/{total}")
        
        try:
            # Run the ingest pipeline with just user_query (no agent_response)
            result = await pipeline.run(
                user_query=msg["text"],
                agent_response="",  # No agent response in benchmark
                user_id=user_id,
                session_datetime=msg.get("session_datetime", ""),
            )
            
            # Count profile facts
            profile_weaver = result.get("profile_weaver")
            if profile_weaver:
                facts_stored += profile_weaver.succeeded
            
            # Count summary facts
            summary_weaver = result.get("summary_weaver")
            if summary_weaver:
                facts_stored += summary_weaver.succeeded
                # Extract the actual summary text for token counting
                summary_result = result.get("summary_result")
                if summary_result and not summary_result.is_empty:
                    all_facts.append(summary_result.summary)
            
            # Count temporal events
            temporal_weaver = result.get("temporal_weaver")
            if temporal_weaver:
                events_stored += temporal_weaver.succeeded
            
        except Exception as e:
            error_msg = f"Message {i} ({msg.get('dia_id', 'unknown')}): {e}"
            errors.append(error_msg)
            logger.warning(f"  [ERROR] {error_msg}")
    
    logger.info(f"  Completed: {facts_stored} facts, {events_stored} events stored")
    
    return {
        "messages_processed": total,
        "facts_stored": facts_stored,
        "events_stored": events_stored,
        "all_facts": all_facts,
        "errors": errors,
    }


# ─── main ──────────────────────────────────────────────────────────────────

async def main():
    from src.pipelines.ingest import IngestPipeline
    from src.graph.neo4j_client import Neo4jClient
    from src.graph.schema import setup_constraints
    from src.config import settings
    
    print("=" * 80)
    print("XMEM BENCHMARK - ADD PHASE (Refactored: No Agent Response)")
    print("=" * 80)
    
    # ── Load dataset ────────────────────────────────────────────────────
    data_path = os.path.join(benchmarks_dir, "dataset", "locomo_10%testing.json")
    logger.info(f"Loading dataset: {data_path}")
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    conversation = data["conversation"]
    qa_pairs = data.get("qa", [])
    sample_id = data.get("sample_id", "conv-0")
    
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]
    sessions = get_all_sessions(conversation)
    
    print(f"\n[CONFIG]")
    print(f"  Dataset:     {data_path}")
    print(f"  Sample ID:   {sample_id}")
    print(f"  Speakers:    {speaker_a}, {speaker_b}")
    print(f"  Sessions:    {len(sessions)}")
    print(f"  QA pairs:    {len(qa_pairs)}")
    
    # Category breakdown
    from collections import Counter
    cat_counts = Counter(q.get("category") for q in qa_pairs)
    CAT_NAMES = {1: "Single Hop", 2: "Temporal", 3: "Multi-Hop", 4: "Open Domain", 5: "Adversarial"}
    for cat in sorted(cat_counts):
        print(f"    Cat {cat} ({CAT_NAMES.get(cat, '?')}): {cat_counts[cat]}")
    
    # ── Collect all messages per speaker ─────────────────────────────────
    messages_a = collect_speaker_messages(sessions, speaker_a)
    messages_b = collect_speaker_messages(sessions, speaker_b)
    
    print(f"\n[MESSAGES]")
    print(f"  {speaker_a}: {len(messages_a)} messages")
    print(f"  {speaker_b}: {len(messages_b)} messages")
    
    # ── Initialize pipeline ──────────────────────────────────────────────
    print("\n[SETUP]")
    logger.info("Initializing IngestPipeline...")
    
    pipeline = IngestPipeline()
    logger.info("Pipeline initialized")
    
    # Define user IDs
    user_a_id = f"{speaker_a.lower()}_benchmark"
    user_b_id = f"{speaker_b.lower()}_benchmark"
    
    print(f"  User A ID: {user_a_id}")
    print(f"  User B ID: {user_b_id}")
    
    # ── Clear old data (optional) ────────────────────────────────────────
    print("\n[CLEAR OLD DATA]")
    try:
        # Clear vector store for these users (if supported)
        # Note: You may want to clear only specific user data, not the whole namespace
        logger.info("Skipping vector store clear (would affect all users)")
        
        # Clear Neo4j events for these users
        pipeline.neo4j.delete_user_events(user_a_id)
        pipeline.neo4j.delete_user_events(user_b_id)
        logger.info(f"Cleared Neo4j events for {user_a_id} and {user_b_id}")
    except Exception as e:
        logger.warning(f"Clear failed (may be OK): {e}")
    
    # ── Process Speaker A ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"PROCESSING SPEAKER A: {speaker_a}")
    print(f"{'=' * 60}")
    
    start_a = time.time()
    result_a = await process_speaker_messages(
        messages=messages_a,
        user_id=user_a_id,
        speaker_name=speaker_a,
        pipeline=pipeline,
    )
    elapsed_a = time.time() - start_a
    
    # ── Process Speaker B ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"PROCESSING SPEAKER B: {speaker_b}")
    print(f"{'=' * 60}")
    
    start_b = time.time()
    result_b = await process_speaker_messages(
        messages=messages_b,
        user_id=user_b_id,
        speaker_name=speaker_b,
        pipeline=pipeline,
    )
    elapsed_b = time.time() - start_b
    
    # ── Metrics ──────────────────────────────────────────────────────────
    total_elapsed = elapsed_a + elapsed_b
    
    # Input tokens (all raw message texts)
    input_texts_a = [m["text"] for m in messages_a]
    input_texts_b = [m["text"] for m in messages_b]
    total_input_tokens = estim_tokens(input_texts_a + input_texts_b)
    
    # Stored tokens (extracted facts)
    total_stored_tokens = estim_tokens(result_a["all_facts"] + result_b["all_facts"])
    
    compression_ratio = total_input_tokens / total_stored_tokens if total_stored_tokens > 0 else 0
    reduction_pct = ((total_input_tokens - total_stored_tokens) / total_input_tokens * 100) if total_input_tokens > 0 else 0
    
    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ADD PHASE COMPLETE")
    print("=" * 80)
    print(f"  {speaker_a}:")
    print(f"    Messages processed: {result_a['messages_processed']}")
    print(f"    Facts stored:       {result_a['facts_stored']}")
    print(f"    Events stored:      {result_a['events_stored']}")
    print(f"    Errors:             {len(result_a['errors'])}")
    print(f"    Time:               {elapsed_a:.1f}s")
    print()
    print(f"  {speaker_b}:")
    print(f"    Messages processed: {result_b['messages_processed']}")
    print(f"    Facts stored:       {result_b['facts_stored']}")
    print(f"    Events stored:      {result_b['events_stored']}")
    print(f"    Errors:             {len(result_b['errors'])}")
    print(f"    Time:               {elapsed_b:.1f}s")
    print()
    print(f"  [TOKEN METRICS (≈ 4 chars/token)]")
    print(f"    Input tokens:       ~{int(total_input_tokens)}")
    print(f"    Stored tokens:      ~{int(total_stored_tokens)}")
    print(f"    Compression ratio:  {compression_ratio:.1f}x")
    print(f"    Reduction:          {reduction_pct:.1f}%")
    print()
    print(f"  Total time:           {total_elapsed:.1f}s")
    
    # ── Save results ─────────────────────────────────────────────────────
    results_dir = os.path.join(benchmarks_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    add_results = {
        "sample_id": sample_id,
        "speaker_a": {
            "name": speaker_a,
            "user_id": user_a_id,
            "messages_processed": result_a["messages_processed"],
            "facts_stored": result_a["facts_stored"],
            "events_stored": result_a["events_stored"],
            "errors": result_a["errors"],
            "time_seconds": round(elapsed_a, 2),
        },
        "speaker_b": {
            "name": speaker_b,
            "user_id": user_b_id,
            "messages_processed": result_b["messages_processed"],
            "facts_stored": result_b["facts_stored"],
            "events_stored": result_b["events_stored"],
            "errors": result_b["errors"],
            "time_seconds": round(elapsed_b, 2),
        },
        "sessions_count": len(sessions),
        "metrics": {
            "input_tokens": int(total_input_tokens),
            "stored_tokens": int(total_stored_tokens),
            "compression_ratio": round(compression_ratio, 2),
            "reduction_percent": round(reduction_pct, 2),
            "total_time_seconds": round(total_elapsed, 2),
        },
    }
    
    results_path = os.path.join(results_dir, "add_results.json")
    with open(results_path, "w") as f:
        json.dump(add_results, f, indent=2)
    
    print(f"\n  Saved → {results_path}")
    print(f"\n  Next: python benchmarks/runners/search.py")
    
    # ── Cleanup ──────────────────────────────────────────────────────────
    pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
