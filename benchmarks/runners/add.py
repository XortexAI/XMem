"""
XMEM Benchmark - ADD Phase

Processes LoCoMo conversations session-by-session.
For each session: process speaker_a messages, then speaker_b messages.

Flow:
  For each session:
    1. Process all speaker_a messages in that session
    2. Process all speaker_b messages in that session
  Move to next session

This ensures temporal ordering within conversations is preserved.

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

# Suppress noisy Neo4j notification logs
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)

# Configure main logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
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


def get_speaker_messages_in_session(
    session_messages: List[Dict],
    speaker_name: str,
    session_key: str,
    session_datetime: str,
) -> List[Dict[str, Any]]:
    """
    Extract all messages from a specific speaker within a single session.
    
    Returns:
        List of message dicts with keys: text, dia_id, session_key, session_datetime, image_url
    """
    messages = []
    for msg in session_messages:
        if msg.get("speaker") == speaker_name and msg.get("text"):
            # Handle img_url as array - take first URL if present
            img_urls = msg.get("img_url", [])
            image_url = img_urls[0] if img_urls else ""
            
            messages.append({
                "text": msg["text"],
                "dia_id": msg.get("dia_id", ""),
                "session_key": session_key,
                "session_datetime": session_datetime,
                "image_url": image_url,
                "blip_caption": msg.get("blip_caption", ""),
            })
    return messages


def estim_tokens(texts: List[str]) -> float:
    """Rough token estimate (4 chars ≈ 1 token)."""
    return sum(len(t) for t in texts) / 4


# ─── core processing ───────────────────────────────────────────────────────

async def process_messages(
    messages: List[Dict[str, Any]],
    user_id: str,
    speaker_name: str,
    session_key: str,
    pipeline,
) -> Dict[str, Any]:
    """
    Process messages for one speaker in one session through the ingest pipeline.
    """
    facts_stored = 0
    events_stored = 0
    images_processed = 0
    all_facts: List[str] = []
    errors: List[str] = []
    
    for i, msg in enumerate(messages):
        has_image = bool(msg.get("image_url"))
        dia_id = msg.get("dia_id", f"msg_{i}")
        
        try:
            # Run the ingest pipeline
            result = await pipeline.run(
                user_query=msg["text"],
                agent_response="",
                user_id=user_id,
                session_datetime=msg.get("session_datetime", ""),
                image_url=msg.get("image_url", ""),
            )
            
            # Count profile facts
            profile_weaver = result.get("profile_weaver")
            if profile_weaver:
                facts_stored += profile_weaver.succeeded
            
            # Count summary facts
            summary_weaver = result.get("summary_weaver")
            if summary_weaver:
                facts_stored += summary_weaver.succeeded
                summary_result = result.get("summary_result")
                if summary_result and not summary_result.is_empty:
                    all_facts.append(summary_result.summary)
            
            # Count temporal events
            temporal_weaver = result.get("temporal_weaver")
            if temporal_weaver:
                events_stored += temporal_weaver.succeeded
            
            # Count image processing
            image_weaver = result.get("image_weaver")
            if image_weaver and image_weaver.succeeded > 0:
                images_processed += 1
            
        except Exception as e:
            error_msg = f"{dia_id}: {e}"
            errors.append(error_msg)
            logger.warning(f"      [ERROR] {error_msg}")
    
    return {
        "messages_processed": len(messages),
        "facts_stored": facts_stored,
        "events_stored": events_stored,
        "images_processed": images_processed,
        "all_facts": all_facts,
        "errors": errors,
    }


# ─── main ──────────────────────────────────────────────────────────────────

async def main():
    from src.pipelines.ingest import IngestPipeline
    from src.config import settings
    
    print("=" * 80)
    print("XMEM BENCHMARK - ADD PHASE (Session-by-Session)")
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
    
    # ── Initialize pipeline ──────────────────────────────────────────────
    print("\n[SETUP]")
    logger.info("Initializing IngestPipeline...")
    
    pipeline = IngestPipeline()
    logger.info("Pipeline initialized")
    
    # Define user IDs
    user_a_id = f"{speaker_a.lower()}_benchmark"
    user_b_id = f"{speaker_b.lower()}_benchmark"
    
    print(f"  User A: {speaker_a} → {user_a_id}")
    print(f"  User B: {speaker_b} → {user_b_id}")
    
    # ── Clear old data ────────────────────────────────────────────────────
    print("\n[CLEAR OLD DATA]")
    try:
        pipeline.neo4j.delete_user_events(user_a_id)
        pipeline.neo4j.delete_user_events(user_b_id)
        logger.info(f"Cleared Neo4j events for both users")
    except Exception as e:
        logger.warning(f"Clear failed (may be OK): {e}")
    
    # ── Process sessions ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PROCESSING SESSIONS")
    print("=" * 80)
    
    start_time = time.time()
    
    # Aggregated results
    total_a = {"messages": 0, "facts": 0, "events": 0, "images": 0, "errors": [], "all_facts": []}
    total_b = {"messages": 0, "facts": 0, "events": 0, "images": 0, "errors": [], "all_facts": []}
    all_input_texts: List[str] = []
    
    for session_idx, (session_key, session_messages, session_datetime) in enumerate(sessions, 1):
        print(f"\n{'─' * 60}")
        print(f"SESSION {session_idx}/{len(sessions)}: {session_key}")
        print(f"  Datetime: {session_datetime}")
        print(f"  Messages: {len(session_messages)}")
        print(f"{'─' * 60}")
        
        # Get messages for each speaker in this session
        msgs_a = get_speaker_messages_in_session(
            session_messages, speaker_a, session_key, session_datetime
        )
        msgs_b = get_speaker_messages_in_session(
            session_messages, speaker_b, session_key, session_datetime
        )
        
        # Count images in this session
        images_a = sum(1 for m in msgs_a if m.get("image_url"))
        images_b = sum(1 for m in msgs_b if m.get("image_url"))
        
        print(f"  {speaker_a}: {len(msgs_a)} messages ({images_a} with images)")
        print(f"  {speaker_b}: {len(msgs_b)} messages ({images_b} with images)")
        
        # Track input texts
        all_input_texts.extend([m["text"] for m in msgs_a])
        all_input_texts.extend([m["text"] for m in msgs_b])
        
        # ── Process Speaker A in this session ────────────────────────────
        if msgs_a:
            print(f"\n    Processing {speaker_a}...")
            result_a = await process_messages(
                messages=msgs_a,
                user_id=user_a_id,
                speaker_name=speaker_a,
                session_key=session_key,
                pipeline=pipeline,
            )
            total_a["messages"] += result_a["messages_processed"]
            total_a["facts"] += result_a["facts_stored"]
            total_a["events"] += result_a["events_stored"]
            total_a["images"] += result_a["images_processed"]
            total_a["errors"].extend(result_a["errors"])
            total_a["all_facts"].extend(result_a["all_facts"])
            
            print(f"      → {result_a['facts_stored']} facts, {result_a['events_stored']} events, {result_a['images_processed']} images")
        
        # ── Process Speaker B in this session ────────────────────────────
        if msgs_b:
            print(f"\n    Processing {speaker_b}...")
            result_b = await process_messages(
                messages=msgs_b,
                user_id=user_b_id,
                speaker_name=speaker_b,
                session_key=session_key,
                pipeline=pipeline,
            )
            total_b["messages"] += result_b["messages_processed"]
            total_b["facts"] += result_b["facts_stored"]
            total_b["events"] += result_b["events_stored"]
            total_b["images"] += result_b["images_processed"]
            total_b["errors"].extend(result_b["errors"])
            total_b["all_facts"].extend(result_b["all_facts"])
            
            print(f"      → {result_b['facts_stored']} facts, {result_b['events_stored']} events, {result_b['images_processed']} images")
    
    elapsed_time = time.time() - start_time
    
    # ── Metrics ──────────────────────────────────────────────────────────
    total_input_tokens = estim_tokens(all_input_texts)
    total_stored_tokens = estim_tokens(total_a["all_facts"] + total_b["all_facts"])
    
    compression_ratio = total_input_tokens / total_stored_tokens if total_stored_tokens > 0 else 0
    reduction_pct = ((total_input_tokens - total_stored_tokens) / total_input_tokens * 100) if total_input_tokens > 0 else 0
    
    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ADD PHASE COMPLETE")
    print("=" * 80)
    print(f"\n  {speaker_a}:")
    print(f"    Messages:  {total_a['messages']}")
    print(f"    Facts:     {total_a['facts']}")
    print(f"    Events:    {total_a['events']}")
    print(f"    Images:    {total_a['images']}")
    print(f"    Errors:    {len(total_a['errors'])}")
    
    print(f"\n  {speaker_b}:")
    print(f"    Messages:  {total_b['messages']}")
    print(f"    Facts:     {total_b['facts']}")
    print(f"    Events:    {total_b['events']}")
    print(f"    Images:    {total_b['images']}")
    print(f"    Errors:    {len(total_b['errors'])}")
    
    print(f"\n  [TOKEN METRICS]")
    print(f"    Input tokens:       ~{int(total_input_tokens)}")
    print(f"    Stored tokens:      ~{int(total_stored_tokens)}")
    print(f"    Compression ratio:  {compression_ratio:.1f}x")
    print(f"    Reduction:          {reduction_pct:.1f}%")
    
    print(f"\n  Sessions:  {len(sessions)}")
    print(f"  Time:      {elapsed_time:.1f}s")
    
    # ── Save results ─────────────────────────────────────────────────────
    results_dir = os.path.join(benchmarks_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    add_results = {
        "sample_id": sample_id,
        "speaker_a": {
            "name": speaker_a,
            "user_id": user_a_id,
            "messages_processed": total_a["messages"],
            "facts_stored": total_a["facts"],
            "events_stored": total_a["events"],
            "images_processed": total_a["images"],
            "errors": total_a["errors"],
        },
        "speaker_b": {
            "name": speaker_b,
            "user_id": user_b_id,
            "messages_processed": total_b["messages"],
            "facts_stored": total_b["facts"],
            "events_stored": total_b["events"],
            "images_processed": total_b["images"],
            "errors": total_b["errors"],
        },
        "sessions_count": len(sessions),
        "metrics": {
            "input_tokens": int(total_input_tokens),
            "stored_tokens": int(total_stored_tokens),
            "compression_ratio": round(compression_ratio, 2),
            "reduction_percent": round(reduction_pct, 2),
            "total_time_seconds": round(elapsed_time, 2),
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
