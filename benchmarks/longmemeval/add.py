import os
import sys
import json
import time
import asyncio
import argparse
from typing import List, Dict, Any

# Setup paths
longmem_dir = os.path.dirname(os.path.abspath(__file__))
benchmarks_dir = os.path.dirname(longmem_dir)
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
logger = logging.getLogger("xmem.benchmark.longmemeval.add")


# ─── helpers ────────────────────────────────────────────────────────────────

def get_user_pairs_in_session(
    session_messages: List[Dict],
    session_datetime: str,
) -> List[Dict[str, Any]]:
    """
    Build conversation pairs for the user within a session.
    
    For each user message, the `agent_response` is the next message from the assistant.
    This gives the summarizer both sides of the conversation.
    
    Returns:
        List of message dicts with keys:
          text, agent_response, session_datetime
    """
    pairs = []
    
    for idx, msg in enumerate(session_messages):
        if msg.get("role") != "user":
            continue
            
        text = msg.get("content", "").strip()
        if not text:
            continue
            
        # Find the next reply from the assistant
        agent_response = ""
        for j in range(idx + 1, len(session_messages)):
            next_msg = session_messages[j]
            if next_msg.get("role") == "assistant" and next_msg.get("content"):
                agent_response = next_msg["content"].strip()
                break
            elif next_msg.get("role") == "user":
                # Stopped at the next user message without an assistant reply
                break
        
        pairs.append({
            "text": text,
            "agent_response": agent_response,
            "session_datetime": session_datetime,
        })
        
    return pairs


def estim_tokens(texts: List[str]) -> float:
    """Rough token estimate (4 chars ≈ 1 token)."""
    return sum(len(t) for t in texts) / 4


# ─── core processing ───────────────────────────────────────────────────────

async def process_messages(
    messages: List[Dict[str, Any]],
    user_id: str,
    pipeline,
) -> Dict[str, Any]:
    """
    Process messages for the user in one session through the ingest pipeline.
    """
    facts_stored = 0
    events_stored = 0
    images_processed = 0
    all_facts: List[str] = []
    errors: List[str] = []
    raw_input_chars = 0
    stored_output_chars = 0
    
    for i, msg in enumerate(messages):
        # Track raw input size
        raw_input_chars += len(msg.get("text", ""))
        raw_input_chars += len(msg.get("agent_response", ""))
        
        try:
            # Run the ingest pipeline with conversation pair context
            result = await pipeline.run(
                user_query=msg["text"],
                agent_response=msg.get("agent_response", ""),
                user_id=user_id,
                session_datetime=msg.get("session_datetime", ""),
                image_url="",  # LongMemEval is text-only
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
                    stored_output_chars += len(summary_result.summary)
            
            # Count temporal events
            temporal_weaver = result.get("temporal_weaver")
            if temporal_weaver:
                events_stored += temporal_weaver.succeeded
                
        except Exception as e:
            error_msg = f"Msg {i}: {e}"
            errors.append(error_msg)
            logger.warning(f"      [ERROR] {error_msg}")
    
    return {
        "messages_processed": len(messages),
        "facts_stored": facts_stored,
        "events_stored": events_stored,
        "images_processed": images_processed,
        "all_facts": all_facts,
        "errors": errors,
        "raw_input_chars": raw_input_chars,
        "stored_output_chars": stored_output_chars,
    }


# ─── main ──────────────────────────────────────────────────────────────────

async def main():
    from src.pipelines.ingest import IngestPipeline
    
    # ── Parse CLI args ──────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="LongMemEval Benchmark - ADD Phase")
    parser.add_argument(
        "--data", type=str, default="longmemeval_oracle.json",
        help="Dataset file inside LongMemEval/data/ (default: longmemeval_oracle.json)"
    )
    parser.add_argument(
        "--start-question", type=int, default=1,
        help="Resume from this question ID (1-based index) to process"
    )
    parser.add_argument(
        "--evalset", type=int, default=50,
        help="Number of questions to process (default: 50)"
    )
    
    args = parser.parse_args()
    data_filename = args.data
    start_q = args.start_question
    eval_size = args.evalset
    
    print("=" * 80)
    print("LONGMEMEVAL BENCHMARK - ADD PHASE")
    print("=" * 80)
    
    # ── Load dataset ────────────────────────────────────────────────────
    data_path = os.path.join(xmem_root, "LongMemEval", "data", data_filename)
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(xmem_root), "LongMemEval", "data", data_filename)
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found: {data_filename}")
        return
        
    logger.info(f"Loading dataset: {data_path}")
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    total_questions = len(data)
    end_q = min(start_q + eval_size - 1, total_questions)
    
    print(f"\n[CONFIG]")
    print(f"  Dataset:     {data_filename}")
    print(f"  Questions:   Processing Q{start_q} to Q{end_q} (out of {total_questions})")
    print(f"  Eval size:   {eval_size}")
    
    # Subset to process
    eval_data = data[start_q-1 : end_q]
    
    # ── Initialize pipeline ──────────────────────────────────────────────
    print("\n[SETUP]")
    logger.info("Initializing IngestPipeline...")
    
    pipeline = IngestPipeline()
    logger.info("Pipeline initialized")
    
    base_user_id = f"longmemeval_{data_filename.replace('.json', '')}"
    
    # ── Clear old data ────────────────────────────────────────────────────
    print("\n[CLEAR OLD DATA]")
    try:
        pipeline.neo4j.delete_user_events(f"{base_user_id}_q{start_q}")
        logger.info(f"Test cleanup format ready")
    except Exception as e:
        logger.warning(f"Clear failed (may be OK): {e}")

    # ── Process questions ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PROCESSING SESSIONS (Per Question)")
    print("=" * 80)
    
    start_time = time.time()
    
    # Aggregated results per question
    question_metrics = []
    
    for q_idx, qa_entry in enumerate(eval_data, start_q):
        q_id = qa_entry["question_id"]
        q_type = qa_entry["question_type"]
        
        # Each question in LongMemEval has its OWN isolated chat history!
        # We must namespace the user so the memory pool is reset per question.
        current_user_id = f"{base_user_id}_q{q_idx}"
        
        sessions = qa_entry["haystack_sessions"]
        session_dates = qa_entry["haystack_dates"]
        
        print(f"\n{'─' * 60}")
        print(f"QUESTION {q_idx}: {q_id} ({q_type})")
        print(f"  User ID: {current_user_id}")
        print(f"  History: {len(sessions)} sessions")
        print(f"{'─' * 60}")
        
        q_metrics = {"q_id": q_id, "messages": 0, "facts": 0, "events": 0, "errors": [], "time": 0, "raw_input_chars": 0, "stored_output_chars": 0}
        q_start = time.time()
        
        # In LongMemEval, the history is a list of sessions. Each session is a list of turns.
        for sess_idx, (sess_msgs, sess_date) in enumerate(zip(sessions, session_dates)):
            print(f"  Session {sess_idx+1}/{len(sessions)} [{sess_date}]: {len(sess_msgs)} turns")
            
            # Format custom LongMemEval datetime to match what XMEM pipeline expects usually
            # Example LongMemEval: "2023/10/22 (Sun) 01:21"
            # It already works reasonably well for Neo4j date extraction, so we pass it directly
            
            pairs = get_user_pairs_in_session(sess_msgs, sess_date)
            # print(f"    -\u003e {len(pairs)} QA pairs extracted")
            
            if pairs:
                result = await process_messages(messages=pairs, user_id=current_user_id, pipeline=pipeline)
                q_metrics["messages"] += result["messages_processed"]
                q_metrics["facts"] += result["facts_stored"]
                q_metrics["events"] += result["events_stored"]
                q_metrics["errors"].extend(result["errors"])
                q_metrics["raw_input_chars"] += result["raw_input_chars"]
                q_metrics["stored_output_chars"] += result["stored_output_chars"]
        
        q_time = time.time() - q_start
        q_metrics["time"] = q_time
        question_metrics.append(q_metrics)
        
        q_ratio = (q_metrics["raw_input_chars"] / q_metrics["stored_output_chars"] if q_metrics["stored_output_chars"] > 0 else 0)
        print(f"\n    [Q{q_idx} SUMMARY]")
        print(f"    Processed {q_metrics['messages']} pairs -> {q_metrics['facts']} facts, {q_metrics['events']} events ({q_time:.1f}s)")
        print(f"    Compression: {q_metrics['raw_input_chars']:,} chars -> {q_metrics['stored_output_chars']:,} chars ({q_ratio:.1f}x)")
        
    elapsed_time = time.time() - start_time
    
    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ADD PHASE COMPLETE")
    print("=" * 80)
    
    total_messages = sum(qm["messages"] for qm in question_metrics)
    total_facts = sum(qm["facts"] for qm in question_metrics)
    total_events = sum(qm["events"] for qm in question_metrics)
    total_errors = sum(len(qm["errors"]) for qm in question_metrics)
    total_raw = sum(qm["raw_input_chars"] for qm in question_metrics)
    total_stored = sum(qm["stored_output_chars"] for qm in question_metrics)
    raw_tokens = total_raw / 4
    stored_tokens = total_stored / 4
    comp_ratio = total_raw / total_stored if total_stored > 0 else 0
    
    print(f"\n  Total Processed over {len(eval_data)} isolated questions:")
    print(f"    Messages:  {total_messages}")
    print(f"    Facts:     {total_facts}")
    print(f"    Events:    {total_events}")
    print(f"    Errors:    {total_errors}")
    print(f"    Time:      {elapsed_time:.1f}s")
    print(f"\n  Token Compression:")
    print(f"    Raw input:         {total_raw:,} chars (~{raw_tokens:,.0f} tokens)")
    print(f"    Stored memory:     {total_stored:,} chars (~{stored_tokens:,.0f} tokens)")
    print(f"    Compression ratio: {comp_ratio:.1f}x")
    if comp_ratio > 0:
        print(f"    Space saved:       {((1 - 1/comp_ratio) * 100):.1f}%")
    
    # ── Save results ─────────────────────────────────────────────────────
    results_dir = os.path.join(longmem_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    add_results = {
        "dataset": data_filename,
        "run_range": f"Q{start_q}-Q{end_q}",
        "total_questions_processed": len(eval_data),
        "metrics": {
            "total_messages": total_messages,
            "total_facts_stored": total_facts,
            "total_events_stored": total_events,
            "total_errors": total_errors,
            "total_time_seconds": round(elapsed_time, 2),
            "compression": {
                "raw_input_chars": total_raw,
                "stored_output_chars": total_stored,
                "compression_ratio": round(comp_ratio, 2),
            },
        },
        "per_question_metrics": question_metrics
    }
    
    # Use a descriptive filename
    results_path = os.path.join(results_dir, f"add_results_{data_filename.split('.')[0]}_{start_q}-{end_q}.json")
    with open(results_path, "w") as f:
        json.dump(add_results, f, indent=2)
    
    print(f"\n  Saved → {results_path}")
    print(f"\n  Next: python benchmarks/longmemeval/search.py --data {data_filename} --start-question {start_q} --evalset {eval_size}")
    
    # ── Cleanup ──────────────────────────────────────────────────────────
    pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
