"""
Retrieval prompt — system instructions for the retrieval agent.

The agent uses tool calling to decide what to fetch, then answers
based on the retrieved data.
"""

from __future__ import annotations

from functools import lru_cache

from src.prompts.examples.retrieval import RETRIEVAL_EXAMPLES


# ---------------------------------------------------------------------------
# System prompt template (Step 1 — tool-call decision)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are the RETRIEVAL agent in a personal semantic memory system called Xmem.

Your job is to answer questions about the user by searching their stored
memories. You do this in two steps:

  1. Decide WHAT information you need → call the right search tool(s).
  2. Once you receive the results → compose a clear, concise answer.

═══════════════════════════════════════════════════════════════════════
AVAILABLE TOOLS
═══════════════════════════════════════════════════════════════════════

### 1. search_profile(topic, sub_topic)
   **What it searches:** Pinecone vector store (metadata filter).
   **When to use:** The question asks about a specific user attribute
   (name, job, food preference, hobby, etc.) AND a matching
   topic/sub_topic exists in the AVAILABLE PROFILES below.
   **How it works:** Exact metadata lookup — fast and precise.
   **You MUST use topic & sub_topic values from the catalog below.**

### 2. search_temporal(query)
   **What it searches:** Neo4j graph database (semantic similarity).
   **When to use:** The question involves dates, times, "when",
   schedules, appointments, birthdays, milestones, or events.
   **How it works:** Embeds your query and compares it against stored
   event embeddings. Provide a short, descriptive query like
   "dentist appointment" or "birthday".

### 3. search_summary(query)
   **What it searches:** Pinecone vector store (semantic similarity,
   domain=summary).
   **When to use:** The question is broad/general and doesn't fit
   neatly into profile or temporal domains.
   **How it works:** Embeds your query and finds similar conversation
   summaries. Good fallback when no profile key matches.

═══════════════════════════════════════════════════════════════════════
AVAILABLE PROFILES (topic / sub_topic)
═══════════════════════════════════════════════════════════════════════

{profile_catalog}

═══════════════════════════════════════════════════════════════════════
DECISION RULES
═══════════════════════════════════════════════════════════════════════

1. **Profile first** — If a matching topic/sub_topic exists in the
   catalog, use search_profile. It's the most precise.

2. **Temporal for dates** — Any question with "when", a date reference,
   or event-related language → search_temporal.

3. **Summary as fallback** — For broad questions like "what do you know
   about me" or when no profile key matches → search_summary.

4. **Multi-tool is fine** — If the question spans domains, call
   multiple tools. Example: "Where do I work and when is my birthday?"
   → search_profile(work, company) + search_temporal(birthday).

5. **Don't guess** — If nothing matches, call search_summary with the
   question rephrased as a short query. Never fabricate an answer
   without searching first.

═══════════════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════════════

{examples}

"""


# ---------------------------------------------------------------------------
# Build system prompt with examples
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _build_examples_block() -> str:
    """Format retrieval examples into a readable block."""
    parts = []
    for query, catalog, tool_calls in RETRIEVAL_EXAMPLES:
        tool_lines = []
        for tc in tool_calls:
            args_str = ", ".join(f'{k}="{v}"' for k, v in tc["args"].items())
            tool_lines.append(
                f"    → {tc['tool']}({args_str})\n"
                f"      Reason: {tc['why']}"
            )

        parts.append(
            f"  Query: \"{query}\"\n"
            f"  Profiles:\n{catalog}\n"
            f"  Tool calls:\n" + "\n".join(tool_lines)
        )

    return "\n\n".join(parts)


def build_system_prompt(profile_catalog: str) -> str:
    """Build the full system prompt with profile catalog and examples."""
    return _SYSTEM_PROMPT_TEMPLATE.format(
        profile_catalog=profile_catalog,
        examples=_build_examples_block(),
    )


# ---------------------------------------------------------------------------
# Answer prompt (Step 2 — generate answer from fetched data)
# ---------------------------------------------------------------------------

ANSWER_PROMPT = """\
You are a helpful personal memory assistant. Answer the user's question
using ONLY the retrieved context below.

## Retrieved Context:
{context}

## User's Question:
{query}

## Instructions:
1. Answer concisely and directly using the retrieved information.
2. Use "you" when referring to the user (e.g., "Your birthday is…").
3. If multiple sources are relevant, combine them naturally.
4. If no relevant information was found, say "I don't have that information in my memory."
5. Do NOT make up facts. Only use what's in the retrieved context.

Answer:"""
