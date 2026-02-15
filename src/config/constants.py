"""
Shared constants used across Xmem agents and prompt formatting.

These are protocol-level values that all agents rely on for structured
communication with the LLM. Changing them requires updating every
system prompt that references the separator format.
"""

# Delimiter used in the tab-separated format between LLM and agents.
# Format in prompts:  `- SOURCE::QUERY`
# Must stay in sync with all system prompts and parsing utilities.
LLM_TAB_SEPARATOR: str = "::"
