import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List


@dataclass
class MessagePair:
    user_query: str
    agent_response: str


def load_parser():
    source = Path("src/api/routes/memory.py").read_text()
    tree = ast.parse(source)
    wanted = {
        "_parse_cursor_transcript",
        "_parse_antigravity_transcript",
        "_content_to_text",
        "_parse_claude_code_transcript",
        "_parse_transcript_text",
    }
    module = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "Any": Any,
        "List": List,
        "MessagePair": MessagePair,
        "json": json,
        "re": re,
    }
    exec(compile(module, "memory_parser_subset", "exec"), namespace)
    return namespace["_parse_transcript_text"]


def test_parse_claude_code_jsonl_transcript():
    parse_transcript_text = load_parser()
    transcript = "\n".join([
        '{"type":"user","message":{"role":"user","content":"Add tests for login"}}',
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"I added the login tests."},{"type":"tool_use","name":"Bash"}]}}',
        '{"type":"user","message":{"role":"user","content":[{"type":"text","text":"Run them"}]}}',
        '{"type":"assistant","message":{"role":"assistant","content":"All tests passed."}}',
    ])

    format_detected, pairs = parse_transcript_text(transcript)

    assert format_detected == "claude_code"
    assert len(pairs) == 2
    assert pairs[0].user_query == "Add tests for login"
    assert pairs[0].agent_response == "I added the login tests."
    assert pairs[1].user_query == "Run them"
    assert pairs[1].agent_response == "All tests passed."


def test_parse_claude_code_ignores_tool_only_blocks():
    parse_transcript_text = load_parser()
    transcript = "\n".join([
        '{"message":{"role":"user","content":"Inspect the repo"}}',
        '{"message":{"role":"assistant","content":[{"type":"tool_use","name":"Read"}]}}',
        '{"message":{"role":"assistant","content":[{"type":"text","text":"The repo uses FastAPI."}]}}',
    ])

    format_detected, pairs = parse_transcript_text(transcript)

    assert format_detected == "claude_code"
    assert len(pairs) == 1
    assert pairs[0].user_query == "Inspect the repo"
    assert pairs[0].agent_response == "The repo uses FastAPI."
