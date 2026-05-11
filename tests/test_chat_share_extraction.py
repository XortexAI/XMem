import json
import os
from types import SimpleNamespace

os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("NEO4J_PASSWORD", "test-neo4j-password")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")

from src.api.chat_share import scrape_failure_message
from src.api.routes.memory import (
    _detect_chat_provider,
    _extract_chat_pairs,
    scrape_chat_link,
)
from src.api.schemas import MessagePair, ScrapeRequest


def test_detects_supported_chat_share_providers():
    assert _detect_chat_provider("https://chatgpt.com/share/abc") == "chatgpt"
    assert _detect_chat_provider("https://chat.openai.com/share/abc") == "chatgpt"
    assert _detect_chat_provider("https://claude.ai/share/abc") == "claude"
    assert _detect_chat_provider("https://gemini.google.com/share/abc") == "gemini"
    assert _detect_chat_provider("https://g.co/gemini/share/abc") == "gemini"


def test_extracts_chatgpt_dom_pairs():
    html = """
    <div data-message-author-role="user">What is XMem?</div>
    <div data-message-author-role="assistant">A long-term memory layer.</div>
    """

    provider, method, pairs = _extract_chat_pairs("https://chatgpt.com/share/abc", html)

    assert provider == "chatgpt"
    assert method == "dom"
    assert pairs == [
        MessagePair(
            user_query="What is XMem?",
            agent_response="A long-term memory layer.",
        )
    ]


def test_extracts_claude_preloaded_state_pairs():
    state = {
        "chat": {
            "messages": [
                {"sender": "human", "text": "Summarize this repo."},
                {"sender": "assistant", "text": "It stores memories for agents."},
            ]
        }
    }
    html = (
        "<script>window.__PRELOADED_STATE__ = "
        f"{json.dumps(state)};"
        "</script>"
    )

    provider, method, pairs = _extract_chat_pairs("https://claude.ai/share/abc", html)

    assert provider == "claude"
    assert method == "structured"
    assert pairs == [
        MessagePair(
            user_query="Summarize this repo.",
            agent_response="It stores memories for agents.",
        )
    ]


def test_extracts_claude_current_public_share_dom_pairs():
    html = """
    <div data-testid="user-message">
      <p class="whitespace-pre-wrap break-words">test test</p>
    </div>
    <div class="font-claude-response relative leading">
      <div>
        <div class="standard-markdown">
          <p class="font-claude-response-body">Hey! I'm here and working.</p>
        </div>
      </div>
    </div>
    """

    provider, method, pairs = _extract_chat_pairs("https://claude.ai/share/abc", html)

    assert provider == "claude"
    assert method == "dom"
    assert pairs == [
        MessagePair(
            user_query="test test",
            agent_response="Hey! I'm here and working.",
        )
    ]


def test_extracts_gemini_dom_pairs():
    html = """
    <message-content role="user">Compare memory tools.</message-content>
    <message-content role="model">XMem focuses on persistent agent memory.</message-content>
    """

    provider, method, pairs = _extract_chat_pairs(
        "https://gemini.google.com/share/abc",
        html,
    )

    assert provider == "gemini"
    assert method == "dom"
    assert pairs == [
        MessagePair(
            user_query="Compare memory tools.",
            agent_response="XMem focuses on persistent agent memory.",
        )
    ]


def test_extracts_gemini_current_public_share_dom_pairs():
    html = """
    <div class="query-text">
      <span class="screen-reader-user-query-label"> You said </span>
      <p class="query-text-line"> Test test </p>
    </div>
    <structured-content-container class="message-content">
      <message-content>
        <div class="markdown">
          <p>Loud and clear! I'm here and ready to roll.</p>
        </div>
      </message-content>
    </structured-content-container>
    """

    provider, method, pairs = _extract_chat_pairs(
        "https://gemini.google.com/share/abc",
        html,
    )

    assert provider == "gemini"
    assert method == "dom"
    assert pairs == [
        MessagePair(
            user_query="Test test",
            agent_response="Loud and clear! I'm here and ready to roll.",
        )
    ]


def test_scrape_failure_message_names_private_or_missing_provider_links():
    message = scrape_failure_message({"provider": "claude"})

    assert "Claude share link" in message
    assert "public" in message
    assert "expired" in message


def test_scrape_failure_message_lists_supported_unknown_links():
    message = scrape_failure_message({"provider": "unknown"})

    assert "Supported public share links" in message
    assert "ChatGPT" in message
    assert "Claude" in message
    assert "Gemini" in message


async def test_scrape_route_failure_uses_elapsed_ms(monkeypatch):
    async def fake_scrape(url: str):
        return {"provider": "gemini", "pairs": []}

    ticks = iter([10.0, 10.12345])

    monkeypatch.setattr("src.api.routes.memory._scrape_chat_share", fake_scrape)
    monkeypatch.setattr("src.api.routes.memory.time.perf_counter", lambda: next(ticks))

    response = await scrape_chat_link(
        ScrapeRequest(url="https://gemini.google.com/share/abc"),
        SimpleNamespace(state=SimpleNamespace(request_id="req-test")),
    )
    body = json.loads(response.body)

    assert response.status_code == 400
    assert body["elapsed_ms"] == 123.45
    assert "Gemini share link" in body["error"]
