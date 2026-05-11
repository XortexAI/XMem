import os

os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("NEO4J_PASSWORD", "test-neo4j-password")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")

from src.api.routes.memory import _detect_chat_provider, _extract_chat_pairs


def test_detects_supported_public_chat_share_providers() -> None:
    assert _detect_chat_provider("https://chatgpt.com/share/abc") == "chatgpt"
    assert _detect_chat_provider("https://chat.openai.com/share/abc") == "chatgpt"
    assert _detect_chat_provider("https://claude.ai/share/abc") == "claude"
    assert _detect_chat_provider("https://gemini.google.com/share/abc") == "gemini"
    assert _detect_chat_provider("https://example.com/share/abc") == "unknown"


def test_extracts_claude_pairs_from_next_data_script() -> None:
    html = """
    <html>
      <body>
        <script id="__NEXT_DATA__" type="application/json">
        {
          "props": {
            "pageProps": {
              "conversation": {
                "messages": [
                  {"sender": "human", "text": "Summarize this release note."},
                  {"sender": "assistant", "text": "Here is a short summary."}
                ]
              }
            }
          }
        }
        </script>
      </body>
    </html>
    """

    provider, method, pairs = _extract_chat_pairs("https://claude.ai/share/abc", html)

    assert provider == "claude"
    assert method == "structured"
    assert len(pairs) == 1
    assert pairs[0].user_query == "Summarize this release note."
    assert pairs[0].agent_response == "Here is a short summary."


def test_extracts_gemini_pairs_from_public_share_dom() -> None:
    html = """
    <html>
      <body>
        <message-content role="user">Draft a launch checklist.</message-content>
        <message-content role="model">Here is a concise checklist.</message-content>
      </body>
    </html>
    """

    provider, method, pairs = _extract_chat_pairs(
        "https://gemini.google.com/share/abc",
        html,
    )

    assert provider == "gemini"
    assert method == "dom"
    assert len(pairs) == 1
    assert pairs[0].user_query == "Draft a launch checklist."
    assert pairs[0].agent_response == "Here is a concise checklist."


def test_known_provider_private_or_missing_page_does_not_use_fallback() -> None:
    html = """
    <html>
      <body>
        <h1>This conversation is private</h1>
        <p>Sign in to request access to this shared conversation.</p>
      </body>
    </html>
    """

    provider, method, pairs = _extract_chat_pairs("https://claude.ai/share/private", html)

    assert provider == "claude"
    assert method == "unavailable"
    assert pairs == []
