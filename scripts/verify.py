from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def log(message: str) -> None:
    print(f"[xmem] {message}")


def request_json(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=payload, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Request failed: {method} {url}\nHTTP {exc.code}\n{text}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed: {method} {url}\n{exc}") from exc


def health_ready(health: dict[str, Any]) -> bool:
    data = health.get("data") or health
    return bool(data.get("pipelines_ready"))


def health_summary(health: dict[str, Any]) -> str:
    data = health.get("data") or health
    return (
        f"status={data.get('status')}, "
        f"pipelines_ready={data.get('pipelines_ready')}, "
        f"error={data.get('error')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a local XMem API")
    parser.add_argument("--base-url", "-BaseUrl", default="http://localhost:8000")
    parser.add_argument("--api-key", "-ApiKey", default="dev-xmem-key")
    parser.add_argument("--user-id", "-UserId", default="xmem-local-user")
    parser.add_argument("--timeout-seconds", "-TimeoutSeconds", type=int, default=180)
    args = parser.parse_args()

    deadline = time.time() + args.timeout_seconds
    health: dict[str, Any] | None = None
    log(f"Waiting for API health at {args.base_url}/health")
    while time.time() < deadline:
        try:
            health = request_json(f"{args.base_url}/health", timeout=10)
            if health_ready(health):
                break
        except Exception:
            time.sleep(3)

    if not health:
        raise RuntimeError(f"XMem API did not become reachable within {args.timeout_seconds} seconds.")

    log(f"Health: {health_summary(health)}")
    if not health_ready(health):
        raise RuntimeError("XMem API is reachable but pipelines are not ready.")

    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }

    log("Ingesting a smoke-test memory")
    ingest = request_json(
        f"{args.base_url}/v1/memory/ingest",
        method="POST",
        headers=headers,
        body={
            "user_query": "Remember that XMem local mode runs directly from the main XMem repository.",
            "agent_response": "Got it. I will remember that XMem local mode runs from the main repository.",
            "user_id": args.user_id,
            "effort_level": "low",
        },
        timeout=650,
    )
    log(f"Ingest status: {ingest.get('status')}")

    log("Searching memory")
    search = request_json(
        f"{args.base_url}/v1/memory/search",
        method="POST",
        headers=headers,
        body={
            "query": "What is XMem local mode?",
            "user_id": args.user_id,
            "domains": ["profile", "temporal", "summary"],
            "top_k": 5,
        },
        timeout=180,
    )
    result_count = len((search.get("data") or {}).get("results") or [])
    log(f"Search result count: {result_count}")

    log("Retrieving answer")
    retrieve = request_json(
        f"{args.base_url}/v1/memory/retrieve",
        method="POST",
        headers=headers,
        body={
            "query": "Where does XMem local mode run from?",
            "user_id": args.user_id,
            "top_k": 5,
        },
        timeout=240,
    )
    print("\nAnswer:")
    print((retrieve.get("data") or {}).get("answer"))
    print("")
    log("Verification complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
