#!/usr/bin/env python3
"""Benchmark the durable v2 memory ingest endpoint against a live environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

TERMINAL_STATUSES = {"succeeded", "dead_letter"}


def _default_payload() -> dict[str, Any]:
    return {
        "user_query": "Benchmark the v2 durable ingest endpoint.",
        "agent_response": "This is a staging benchmark payload.",
        "session_datetime": datetime.now(timezone.utc).isoformat(),
        "effort_level": "low",
    }


def _load_payload(path: str | None) -> dict[str, Any]:
    if path is None:
        return _default_payload()
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("payload file must contain a JSON object")
    return payload


def _api_data(body: dict[str, Any]) -> dict[str, Any]:
    data = body.get("data", body)
    if not isinstance(data, dict):
        raise ValueError("API response did not contain an object data field")
    return data


def _status_endpoint(base_url: str, status_url: str) -> str:
    if status_url.startswith(("http://", "https://")):
        return status_url
    return urljoin(base_url.rstrip("/") + "/", status_url.lstrip("/"))


def _request_json(
    client: httpx.Client,
    method: str,
    url: str,
    **kwargs: Any,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = client.request(method, url, **kwargs)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("API response was not a JSON object")
    return body, elapsed_ms


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"{args.api_key_env} is required")

    base_url = args.base_url.rstrip("/")
    ingest_url = f"{base_url}/v2/memory/ingest"
    payload = _load_payload(args.payload_file)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    overall_started = time.perf_counter()
    with httpx.Client(timeout=args.request_timeout) as client:
        accepted_body, enqueue_latency_ms = _request_json(
            client,
            "POST",
            ingest_url,
            headers=headers,
            json=payload,
        )
        accepted = _api_data(accepted_body)
        job_id = accepted.get("job_id")
        status_url = accepted.get("status_url")
        if not isinstance(job_id, str) or not isinstance(status_url, str):
            raise ValueError("v2 ingest response must include job_id and status_url")

        status_endpoint = _status_endpoint(base_url, status_url)
        final_data = accepted
        completion_latency_ms: float | None = None
        poll_count = 0
        deadline = time.monotonic() + args.completion_timeout

        while time.monotonic() < deadline:
            status = str(final_data.get("status", "unknown"))
            if status in TERMINAL_STATUSES:
                completion_latency_ms = (
                    time.perf_counter() - overall_started
                ) * 1000
                break

            time.sleep(args.poll_interval)
            poll_count += 1
            status_body, _ = _request_json(
                client,
                "GET",
                status_endpoint,
                headers=headers,
            )
            final_data = _api_data(status_body)

        if completion_latency_ms is None:
            completion_latency_ms = (time.perf_counter() - overall_started) * 1000

    return {
        "base_url": base_url,
        "endpoint": "/v2/memory/ingest",
        "job_id": job_id,
        "initial_status": accepted.get("status"),
        "final_status": final_data.get("status"),
        "enqueue_latency_ms": round(enqueue_latency_ms, 2),
        "completion_latency_ms": round(completion_latency_ms, 2),
        "poll_count": poll_count,
        "created": accepted.get("created"),
        "status_url": status_url,
        "timed_out": str(final_data.get("status")) not in TERMINAL_STATUSES,
        "final_error": final_data.get("error"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure POST /v2/memory/ingest enqueue latency and durable job "
            "completion latency for a staging XMem API."
        ),
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Base URL for the target XMem API, for example https://staging.example.com",
    )
    parser.add_argument(
        "--api-key-env",
        default="XMEM_API_KEY",
        help="Environment variable containing the Bearer API key.",
    )
    parser.add_argument(
        "--payload-file",
        help="Optional JSON file to use instead of the default low-effort payload.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between status polling requests.",
    )
    parser.add_argument(
        "--completion-timeout",
        type=float,
        default=300.0,
        help="Maximum seconds to wait for the durable job to finish.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds for each request.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_benchmark(args)
    except Exception as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
