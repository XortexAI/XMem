"""Shared async HTTP client for XMem benchmark harnesses."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx


TERMINAL_JOB_STATUSES = {"succeeded", "dead_letter"}


@dataclass(frozen=True)
class ApiCallResult:
    data: dict[str, Any]
    elapsed_ms: float


class XMemApiClient:
    """Small async client around the Python XMem API."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_seconds),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "xmem-benchmark/1.0",
            },
        )

    async def __aenter__(self) -> "XMemApiClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def batch_ingest_v1(self, items: list[dict[str, Any]]) -> ApiCallResult:
        return await self._post("/v1/memory/batch-ingest", {"items": items})

    async def batch_ingest_v2(self, items: list[dict[str, Any]]) -> ApiCallResult:
        return await self._post("/v2/memory/batch-ingest", {"items": items})

    async def retrieve(self, payload: dict[str, Any]) -> ApiCallResult:
        return await self._post("/v1/memory/retrieve", payload)

    async def job_status(self, status_url: str) -> ApiCallResult:
        return await self._get(status_url)

    async def poll_job(
        self,
        status_url: str,
        *,
        interval_seconds: float,
        timeout_seconds: float,
    ) -> ApiCallResult:
        deadline = time.monotonic() + timeout_seconds
        last_result: ApiCallResult | None = None
        while time.monotonic() < deadline:
            last_result = await self.job_status(status_url)
            status = str(last_result.data.get("status") or "").lower()
            if status in TERMINAL_JOB_STATUSES:
                return last_result
            await asyncio.sleep(interval_seconds)
        status = last_result.data.get("status") if last_result else "unknown"
        raise TimeoutError(f"Timed out polling job {status_url}; last status={status}")

    async def _get(self, path: str) -> ApiCallResult:
        return await self._request("GET", path)

    async def _post(self, path: str, payload: dict[str, Any]) -> ApiCallResult:
        return await self._request("POST", path, json=payload)

    async def _request(self, method: str, path: str, **kwargs: Any) -> ApiCallResult:
        request_path = self._request_path(path)
        start = time.perf_counter()
        response: httpx.Response | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.request(method, request_path, **kwargs)
                if response.status_code < 500 and response.status_code != 429:
                    break
            except httpx.HTTPError:
                if attempt >= self.max_retries:
                    raise
            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_backoff_seconds * (attempt + 1))

        if response is None:
            raise RuntimeError(f"No response from {method} {request_path}")
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        try:
            body = response.json()
        except ValueError:
            body = {}
        if isinstance(body, dict) and body.get("status") == "error":
            error = body.get("error") or f"XMem API error from {request_path}"
            raise RuntimeError(error)
        response.raise_for_status()
        if not isinstance(body, dict):
            body = {}
        data = body.get("data")
        if data is None:
            data = {}
        if not isinstance(data, dict):
            data = {"value": data}
        return ApiCallResult(data=data, elapsed_ms=elapsed_ms)

    @staticmethod
    def _request_path(path: str) -> str:
        if path.startswith(("http://", "https://", "/")):
            return path
        return f"/{path}"
