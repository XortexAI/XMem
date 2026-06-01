"""Shared benchmark file and download helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx


def download_file(
    url: str,
    destination: Path,
    *,
    timeout_seconds: float = 120.0,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_destination = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with httpx.stream(
            "GET",
            url,
            follow_redirects=True,
            timeout=timeout_seconds,
        ) as response:
            response.raise_for_status()
            with partial_destination.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        partial_destination.replace(destination)
    except BaseException:
        partial_destination.unlink(missing_ok=True)
        raise
    return destination


def read_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "examples", "records", "questions"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
            return [payload]
    if suffix == ".parquet":
        return read_parquet_records(path)
    raise ValueError(
        f"Unsupported dataset format for {path}. Expected JSON, JSONL, or Parquet."
    )


def read_parquet_records(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Reading BEAM parquet files requires pyarrow. Install it with "
            "`pip install pyarrow`, or convert the dataset to JSON/JSONL and "
            "pass --dataset-path."
        ) from exc

    table = pq.read_table(path)
    return table.to_pylist()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
