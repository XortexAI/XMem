"""Configuration for the BEAM benchmark harness."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_API_BASE_URL = "https://api.xmem.in"
DEFAULT_API_KEY_ENV = "XMEM_API_KEY"
DEFAULT_SPLIT = "1M"
DEFAULT_DATASET_URLS = {
    "100K": (
        "https://huggingface.co/datasets/Mohammadta/BEAM/resolve/main/"
        "data/100K-00000-of-00001.parquet"
    ),
    "500K": (
        "https://huggingface.co/datasets/Mohammadta/BEAM/resolve/main/"
        "data/500K-00000-of-00001.parquet"
    ),
    "1M": (
        "https://huggingface.co/datasets/Mohammadta/BEAM/resolve/main/"
        "data/1M-00000-of-00001.parquet"
    ),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset_path: Path
    output_dir: Path
    api_base_url: str = DEFAULT_API_BASE_URL
    api_key_env: str = DEFAULT_API_KEY_ENV
    api_timeout_seconds: float = 120.0
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    batch_size: int = 25
    ingest_api_version: str = "v2"
    poll_interval_seconds: float = 2.0
    poll_timeout_seconds: float = 1800.0
    top_k: int = 10
    effort_level: str = "low"
    user_prefix: str = "beam"
    limit: int | None = None
    offset: int = 0
    question_type: str | None = None
    sample_percent_per_question_type: float | None = None
    sample_min_per_question_type: int = 1
    sample_seed: int = 13
    split: str = DEFAULT_SPLIT
    skip_ingest: bool = False
    resume: bool = True
    dry_run: bool = False

    @property
    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "").strip()

    def require_api_key(self) -> str:
        api_key = self.api_key
        if not api_key:
            raise RuntimeError(
                f"Missing API key. Set {self.api_key_env} before running BEAM."
            )
        return api_key
