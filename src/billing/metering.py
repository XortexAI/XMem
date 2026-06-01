from __future__ import annotations

import math
from typing import Any, Mapping

from src.billing.types import CreditEstimate
from src.utils import billing as billing_config


def _ingest_text(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        str(payload.get(key) or "")
        for key in ("user_query", "agent_response")
        if payload.get(key)
    )


def content_tokens_for_job(job_type: str, payload: Mapping[str, Any]) -> int:
    if job_type == "memory_batch_ingest":
        return sum(
            content_tokens_for_job("memory_ingest", item)
            for item in list(payload.get("items") or [])
            if isinstance(item, Mapping)
        )
    if job_type == "memory_ingest":
        return billing_config.estimate_tokens(_ingest_text(payload))
    if job_type == "memory_retrieve":
        return billing_config.estimate_tokens(str(payload.get("query") or ""))
    return billing_config.estimate_tokens(str(payload))


def estimate_required_credits(
    job_type: str,
    payload: Mapping[str, Any],
    *,
    include_reservation_buffer: bool = True,
) -> CreditEstimate:
    tokens = content_tokens_for_job(job_type, payload)
    multiplier = billing_config.workflow_multiplier(job_type, payload)
    billable = max(1, math.ceil(tokens * multiplier))
    buffer = billing_config.RESERVATION_BUFFER_MULTIPLIER if include_reservation_buffer else 1.0
    reserved = max(billable, math.ceil(billable * buffer))
    return CreditEstimate(
        job_type=job_type,
        content_tokens=tokens,
        multiplier=multiplier,
        billable_credits=billable,
        reserved_credits=reserved,
    )
