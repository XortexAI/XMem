"""Editable billing knobs for XMem.

Change this file when plan pricing, monthly credits, top-up packs, or
workflow credit consumption rules need to change. The billing service imports
these values at runtime so the rest of the implementation can stay stable.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

PLANS: dict[str, dict[str, Any]] = {
    "free": {
        "name": "Free Trial",
        "price_paise": 0,
        "currency": "INR",
        "trial_credits": 10_000,
        "trial_days": 30,
        "monthly_credits": 0,
    },
    "pro": {
        "name": "Pro",
        "price_paise": 9_900,
        "currency": "INR",
        "monthly_credits": 5_000,
    },
}

TOP_UP_PACKS: dict[str, dict[str, Any]] = {
    "topup_99": {"price_paise": 9_900, "credits": 5_000, "currency": "INR"},
    "topup_199": {"price_paise": 19_900, "credits": 12_000, "currency": "INR"},
    "topup_499": {"price_paise": 49_900, "credits": 35_000, "currency": "INR"},
}

WORKFLOW_MULTIPLIERS: dict[str, float] = {
    "memory_ingest_low": 1.0,
    "memory_ingest_standard": 1.5,
    "memory_ingest_high": 2.5,
    "memory_batch_ingest": 1.5,
    "memory_retrieve": 0.5,
}

RESERVATION_BUFFER_MULTIPLIER = 1.25
TOKEN_ESTIMATE_CHARS_PER_TOKEN = 4
TOP_UP_EXPIRY_DAYS = 365

# Provider usage is priced in USD, while XMem sells credits in INR through
# Razorpay. These defaults keep one XMem credit close to the current Pro/top-up
# value: Rs.99 / 5,000 credits = 1.98 paise per credit.
USD_TO_INR_RATE = 85.0
CREDIT_VALUE_PAISE = 2.0

# Revenue charged to users as a multiple of raw provider cost. A value of 2.0
# means 50% gross margin before non-LLM infrastructure costs.
MODEL_COST_MARKUP_MULTIPLIER = 2.0
MIN_MODEL_USAGE_CREDITS = 1


def estimate_tokens(text: str) -> int:
    """Approximate billable tokens when provider usage is not available."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / TOKEN_ESTIMATE_CHARS_PER_TOKEN))


def workflow_multiplier(job_type: str, payload: Mapping[str, Any]) -> float:
    if job_type == "memory_ingest":
        effort = str(payload.get("effort_level") or "low").strip().lower()
        if effort == "high":
            return WORKFLOW_MULTIPLIERS["memory_ingest_high"]
        if effort in {"standard", "medium"}:
            return WORKFLOW_MULTIPLIERS["memory_ingest_standard"]
        return WORKFLOW_MULTIPLIERS["memory_ingest_low"]
    return WORKFLOW_MULTIPLIERS.get(job_type, 1.0)


def nominal_paise_per_credit(plan_id: str) -> float:
    plan = PLANS[plan_id]
    credits = int(plan.get("monthly_credits") or plan.get("trial_credits") or 0)
    if credits <= 0:
        return 0.0
    return float(plan["price_paise"]) / credits
