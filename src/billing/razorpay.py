from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Any, Optional

import httpx

from src.config import settings

logger = logging.getLogger("xmem.billing.razorpay")

RAZORPAY_API = "https://api.razorpay.com/v1"


class RazorpayConfigError(RuntimeError):
    pass


def require_razorpay_keys() -> tuple[str, str]:
    if not settings.razorpay_key_id or not settings.razorpay_key_secret:
        raise RazorpayConfigError(
            "Razorpay is not configured. Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET."
        )
    return settings.razorpay_key_id, settings.razorpay_key_secret


def verify_order_signature(order_id: str, payment_id: str, signature: str) -> bool:
    _, secret = require_razorpay_keys()
    payload = f"{order_id}|{payment_id}".encode("utf-8")
    expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def verify_subscription_signature(
    subscription_id: str,
    payment_id: str,
    signature: str,
) -> bool:
    _, secret = require_razorpay_keys()
    payload = f"{payment_id}|{subscription_id}".encode("utf-8")
    expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def verify_webhook_signature(body: bytes, signature: str) -> bool:
    if not settings.razorpay_webhook_secret:
        raise RazorpayConfigError("RAZORPAY_WEBHOOK_SECRET is not configured.")
    expected = hmac.new(
        settings.razorpay_webhook_secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


async def create_order(
    *,
    amount_paise: int,
    currency: str,
    receipt: str,
    notes: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    key_id, key_secret = require_razorpay_keys()
    payload = {
        "amount": amount_paise,
        "currency": currency,
        "receipt": receipt,
        "notes": notes or {},
    }
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(
            f"{RAZORPAY_API}/orders",
            auth=(key_id, key_secret),
            json=payload,
        )
    if response.status_code >= 400:
        logger.warning("Razorpay order creation failed: %s %s", response.status_code, response.text[:500])
        response.raise_for_status()
    return response.json()


async def create_subscription(
    *,
    plan_id: str,
    total_count: int = 120,
    notes: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    key_id, key_secret = require_razorpay_keys()
    payload = {
        "plan_id": plan_id,
        "total_count": total_count,
        "quantity": 1,
        "customer_notify": 1,
        "notes": notes or {},
    }
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(
            f"{RAZORPAY_API}/subscriptions",
            auth=(key_id, key_secret),
            json=payload,
        )
    if response.status_code >= 400:
        logger.warning("Razorpay subscription creation failed: %s %s", response.status_code, response.text[:500])
        response.raise_for_status()
    return response.json()
