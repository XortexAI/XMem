"""Billing routes backed by the modular credit ledger."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel

from src.api.dependencies import get_current_user
from src.billing.razorpay import (
    RazorpayConfigError,
    create_order,
    create_subscription,
    require_razorpay_keys,
    verify_order_signature,
    verify_subscription_signature,
    verify_webhook_signature,
)
from src.billing.service import get_default_billing_service, public_plans, public_topups
from src.billing.types import (
    BillingSummary,
    CheckoutRequest,
    CheckoutResponse,
    LedgerEntryPublic,
    PlanPublic,
    TopUpPackPublic,
    VerifyPaymentRequest,
)
from src.config import settings
from src.utils import billing as billing_config

logger = logging.getLogger("xmem.api.billing")

router = APIRouter(prefix="/api/billing", tags=["Billing"])


class BillingSummaryResponse(BaseModel):
    summary: BillingSummary
    plans: list[PlanPublic]
    topups: list[TopUpPackPublic]


class VerifyPaymentResponse(BaseModel):
    status: str = "ok"
    summary: BillingSummary


async def require_auth(current_user: dict = Depends(get_current_user)) -> dict:
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


def _user_id(user: dict) -> str:
    user_id = user.get("id") or user.get("_id") or user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authenticated user is missing an id")
    return str(user_id)


def _receipt(user_id: str, package_id: str) -> str:
    ts = int(datetime.now(timezone.utc).timestamp())
    safe_user = user_id.replace(":", "_")[:16]
    return f"xmem-{package_id}-{safe_user}-{ts}"


def _pack_or_plan(package_id: str) -> tuple[str, dict[str, Any]]:
    if package_id in billing_config.PLANS:
        return "plan", billing_config.PLANS[package_id]
    if package_id in billing_config.TOP_UP_PACKS:
        return "topup", billing_config.TOP_UP_PACKS[package_id]
    raise HTTPException(status_code=400, detail="Unknown billing package")


def _checkout_package(package_id: str, package: dict[str, Any], region: str) -> dict[str, Any]:
    checkout_package = dict(package)
    if package_id in billing_config.PLANS:
        checkout_package.update(billing_config.plan_price(package_id, region))
    return checkout_package


def _pro_plan_id_for_region(region: str) -> str | None:
    if region == billing_config.BILLING_REGION_GLOBAL:
        return settings.razorpay_global_pro_plan_id
    return settings.razorpay_pro_plan_id


@router.get("/plans", response_model=list[PlanPublic])
async def list_billing_plans() -> list[PlanPublic]:
    return public_plans()


@router.get("/summary", response_model=BillingSummaryResponse)
async def billing_summary(current_user: dict = Depends(require_auth)) -> BillingSummaryResponse:
    service = get_default_billing_service()
    summary = await asyncio.to_thread(service.get_billing_summary, current_user)
    return BillingSummaryResponse(
        summary=summary,
        plans=public_plans(),
        topups=public_topups(),
    )


@router.post("/razorpay/order", response_model=CheckoutResponse)
async def create_razorpay_checkout(
    request: CheckoutRequest,
    current_user: dict = Depends(require_auth),
) -> CheckoutResponse:
    try:
        key_id, _ = require_razorpay_keys()
    except RazorpayConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    user_id = _user_id(current_user)
    package_type, package = _pack_or_plan(request.package_id)
    billing_region = billing_config.normalize_billing_region(request.billing_region)
    checkout_package = _checkout_package(request.package_id, package, billing_region)
    service = get_default_billing_service()
    account = await asyncio.to_thread(service.ensure_billing_account, current_user)

    if request.package_id == "free":
        raise HTTPException(status_code=400, detail="Free plan does not require checkout")

    notes = {
        "user_id": user_id,
        "billing_account_id": account["id"],
        "package_id": request.package_id,
        "package_type": package_type,
        "billing_region": billing_region,
    }
    receipt = _receipt(user_id, request.package_id)

    try:
        pro_plan_id = _pro_plan_id_for_region(billing_region)
        if request.package_id == "pro" and pro_plan_id:
            subscription = await create_subscription(
                plan_id=pro_plan_id,
                notes=notes,
            )
            checkout_id = str(subscription["id"])
            await asyncio.to_thread(
                service.store.save_checkout,
                checkout_id,
                {
                    "type": "subscription",
                    "user_id": user_id,
                    "billing_account_id": account["id"],
                    "package_id": request.package_id,
                    "billing_region": billing_region,
                    "subscription_id": checkout_id,
                    "amount": int(checkout_package["price_minor_unit"]),
                    "currency": str(checkout_package.get("currency") or "INR"),
                    "credits": int(checkout_package.get("monthly_credits") or 0),
                    "status": "created",
                },
            )
            return CheckoutResponse(
                id=checkout_id,
                subscription_id=checkout_id,
                package_id=request.package_id,
                amount=int(checkout_package["price_minor_unit"]),
                currency=str(checkout_package.get("currency") or "INR"),
                key_id=key_id,
                receipt=receipt,
            )

        amount = int(checkout_package.get("price_minor_unit") or checkout_package["price_paise"])
        order = await create_order(
            amount_paise=amount,
            currency=str(checkout_package.get("currency") or "INR"),
            receipt=receipt,
            notes=notes,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Razorpay checkout creation failed") from exc

    order_id = str(order["id"])
    await asyncio.to_thread(
        service.store.save_checkout,
        order_id,
        {
            "type": package_type,
            "user_id": user_id,
            "billing_account_id": account["id"],
            "package_id": request.package_id,
            "billing_region": billing_region,
            "order_id": order_id,
            "amount": amount,
            "currency": str(checkout_package.get("currency") or "INR"),
            "status": "created",
        },
    )
    return CheckoutResponse(
        id=order_id,
        order_id=order_id,
        package_id=request.package_id,
        amount=amount,
        currency=str(checkout_package.get("currency") or "INR"),
        key_id=key_id,
        receipt=receipt,
    )


@router.post("/topups", response_model=CheckoutResponse)
async def create_topup_checkout(
    request: CheckoutRequest,
    current_user: dict = Depends(require_auth),
) -> CheckoutResponse:
    if request.package_id not in billing_config.TOP_UP_PACKS:
        raise HTTPException(status_code=400, detail="Unknown top-up pack")
    return await create_razorpay_checkout(request, current_user)


@router.post("/razorpay/verify", response_model=VerifyPaymentResponse)
async def verify_razorpay_payment(
    request: VerifyPaymentRequest,
    current_user: dict = Depends(require_auth),
) -> VerifyPaymentResponse:
    service = get_default_billing_service()
    user_id = _user_id(current_user)

    if request.razorpay_subscription_id:
        if not verify_subscription_signature(
            request.razorpay_subscription_id,
            request.razorpay_payment_id,
            request.razorpay_signature,
        ):
            raise HTTPException(status_code=400, detail="Invalid Razorpay signature")
        checkout = await asyncio.to_thread(
            service.store.get_checkout,
            request.razorpay_subscription_id,
        )
        if not checkout:
            raise HTTPException(status_code=400, detail="Unknown Razorpay subscription checkout")
        if checkout.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Payment subscription does not belong to this user")
        if checkout.get("package_id") != "pro":
            raise HTTPException(status_code=400, detail="Payment subscription package mismatch")
        await asyncio.to_thread(
            service.grant_pro_subscription,
            user_id=user_id,
            payment_id=request.razorpay_payment_id,
            subscription_id=request.razorpay_subscription_id,
            amount=int(checkout.get("amount") or 0) or None,
            currency=checkout.get("currency"),
            billing_region=checkout.get("billing_region") or request.billing_region,
        )
    elif request.razorpay_order_id:
        if not verify_order_signature(
            request.razorpay_order_id,
            request.razorpay_payment_id,
            request.razorpay_signature,
        ):
            raise HTTPException(status_code=400, detail="Invalid Razorpay signature")
        checkout = await asyncio.to_thread(
            service.store.get_checkout,
            request.razorpay_order_id,
        )
        if not checkout:
            raise HTTPException(status_code=400, detail="Unknown Razorpay payment order")
        if checkout.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Payment order does not belong to this user")
        if checkout.get("package_id") != request.package_id:
            raise HTTPException(status_code=400, detail="Payment order package mismatch")
        package_id = str(checkout.get("package_id"))
        if package_id == "pro":
            await asyncio.to_thread(
                service.grant_pro_subscription,
                user_id=user_id,
                payment_id=request.razorpay_payment_id,
                subscription_id=request.razorpay_order_id,
                amount=int(checkout.get("amount") or 0) or None,
                currency=checkout.get("currency"),
                billing_region=checkout.get("billing_region") or request.billing_region,
            )
        else:
            await asyncio.to_thread(
                service.grant_topup,
                user_id=user_id,
                pack_id=package_id,
                payment_id=request.razorpay_payment_id,
                order_id=request.razorpay_order_id,
                amount=int(checkout.get("amount") or 0) or None,
                currency=checkout.get("currency"),
                billing_region=checkout.get("billing_region") or request.billing_region,
            )
    else:
        raise HTTPException(status_code=400, detail="Missing Razorpay order or subscription id")

    summary = await asyncio.to_thread(service.get_billing_summary, current_user)
    return VerifyPaymentResponse(summary=summary)


@router.post("/razorpay/webhook")
async def razorpay_webhook(request: Request) -> dict[str, str]:
    body = await request.body()
    signature = request.headers.get("x-razorpay-signature", "")
    try:
        if not verify_webhook_signature(body, signature):
            raise HTTPException(status_code=400, detail="Invalid Razorpay webhook signature")
    except RazorpayConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        payload = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("Razorpay webhook body is not valid JSON: %s", exc)
        raise HTTPException(status_code=400, detail="Webhook body must be valid JSON") from exc
    event_id = str(
        request.headers.get("x-razorpay-event-id")
        or payload.get("id")
        or ""
    )
    if not event_id:
        raise HTTPException(status_code=400, detail="Webhook event id is required")
    event_name = str(payload.get("event") or "")
    service = get_default_billing_service()
    if await asyncio.to_thread(service.store.has_payment_event, event_id):
        return {"status": "ignored_duplicate"}

    payment = (((payload.get("payload") or {}).get("payment") or {}).get("entity") or {})
    subscription = (((payload.get("payload") or {}).get("subscription") or {}).get("entity") or {})
    order = (((payload.get("payload") or {}).get("order") or {}).get("entity") or {})
    notes = payment.get("notes") or subscription.get("notes") or order.get("notes") or {}
    user_id = str(notes.get("user_id") or "")
    package_id = str(notes.get("package_id") or "")
    payment_id = str(payment.get("id") or "")
    order_id = str(payment.get("order_id") or order.get("id") or "")
    subscription_id = str(payment.get("subscription_id") or subscription.get("id") or "")

    if not user_id or not package_id:
        logger.info("Ignoring Razorpay webhook without XMem user/package notes: %s", event_name)
        await asyncio.to_thread(
            service.store.mark_payment_event,
            event_id,
            {"event": event_name, "payload": payload},
        )
        return {"status": "ignored"}

    if event_name in {"payment.captured", "order.paid", "subscription.charged"}:
        if not payment_id:
            logger.warning("Razorpay webhook missing payment id for grantable event: %s", event_name)
            raise HTTPException(status_code=400, detail="Webhook payment id is required for credit grant")
        elif package_id == "pro" and (subscription_id or order_id):
            await asyncio.to_thread(
                service.grant_pro_subscription,
                user_id=user_id,
                payment_id=payment_id,
                subscription_id=subscription_id or order_id,
                amount=int(payment.get("amount") or 0) or None,
                currency=payment.get("currency"),
                billing_region=notes.get("billing_region"),
            )
        elif package_id == "pro":
            logger.warning("Razorpay pro webhook missing subscription/order id: %s", event_name)
            raise HTTPException(status_code=400, detail="Webhook subscription or order id is required for credit grant")
        elif package_id in billing_config.TOP_UP_PACKS and order_id:
            await asyncio.to_thread(
                service.grant_topup,
                user_id=user_id,
                pack_id=package_id,
                payment_id=payment_id,
                order_id=order_id,
                amount=int(payment.get("amount") or 0) or None,
                currency=payment.get("currency"),
                billing_region=notes.get("billing_region"),
            )
        elif package_id in billing_config.TOP_UP_PACKS:
            logger.warning("Razorpay top-up webhook missing order id: %s", event_name)
            raise HTTPException(status_code=400, detail="Webhook order id is required for credit grant")
        else:
            logger.warning("Razorpay webhook has unknown grant package id: %s", package_id)
            raise HTTPException(status_code=400, detail="Webhook package id is not configured for credit grant")

    first_seen = await asyncio.to_thread(
        service.store.mark_payment_event,
        event_id,
        {"event": event_name, "payload": payload},
    )
    if not first_seen:
        return {"status": "ignored_duplicate"}

    return {"status": "ok"}


@router.get("/ledger", response_model=list[LedgerEntryPublic])
async def billing_ledger(
    current_user: dict = Depends(require_auth),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[LedgerEntryPublic]:
    service = get_default_billing_service()
    entries = await asyncio.to_thread(service.list_ledger, current_user, limit)
    return [
        LedgerEntryPublic(
            id=str(entry["id"]),
            type=str(entry["type"]),
            amount=int(entry["amount"]),
            idempotency_key=str(entry["idempotency_key"]),
            job_id=entry.get("job_id"),
            source=entry.get("source"),
            metadata=entry.get("metadata") or {},
            created_at=entry["created_at"],
        )
        for entry in entries
    ]
