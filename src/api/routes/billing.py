"""Billing and Razorpay payment routes."""

from __future__ import annotations

import hashlib
import hmac
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.dependencies import get_current_user
from src.config import settings

logger = logging.getLogger("xmem.api.billing")

router = APIRouter(prefix="/api/billing", tags=["Billing"])


class BillingPlan(BaseModel):
    id: str
    name: str
    amount: int
    currency: str
    description: str
    features: List[str]


class UsageSnapshot(BaseModel):
    memories_written: int = 0
    retrievals: int = 0
    graph_queries: int = 0
    credits_used: int = 0
    credits_limit: int = 5000


class Invoice(BaseModel):
    id: str
    date: datetime
    amount_paise: int
    status: Literal["paid", "pending", "failed"]
    credits: int = 0
    receipt_url: Optional[str] = None


class BillingSummary(BaseModel):
    plan_name: str
    account_status: Literal["active", "trial", "paused", "past_due"]
    currency: str
    credit_balance: int
    prepaid_balance_paise: int
    current_month: UsageSnapshot
    next_invoice_paise: int
    last_payment_at: Optional[datetime] = None
    invoices: List[Invoice] = Field(default_factory=list)


class BillingSummaryResponse(BaseModel):
    summary: BillingSummary
    plans: List[BillingPlan]


class CreateRazorpayOrderRequest(BaseModel):
    package_id: str = Field(..., description="Plan/package ID selected by the user")
    credits: int = Field(default=0, ge=0)
    amount: int = Field(default=0, ge=0)
    currency: str = Field(default="INR", min_length=3, max_length=3)


class RazorpayOrderResponse(BaseModel):
    id: str
    order_id: str
    amount: int
    currency: str
    key_id: str
    receipt: str
    package_id: str


class VerifyRazorpayPaymentRequest(BaseModel):
    razorpay_payment_id: str
    razorpay_order_id: str
    razorpay_signature: str
    package_id: str
    credits: int = Field(default=0, ge=0)
    amount: int = Field(default=0, ge=0)
    currency: str = Field(default="INR", min_length=3, max_length=3)


class VerifyRazorpayPaymentResponse(BaseModel):
    status: Literal["ok"]
    summary: BillingSummary


_PLANS: Dict[str, BillingPlan] = {
    "free": BillingPlan(
        id="free",
        name="Free",
        amount=0,
        currency="USD",
        description="30 days free with access to the core platform, Chrome extension, MCP, and SDKs.",
        features=[
            "Full XMem dashboard access",
            "Chrome extension included",
            "MCP server access included",
            "Python and TypeScript SDKs included",
            "No credit card required",
        ],
    ),
    "pro": BillingPlan(
        id="pro",
        name="Pro",
        amount=100,
        currency="USD",
        description="Full access for production apps, priority support, and pay-as-you-go usage.",
        features=[
            "Everything in Free",
            "Production-ready API access",
            "Pay-as-you-go usage for higher volume",
            "24/7 customer support",
            "Access to exclusive features coming soon",
        ],
    ),
    "enterprise": BillingPlan(
        id="enterprise",
        name="Enterprise",
        amount=0,
        currency="USD",
        description="Dedicated onboarding, custom limits, security reviews, and team support.",
        features=[
            "Everything in Pro",
            "Custom usage limits",
            "Security and procurement support",
            "Dedicated onboarding",
        ],
    ),
}

_in_memory_billing: Dict[str, Dict[str, Any]] = {}
_in_memory_orders: Dict[str, Dict[str, Any]] = {}


class BillingStore:
    """Small billing metadata store backed by MongoDB with local memory fallback."""

    def __init__(self) -> None:
        self._client = None
        self._db = None
        self.billing = None
        self.orders = None
        self._connected = False
        self._in_memory = False
        self._try_connect()

    def _requires_durable_storage(self) -> bool:
        return settings.environment.lower() in {"production", "prod"}

    def _enable_memory_fallback(self, error: Exception) -> None:
        message = f"MongoDB connection failed for billing storage: {error}"
        if self._requires_durable_storage():
            logger.error("%s; refusing in-memory fallback in production", message)
            raise RuntimeError(
                "MongoDB is required for billing storage when ENVIRONMENT=production"
            ) from error
        logger.warning("%s; using in-memory billing storage", message)
        self._connected = False
        self._in_memory = True

    def _try_connect(self) -> None:
        provider = (settings.app_store_provider or "mongo").strip().lower()
        if provider == "memory":
            self._connected = False
            self._in_memory = True
            return
        if provider == "postgres":
            self._enable_memory_fallback(RuntimeError("Postgres billing storage is not implemented"))
            return

        try:
            from pymongo import ASCENDING, MongoClient

            self._client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)
            self._client.admin.command("ping")
            self._db = self._client[settings.mongodb_database]
            self.billing = self._db["billing_profiles"]
            self.orders = self._db["billing_orders"]
            self.billing.create_index([("user_id", ASCENDING)], unique=True)
            self.orders.create_index([("order_id", ASCENDING)], unique=True)
            self.orders.create_index([("user_id", ASCENDING)])
            self._connected = True
            self._in_memory = False
        except Exception as exc:
            self._enable_memory_fallback(exc)

    def get_summary(self, user_id: str) -> BillingSummary:
        if self._in_memory:
            summary = _in_memory_billing.setdefault(
                user_id,
                _default_summary().model_dump(),
            )
            return BillingSummary.model_validate(summary)

        doc = self.billing.find_one({"user_id": user_id})
        if not doc:
            summary = _default_summary()
            self.save_summary(user_id, summary)
            return summary

        doc.pop("_id", None)
        doc.pop("user_id", None)
        return BillingSummary.model_validate(doc)

    def save_summary(self, user_id: str, summary: BillingSummary) -> None:
        payload = summary.model_dump()
        if self._in_memory:
            _in_memory_billing[user_id] = payload
            return

        self.billing.update_one(
            {"user_id": user_id},
            {"$set": {"user_id": user_id, **payload}},
            upsert=True,
        )

    def save_order(self, order_id: str, order: Dict[str, Any]) -> None:
        payload = {"order_id": order_id, **order}
        if self._in_memory:
            _in_memory_orders[order_id] = payload
            return

        self.orders.update_one(
            {"order_id": order_id},
            {"$set": payload},
            upsert=True,
        )

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        if self._in_memory:
            return _in_memory_orders.get(order_id)

        doc = self.orders.find_one({"order_id": order_id})
        if doc:
            doc.pop("_id", None)
        return doc


_billing_store = BillingStore()


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


def _default_summary() -> BillingSummary:
    return BillingSummary(
        plan_name="Free trial",
        account_status="trial",
        currency="INR",
        credit_balance=5000,
        prepaid_balance_paise=0,
        current_month=UsageSnapshot(),
        next_invoice_paise=0,
        invoices=[],
    )


def _get_summary(user_id: str) -> BillingSummary:
    return _billing_store.get_summary(user_id)


def _require_razorpay_config() -> tuple[str, str]:
    if not settings.razorpay_key_id or not settings.razorpay_key_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Razorpay is not configured. Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET.",
        )
    return settings.razorpay_key_id, settings.razorpay_key_secret


def _get_plan(package_id: str) -> BillingPlan:
    plan = _PLANS.get(package_id)
    if not plan:
        raise HTTPException(status_code=400, detail="Unknown billing plan")
    return plan


def _verify_signature(order_id: str, payment_id: str, signature: str, secret: str) -> bool:
    payload = f"{order_id}|{payment_id}".encode("utf-8")
    expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


@router.get("/plans", response_model=List[BillingPlan])
async def list_billing_plans() -> List[BillingPlan]:
    """Return the server-authoritative billing plans."""
    return list(_PLANS.values())


@router.get("/summary", response_model=BillingSummaryResponse)
async def billing_summary(current_user: dict = Depends(require_auth)) -> BillingSummaryResponse:
    """Return the current user's billing summary."""
    return BillingSummaryResponse(
        summary=_get_summary(_user_id(current_user)),
        plans=list(_PLANS.values()),
    )


@router.post("/razorpay/order", response_model=RazorpayOrderResponse)
async def create_razorpay_order(
    request: CreateRazorpayOrderRequest,
    current_user: dict = Depends(require_auth),
) -> RazorpayOrderResponse:
    """Create a Razorpay order for the selected plan.

    The server owns plan amount and currency. Client-supplied amount/currency are
    accepted only for compatibility and are intentionally ignored.
    """
    key_id, key_secret = _require_razorpay_config()
    plan = _get_plan(request.package_id)

    if plan.id != "pro":
        raise HTTPException(status_code=400, detail="Only the Pro plan can be purchased online")

    user_id = _user_id(current_user)
    receipt = f"xmem-{user_id[:16]}-{int(datetime.now(timezone.utc).timestamp())}"

    payload = {
        "amount": plan.amount,
        "currency": plan.currency,
        "receipt": receipt,
        "notes": {
            "user_id": user_id,
            "package_id": plan.id,
            "plan_name": plan.name,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                "https://api.razorpay.com/v1/orders",
                auth=(key_id, key_secret),
                json=payload,
            )
    except httpx.HTTPError as exc:
        logger.exception("Failed to create Razorpay order")
        raise HTTPException(status_code=502, detail="Failed to reach Razorpay") from exc

    if response.status_code >= 400:
        logger.warning("Razorpay order creation failed: %s %s", response.status_code, response.text[:500])
        raise HTTPException(status_code=502, detail="Razorpay order creation failed")

    order = response.json()
    order_id = order["id"]
    _billing_store.save_order(order_id, {
        "user_id": user_id,
        "package_id": plan.id,
        "amount": plan.amount,
        "currency": plan.currency,
        "receipt": receipt,
        "created_at": datetime.now(timezone.utc),
    })

    return RazorpayOrderResponse(
        id=order_id,
        order_id=order_id,
        amount=plan.amount,
        currency=plan.currency,
        key_id=key_id,
        receipt=receipt,
        package_id=plan.id,
    )


@router.post("/razorpay/verify", response_model=VerifyRazorpayPaymentResponse)
async def verify_razorpay_payment(
    request: VerifyRazorpayPaymentRequest,
    current_user: dict = Depends(require_auth),
) -> VerifyRazorpayPaymentResponse:
    """Verify a Razorpay checkout signature and activate the paid plan."""
    _, key_secret = _require_razorpay_config()
    plan = _get_plan(request.package_id)

    if not _verify_signature(
        request.razorpay_order_id,
        request.razorpay_payment_id,
        request.razorpay_signature,
        key_secret,
    ):
        raise HTTPException(status_code=400, detail="Invalid Razorpay signature")

    user_id = _user_id(current_user)
    tracked_order = _billing_store.get_order(request.razorpay_order_id)
    if tracked_order and tracked_order.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Payment order does not belong to this user")
    if tracked_order and tracked_order.get("package_id") != plan.id:
        raise HTTPException(status_code=400, detail="Payment order package mismatch")

    now = datetime.now(timezone.utc)
    summary = _get_summary(user_id)
    summary.plan_name = plan.name
    summary.account_status = "active"
    summary.currency = plan.currency
    summary.prepaid_balance_paise += plan.amount
    summary.next_invoice_paise = 0
    summary.last_payment_at = now
    summary.invoices.insert(
        0,
        Invoice(
            id=request.razorpay_payment_id,
            date=now,
            amount_paise=plan.amount,
            status="paid",
            credits=0,
        ),
    )
    _billing_store.save_summary(user_id, summary)

    return VerifyRazorpayPaymentResponse(status="ok", summary=summary)
