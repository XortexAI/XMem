from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class PlanPublic(BaseModel):
    id: str
    name: str
    price_paise: int
    currency: str = "INR"
    monthly_credits: int = 0
    trial_credits: int = 0
    trial_days: int = 0
    nominal_paise_per_credit: float = 0.0


class TopUpPackPublic(BaseModel):
    id: str
    price_paise: int
    currency: str = "INR"
    credits: int


class CreditLotPublic(BaseModel):
    id: str
    source: str
    remaining_credits: int
    expires_at: Optional[datetime] = None


class BillingSummary(BaseModel):
    billing_account_id: str
    owner_type: str = "user"
    owner_id: str
    plan_id: str
    plan_name: str
    status: str
    currency: str = "INR"
    available_credits: int = 0
    reserved_credits: int = 0
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    credit_lots: list[CreditLotPublic] = Field(default_factory=list)


class CreditEstimate(BaseModel):
    job_type: str
    content_tokens: int
    multiplier: float
    billable_credits: int
    reserved_credits: int


class ReservationResult(BaseModel):
    reservation_id: str
    billing_account_id: str
    job_id: str
    reserved_credits: int
    status: Literal["active", "committed", "released", "expired"]
    available_credits: int
    created: bool = False


class CheckoutRequest(BaseModel):
    package_id: str = Field(..., description="Plan ID or top-up pack ID")


class CheckoutResponse(BaseModel):
    id: str
    package_id: str
    amount: int
    currency: str
    key_id: str
    order_id: Optional[str] = None
    subscription_id: Optional[str] = None
    receipt: Optional[str] = None


class VerifyPaymentRequest(BaseModel):
    package_id: str
    razorpay_payment_id: str
    razorpay_signature: str
    razorpay_order_id: Optional[str] = None
    razorpay_subscription_id: Optional[str] = None


class LedgerEntryPublic(BaseModel):
    id: str
    type: str
    amount: int
    idempotency_key: str
    job_id: Optional[str] = None
    source: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
