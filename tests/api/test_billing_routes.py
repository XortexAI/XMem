from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from src.api.routes import billing
from src.billing.types import VerifyPaymentRequest


@pytest.mark.asyncio
async def test_subscription_verify_checks_signature_before_checkout_lookup(monkeypatch):
    class Store:
        def get_checkout(self, checkout_id):
            raise AssertionError("checkout lookup should not run before signature verification")

    class Service:
        store = Store()

    monkeypatch.setattr(billing, "get_default_billing_service", lambda: Service())
    monkeypatch.setattr(billing, "verify_subscription_signature", lambda *args: False)

    with pytest.raises(billing.HTTPException) as exc:
        await billing.verify_razorpay_payment(
            VerifyPaymentRequest(
                package_id="pro",
                razorpay_payment_id="pay_1",
                razorpay_signature="bad",
                razorpay_subscription_id="sub_1",
            ),
            current_user={"id": "user-1"},
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "Invalid Razorpay signature"
