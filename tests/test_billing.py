import os
from datetime import timedelta

os.environ.setdefault("APP_STORE_PROVIDER", "memory")
os.environ.setdefault("FALLBACK_ORDER", '["ollama"]')
os.environ.setdefault("NEO4J_PASSWORD", "test")

import pytest

from src.billing import store as billing_store
from src.billing.service import BillingService
from src.billing.store import BillingStore, BillingStoreError, InsufficientCredits, utc_now
from src.utils import billing as billing_config


@pytest.fixture(autouse=True)
def clear_memory_billing():
    billing_store._memory_accounts.clear()
    billing_store._memory_wallets.clear()
    billing_store._memory_lots.clear()
    billing_store._memory_ledger.clear()
    billing_store._memory_reservations.clear()
    billing_store._memory_usage_events.clear()
    billing_store._memory_checkouts.clear()
    billing_store._memory_payment_events.clear()
    billing_store._memory_payment_records.clear()


def service() -> BillingService:
    return BillingService(BillingStore())


def test_token_estimate_uses_configured_chars_per_token():
    assert billing_config.estimate_tokens("a" * 9) == 3


def test_pro_nominal_credit_value():
    assert billing_config.nominal_paise_per_credit("pro") == pytest.approx(1.98)


def test_free_trial_grant_is_idempotent():
    svc = service()
    user = {"id": "user_1"}

    first = svc.ensure_billing_account(user)
    second = svc.ensure_billing_account(user)

    assert first["id"] == second["id"]
    summary = svc.get_billing_summary(user)
    assert summary.available_credits == 10_000
    grants = [entry for entry in svc.list_ledger(user) if entry["type"] == "grant"]
    assert len(grants) == 1


def test_reservation_debit_and_refund_flow():
    svc = service()
    user = {"id": "user_1"}
    account = svc.ensure_billing_account(user)

    reservation = svc.reserve_credits(account["id"], "job_1", 1000)
    assert reservation.available_credits == 9000

    svc.commit_job_debit(account["id"], "job_1", 750)
    summary = svc.get_billing_summary(user)

    assert summary.available_credits == 9250
    assert summary.reserved_credits == 0
    ledger_types = {entry["type"] for entry in svc.list_ledger(user)}
    assert {"reserve", "debit", "refund"}.issubset(ledger_types)


def test_reused_reservation_is_marked_not_created():
    svc = service()
    user = {"id": "user_1"}
    account = svc.ensure_billing_account(user)

    first = svc.reserve_credits(account["id"], "job_1", 1000)
    second = svc.reserve_credits(account["id"], "job_1", 1000)

    summary = svc.get_billing_summary(user)
    assert first.created
    assert not second.created
    assert summary.available_credits == 9000
    assert summary.reserved_credits == 1000


def test_failed_job_releases_reserved_credits():
    svc = service()
    user = {"id": "user_1"}
    account = svc.ensure_billing_account(user)

    svc.reserve_credits(account["id"], "job_1", 1000)
    svc.release_job_reservation(account["id"], "job_1")

    summary = svc.get_billing_summary(user)
    assert summary.available_credits == 10_000
    assert summary.reserved_credits == 0


def test_duplicate_release_does_not_double_refund():
    svc = service()
    user = {"id": "user_1"}
    account = svc.ensure_billing_account(user)

    svc.reserve_credits(account["id"], "job_1", 1000)
    svc.release_job_reservation(account["id"], "job_1")
    svc.release_job_reservation(account["id"], "job_1")

    summary = svc.get_billing_summary(user)
    releases = [entry for entry in svc.list_ledger(user) if entry["type"] == "release"]
    assert summary.available_credits == 10_000
    assert summary.reserved_credits == 0
    assert len(releases) == 1


def test_insufficient_credits_blocks_reservation():
    svc = service()
    user = {"id": "user_1"}
    account = svc.ensure_billing_account(user)

    with pytest.raises(InsufficientCredits):
        svc.reserve_credits(account["id"], "job_1", 10_001)


def test_reservation_cannot_be_reused_by_another_account():
    svc = service()
    account_1 = svc.ensure_billing_account({"id": "user_1"})
    account_2 = svc.ensure_billing_account({"id": "user_2"})

    svc.reserve_credits(account_1["id"], "job_1", 100)

    with pytest.raises(BillingStoreError):
        svc.reserve_credits(account_2["id"], "job_1", 100)


def test_duplicate_commit_does_not_double_debit():
    svc = service()
    user = {"id": "user_1"}
    account = svc.ensure_billing_account(user)
    svc.reserve_credits(account["id"], "job_1", 1000)

    svc.commit_job_debit(account["id"], "job_1", 750)
    svc.commit_job_debit(account["id"], "job_1", 750)

    summary = svc.get_billing_summary(user)
    debits = [entry for entry in svc.list_ledger(user) if entry["type"] == "debit"]
    assert summary.available_credits == 9250
    assert len(debits) == 1


def test_pro_grant_is_idempotent_per_payment():
    svc = service()

    svc.grant_pro_subscription(
        user_id="user_1",
        payment_id="pay_1",
        subscription_id="sub_1",
        amount=300,
        currency="USD",
        billing_region="GLOBAL",
        period_end=utc_now() + timedelta(days=30),
    )
    svc.grant_pro_subscription(
        user_id="user_1",
        payment_id="pay_1",
        subscription_id="sub_1",
        amount=300,
        currency="USD",
        billing_region="GLOBAL",
        period_end=utc_now() + timedelta(days=30),
    )

    summary = svc.get_billing_summary({"id": "user_1"})
    assert summary.available_credits == 15_000
    pro_grants = [
        entry
        for entry in svc.list_ledger({"id": "user_1"})
        if entry["source"] == "pro_monthly"
    ]
    assert len(pro_grants) == 1
    assert len(summary.invoices) == 1
    assert summary.invoices[0].id == "pay_1"
    assert summary.invoices[0].amount_paise == 300
    assert summary.invoices[0].currency == "USD"
    assert summary.invoices[0].credits == 5_000


def test_topup_grant_adds_dashboard_invoice():
    svc = service()

    svc.grant_topup(
        user_id="user_1",
        pack_id="topup_99",
        payment_id="pay_topup_1",
        order_id="order_1",
        amount=9_900,
        currency="INR",
        billing_region="IN",
    )

    summary = svc.get_billing_summary({"id": "user_1"})
    assert summary.credit_balance == 15_000
    assert len(summary.invoices) == 1
    invoice = summary.invoices[0]
    assert invoice.id == "pay_topup_1"
    assert invoice.amount_paise == 9_900
    assert invoice.currency == "INR"
    assert invoice.credits == 5_000
    assert invoice.status == "paid"


def test_billing_config_changes_affect_estimates(monkeypatch):
    svc = service()
    payload = {"user_query": "a" * 400, "agent_response": "", "effort_level": "low"}
    baseline = svc.estimate_required_credits("memory_ingest", payload)

    monkeypatch.setitem(billing_config.WORKFLOW_MULTIPLIERS, "memory_ingest_low", 2.0)
    changed = svc.estimate_required_credits("memory_ingest", payload)

    assert changed.billable_credits == baseline.billable_credits * 2


def test_missing_payment_event_id_is_rejected():
    store = BillingStore()

    with pytest.raises(ValueError):
        store.mark_payment_event("", {"event": "payment.captured"})


def test_in_memory_checkout_and_webhook_events_are_isolated():
    store = BillingStore()

    store.save_checkout("same_id", {"user_id": "user_1", "package_id": "topup_99"})

    assert store.mark_payment_event("same_id", {"event": "payment.captured"})
    assert not store.mark_payment_event("same_id", {"event": "payment.captured"})
    assert store.get_checkout("same_id")["package_id"] == "topup_99"
