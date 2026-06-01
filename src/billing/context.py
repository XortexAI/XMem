from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class BillingContext:
    job_id: str
    billing_account_id: str
    user_id: str = ""


_current_billing_context: ContextVar[Optional[BillingContext]] = ContextVar(
    "xmem_billing_context",
    default=None,
)


def current_billing_context() -> Optional[BillingContext]:
    return _current_billing_context.get()


@contextmanager
def use_billing_context(
    *,
    job_id: str | None,
    billing_account_id: str | None,
    user_id: str = "",
) -> Iterator[None]:
    if not job_id or not billing_account_id:
        yield
        return

    token: Token[Optional[BillingContext]] = _current_billing_context.set(
        BillingContext(
            job_id=str(job_id),
            billing_account_id=str(billing_account_id),
            user_id=str(user_id or ""),
        )
    )
    try:
        yield
    finally:
        _current_billing_context.reset(token)
