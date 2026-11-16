from typing import Literal

from pydantic import BaseModel, Field

PaymentStatus = Literal["PAID", "PAYMENT_FAILED", "REJECTED", "PENDING_REVIEW"]


class PaymentAttempt(BaseModel):
    attempt: int
    response: dict | None = None
    error: str | None = None


class PaymentResult(BaseModel):
    invoice_number: str | None
    status: PaymentStatus
    message: str
    reasons: list[str] = Field(default_factory=list)
    amount_paid: float | None = None
    currency: str | None = None
    invoice_total: float | None = None
    already_paid: float = 0.0  # paid earlier for older revisions of this invoice
    payment_kind: Literal["full", "difference"] | None = None
    reference: str | None = None
    attempts: list[PaymentAttempt] = Field(default_factory=list)
    ledger_row_id: int | None = None
    log_file: str | None = None  # where the rejection / review record was written
