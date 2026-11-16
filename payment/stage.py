"""Payment stage.

APPROVED  -> re-check the ledger, work out the amount (full, or the difference if an older revision was
             already paid), call the mock bank with retries, record the outcome.
REJECTED  -> no payment; record in the ledger and logs/rejections.jsonl with full reasoning.
ESCALATED -> no payment; record in the ledger and logs/human_review.jsonl for a human VP.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from approval.models import ApprovalResult
from db.inventory import find_processed, record_outcome
from db.seed_inventory import DB_PATH
from ingestion.models import IngestionResult
from observability import LOG_DIR
from payment.bank import mock_payment
from payment.models import PaymentAttempt, PaymentResult
from validation.models import ValidationResult

log = logging.getLogger("invoice.payment")

MAX_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 1.0
TOLERANCE = 0.005


def _audit_record(ingestion: IngestionResult, validation: ValidationResult, approval: ApprovalResult,
                  run_id: str | None, reasons: list[str], outcome: str) -> dict:
    inv = ingestion.invoice
    last = approval.rounds[-1] if approval.rounds else None
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "outcome": outcome,
        "invoice_number": inv.invoice_number,
        "revision": inv.revision,
        "vendor": inv.vendor,
        "total": inv.total,
        "currency": inv.currency,
        "source_file": ingestion.source_file,
        "reasons": reasons,
        "validation_status": validation.status,
        "failed_checks": [{"check": c.label, "details": [f.message for f in c.flags if f.severity == "error"]}
                          for c in validation.checks if c.status == "fail"],
        "approval_status": approval.status,
        "approval_rounds": len(approval.rounds),
        "agent_decision": last.decision.decision if last else None,
        "agent_reasons": last.decision.reasons if last else [],
        "critic_model": last.critic_model if last else None,
        "critic_verdict": last.critique.verdict if last and last.critique else None,
        "critic_problems": [c.statement for c in last.critique.problems] if last and last.critique else [],
        "critic_unverified": last.critique_unverified if last else [],
        "approval_notes": approval.notes,
    }


def _append(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _pay_with_retries(pay: Callable[..., dict], wait: float, **kwargs) -> tuple[bool, list[PaymentAttempt]]:
    attempts: list[PaymentAttempt] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = pay(**kwargs)
            attempts.append(PaymentAttempt(attempt=attempt, response=response))
            if isinstance(response, dict) and response.get("status") == "success":
                return True, attempts
            log.warning("Payment attempt %d returned %s", attempt, response)
        except Exception as exc:  # the bank API could fail in any way
            attempts.append(PaymentAttempt(attempt=attempt, error=f"{type(exc).__name__}: {exc}"))
            log.warning("Payment attempt %d raised %s", attempt, exc)
        if attempt < MAX_ATTEMPTS:
            time.sleep(wait * attempt)
    return False, attempts


def process_payment(ingestion: IngestionResult, validation: ValidationResult, approval: ApprovalResult,
                    run_id: str | None = None, db_path: Path = DB_PATH, log_dir: Path = LOG_DIR,
                    pay: Callable[..., dict] = mock_payment, retry_wait: float = RETRY_WAIT_SECONDS) -> PaymentResult:
    inv = ingestion.invoice
    common = dict(invoice_number=inv.invoice_number, revision=inv.revision, vendor=inv.vendor, total=inv.total,
                  currency=inv.currency, source_file=ingestion.source_file, run_id=run_id, db_path=db_path)

    def not_paid(status: str, ledger_status: str, log_name: str, message: str, reasons: list[str],
                 **extra) -> PaymentResult:
        row = record_outcome(status=ledger_status, reasons=reasons, **common)
        path = log_dir / log_name
        _append(path, _audit_record(ingestion, validation, approval, run_id, reasons, ledger_status) | extra)
        log.info("%s: %s", inv.invoice_number, message,
                 extra={"data": {"invoice_number": inv.invoice_number, "status": ledger_status, "reasons": reasons}})
        return PaymentResult(invoice_number=inv.invoice_number, status=status, message=message, reasons=reasons,
                             currency=inv.currency, invoice_total=inv.total, ledger_row_id=row, log_file=str(path),
                             **{k: v for k, v in extra.items() if k in PaymentResult.model_fields})

    if approval.status == "REJECTED":
        return not_paid("REJECTED", "rejected", "rejections.jsonl", "Rejected - no payment made",
                        approval.final_reasons)
    if approval.status == "ESCALATED":
        return not_paid("PENDING_REVIEW", "escalated", "human_review.jsonl",
                        "Escalated - held for human VP review, no payment made", approval.final_reasons)

    # APPROVED: final safety checks before money moves.
    if not inv.invoice_number or not inv.vendor or inv.total is None or inv.total <= 0:
        return not_paid("REJECTED", "rejected", "rejections.jsonl", "Rejected at payment - cannot pay",
                        ["Approved invoice is missing an invoice number, vendor or positive total"])
    if inv.currency is None:  # approval already escalates these; this is a second line of defence
        return not_paid("PENDING_REVIEW", "held_for_review", "human_review.jsonl",
                        "Held for human review - currency not stated, no payment made",
                        ["Currency not stated on the invoice; a payment cannot be made without it"])

    paid_rows = find_processed(inv.invoice_number, db_path)
    same_or_newer = [r for r in paid_rows if r["revision"] >= inv.revision]
    if same_or_newer:
        r = same_or_newer[0]
        return not_paid("REJECTED", "rejected", "rejections.jsonl", "Rejected at payment - already paid",
                        [f"{inv.invoice_number} R{r['revision']} was already paid ({r['amount_paid']} "
                         f"{r['currency'] or ''}, ref {r['payment_reference']}) on {r['processed_at'][:10]}"])

    already_paid = round(sum(r["amount_paid"] or 0 for r in paid_rows), 2)
    amount, kind = inv.total, "full"
    if paid_rows:
        amount, kind = round(inv.total - already_paid, 2), "difference"
        if amount <= TOLERANCE:
            balance = "nothing is owed" if abs(amount) <= TOLERANCE else f"overpaid by {abs(amount):,.2f}"
            return not_paid("PENDING_REVIEW", "held_for_review", "human_review.jsonl",
                            "Held for human review - no balance to pay",
                            [f"R{inv.revision} total {inv.total:,.2f} but {already_paid:,.2f} already paid for "
                             f"earlier revisions: {balance} (possible refund/credit)"],
                            already_paid=already_paid)

    reference = f"PAY-{inv.invoice_number}-R{inv.revision}-{uuid.uuid4().hex[:8].upper()}"
    log.info("Paying %s %s to %s for %s (%s)", f"{amount:,.2f}", inv.currency or "(currency not stated)", inv.vendor,
             inv.invoice_number, kind,
             extra={"data": {"invoice_number": inv.invoice_number, "amount": amount, "currency": inv.currency,
                             "kind": kind, "already_paid": already_paid, "reference": reference}})
    ok, attempts = _pay_with_retries(pay, retry_wait, vendor=inv.vendor, amount=amount, currency=inv.currency,
                                     invoice_number=inv.invoice_number, reference=reference)
    result = PaymentResult(invoice_number=inv.invoice_number, status="PAID" if ok else "PAYMENT_FAILED",
                           message="", currency=inv.currency, invoice_total=inv.total, already_paid=already_paid,
                           payment_kind=kind, reference=reference, attempts=attempts)
    if ok:
        result.amount_paid = amount
        result.message = (f"Paid {amount:,.2f}" + (f" (difference after {already_paid:,.2f} already paid)"
                                                   if kind == "difference" else ""))
        result.ledger_row_id = record_outcome(status="paid", amount_paid=amount, payment_reference=reference,
                                              reasons=approval.final_reasons, **common)
    else:
        last = attempts[-1]
        result.message = f"Payment failed after {len(attempts)} attempts: {last.error or last.response}"
        result.reasons = [result.message]
        result.ledger_row_id = record_outcome(status="payment_failed", payment_reference=reference,
                                              reasons=result.reasons, **common)
    log.info("%s: %s", inv.invoice_number, result.message,
             extra={"data": {"invoice_number": inv.invoice_number, "status": result.status, "reference": reference,
                             "attempts": len(attempts)}})
    return result
