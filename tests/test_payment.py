import json
import sqlite3
from datetime import date

import pytest

from approval.models import ApprovalResult, ApprovalRound, Claim, Critique, Decision
from db.inventory import find_processed, record_outcome, reset_ledger
from db.seed_inventory import seed_inventory
from ingestion.models import IngestionResult, Invoice, LineItem
from payment import process_payment
from validation.models import CheckResult, ValidationResult


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "inventory.db"
    seed_inventory(path)
    return path


def make(total=5000.0, revision=0, currency="USD", status="APPROVED", validation_status="PASS"):
    inv = Invoice(invoice_number="INV-9001", revision=revision, vendor="Acme Vendor", invoice_date=date(2026, 1, 1),
                  due_date=date(2026, 2, 1), currency=currency,
                  line_items=[LineItem(line=1, raw_name="WidgetA", item="WidgetA", quantity=1, unit_price=total,
                                       amount=None)],
                  subtotal=None, tax=None, shipping=None, total=total, payment_terms=None, notes=None)
    ingestion = IngestionResult(source_file="invoice_9001.json", extraction_method="parser", invoice=inv)
    check = CheckResult(name="check_inventory", label="Inventory",
                        status="pass" if validation_status == "PASS" else "fail", summary="ok")
    validation = ValidationResult(invoice_number="INV-9001", status=validation_status, checks=[check])
    decision = Decision(decision="APPROVE" if status == "APPROVED" else "REJECT", reasons=["because"])
    critique = Critique(verdict="AGREE", recommended_decision=decision.decision, verified=[Claim(statement="ok", evidence_type="validation_check", evidence="Inventory")], problems=[],
                        explanation="ok")
    approval = ApprovalResult(invoice_number="INV-9001", status=status, high_value=False,
                              rounds=[ApprovalRound(round=1, decision=decision, critique=critique,
                                                    critic_model="claude-opus-5-5")],
                              final_reasons=[f"reason for {status}"])
    return ingestion, validation, approval


class Bank:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0) if self.responses else {"status": "success"}
        if isinstance(item, Exception):
            raise item
        return item


def pay(db, tmp_path, *case, bank=None):
    bank = bank or Bank()
    result = process_payment(*case, run_id="run1", db_path=db, log_dir=tmp_path / "logs", pay=bank, retry_wait=0)
    return result, bank


def read_log(tmp_path, name):
    return [json.loads(line) for line in (tmp_path / "logs" / name).read_text().splitlines()]


def test_approved_invoice_is_paid_and_recorded(db, tmp_path):
    result, bank = pay(db, tmp_path, *make())
    assert result.status == "PAID" and result.amount_paid == 5000.0 and result.payment_kind == "full"
    assert bank.calls == [{"vendor": "Acme Vendor", "amount": 5000.0, "currency": "USD",
                           "invoice_number": "INV-9001", "reference": result.reference}]
    rows = find_processed("INV-9001", db)
    assert len(rows) == 1 and rows[0]["payment_reference"] == result.reference and rows[0]["run_id"] == "run1"


def test_rejected_invoice_is_logged_not_paid(db, tmp_path):
    result, bank = pay(db, tmp_path, *make(status="REJECTED", validation_status="FAIL"))
    assert result.status == "REJECTED" and bank.calls == []
    entry = read_log(tmp_path, "rejections.jsonl")[0]
    assert entry["reasons"] == ["reason for REJECTED"]
    assert entry["failed_checks"][0]["check"] == "Inventory"
    assert entry["critic_model"] == "claude-opus-5-5" and entry["critic_verdict"] == "AGREE"
    assert find_processed("INV-9001", db, status="rejected")[0]["reasons"] == '["reason for REJECTED"]'


def test_escalated_invoice_goes_to_human_review(db, tmp_path):
    result, bank = pay(db, tmp_path, *make(status="ESCALATED"))
    assert result.status == "PENDING_REVIEW" and bank.calls == []
    assert read_log(tmp_path, "human_review.jsonl")[0]["outcome"] == "escalated"
    assert not (tmp_path / "logs" / "rejections.jsonl").exists()


def test_already_paid_is_caught_at_payment_time(db, tmp_path):
    pay(db, tmp_path, *make())
    result, bank = pay(db, tmp_path, *make())
    assert result.status == "REJECTED" and bank.calls == []
    assert "already paid" in result.reasons[0]


def test_newer_revision_pays_only_the_difference(db, tmp_path):
    pay(db, tmp_path, *make(total=1890.0, revision=0))
    result, bank = pay(db, tmp_path, *make(total=5940.0, revision=1))
    assert result.status == "PAID" and result.payment_kind == "difference"
    assert result.amount_paid == 4050.0 and result.already_paid == 1890.0
    assert bank.calls[0]["amount"] == 4050.0


@pytest.mark.parametrize("new_total, text", [(1890.0, "nothing is owed"), (1500.0, "overpaid by 390.00")])
def test_no_balance_is_held_for_review(db, tmp_path, new_total, text):
    pay(db, tmp_path, *make(total=1890.0, revision=0))
    result, bank = pay(db, tmp_path, *make(total=new_total, revision=1))
    assert result.status == "PENDING_REVIEW" and bank.calls == []
    assert text in result.reasons[0]
    assert find_processed("INV-9001", db, status="held_for_review")


def test_payment_retries_then_succeeds(db, tmp_path):
    result, bank = pay(db, tmp_path, *make(), bank=Bank([ConnectionError("timeout"), {"status": "success"}]))
    assert result.status == "PAID" and len(result.attempts) == 2 and len(bank.calls) == 2


def test_payment_fails_after_three_attempts(db, tmp_path):
    result, bank = pay(db, tmp_path, *make(), bank=Bank([{"status": "declined"}] * 3))
    assert result.status == "PAYMENT_FAILED" and len(bank.calls) == 3
    assert find_processed("INV-9001", db) == []  # not marked paid, so it can be re-run
    assert find_processed("INV-9001", db, status="payment_failed")


def test_missing_currency_is_never_paid_even_if_approved(db, tmp_path):
    result, bank = pay(db, tmp_path, *make(currency=None))
    assert result.status == "PENDING_REVIEW" and bank.calls == []
    assert find_processed("INV-9001", db, status="held_for_review")
    assert read_log(tmp_path, "human_review.jsonl")[0]["outcome"] == "held_for_review"


def test_reset_ledger_keeps_inventory(db, tmp_path):
    pay(db, tmp_path, *make())
    pay(db, tmp_path, *make(status="REJECTED"))
    assert reset_ledger(db) == 2
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM inventory").fetchone()[0] == 4


def test_old_ledger_table_is_migrated(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE processed_invoices (invoice_number TEXT, revision INTEGER, vendor TEXT, total REAL,"
                     " currency TEXT, source_file TEXT, status TEXT, processed_at TEXT,"
                     " PRIMARY KEY (invoice_number, revision))")
    seed_inventory(path)
    record_outcome(invoice_number="INV-1", revision=0, vendor="v", total=1.0, currency=None, status="rejected",
                   source_file="f", db_path=path)
    record_outcome(invoice_number="INV-1", revision=0, vendor="v", total=1.0, currency=None, status="paid",
                   amount_paid=1.0, source_file="f", db_path=path)
    assert len(find_processed("INV-1", path)) == 1
