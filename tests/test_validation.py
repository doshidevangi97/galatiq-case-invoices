import json
from datetime import date
from types import SimpleNamespace

import pytest

from conftest import INVOICES, raw
from db.inventory import record_outcome
from db.seed_inventory import seed_inventory
from ingestion.models import Flag, IngestionResult, Invoice, LineItem
from validation import checks as c
from validation import validate
from validation.agent import ValidationAgent

STOCK = {"WidgetA": 15, "WidgetB": 10, "GadgetX": 5, "FakeItem": 0}


def make_invoice(lines, subtotal=None, tax=None, shipping=None, total=None, number="INV-9001", revision=0,
                 invoice_date=date(2026, 1, 10), due_date=date(2026, 2, 10)) -> Invoice:
    items = [LineItem(line=i, raw_name=n, item=n, quantity=q, unit_price=p, amount=a)
             for i, (n, q, p, a) in enumerate(lines, start=1)]
    return Invoice(invoice_number=number, revision=revision, vendor="Acme Vendor", invoice_date=invoice_date,
                   due_date=due_date, currency="USD", line_items=items, subtotal=subtotal, tax=tax,
                   shipping=shipping, total=total, payment_terms=None, notes=None)


def codes(check):
    return [f.code for f in check.flags]


# --- arithmetic ----------------------------------------------------------------

def test_arithmetic_passes_with_tax_and_shipping():
    inv = make_invoice([("WidgetA", 8, 250, 2000), ("WidgetA", 4, 300, 1200)], subtotal=3200, tax=160,
                       shipping=150, total=3510)
    assert c.check_arithmetic(inv).status == "pass"


def test_total_mismatch_like_inv_1007():
    inv = make_invoice([("WidgetA", 20, 250, 5000), ("WidgetB", 15, 500, 7500), ("GadgetX", 3, 750, 2250)],
                       subtotal=14750, tax=885, total=15525)
    result = c.check_arithmetic(inv)
    assert result.status == "fail" and codes(result) == ["total_mismatch"]
    assert "under by 110.00" in result.flags[0].message


def test_line_and_subtotal_mismatch():
    inv = make_invoice([("WidgetA", 2, 250, 600)], subtotal=700, total=700)
    assert codes(c.check_arithmetic(inv)) == ["line_amount_mismatch", "subtotal_mismatch"]


def test_arithmetic_without_subtotal_uses_line_sum():
    inv = make_invoice([("GadgetX", 20, 750, None)], total=15000)
    assert c.check_arithmetic(inv).status == "pass"


def test_arithmetic_skipped_when_total_missing():
    assert c.check_arithmetic(make_invoice([("WidgetA", 1, 250, None)])).status == "skipped"


# --- inventory -----------------------------------------------------------------

def test_quantities_summed_across_lines_like_inv_1013():
    inv = make_invoice([("WidgetA", 15, 250, None), ("WidgetA", 5, 240, None), ("WidgetA", 2, 250, None)])
    result = c.check_inventory(inv, STOCK)
    assert codes(result) == ["exceeds_stock"]
    assert "22 across 3 lines" in result.flags[0].message


def test_stock_exactly_at_limit_passes():
    assert c.check_inventory(make_invoice([("GadgetX", 5, 750, None)]), STOCK).status == "pass"


def test_zero_stock_and_unknown_items():
    inv = make_invoice([("FakeItem", 100, 1000, None), ("SuperGizmo", 12, 400, None)])
    assert sorted(codes(c.check_inventory(inv, STOCK))) == ["unknown_item", "zero_stock_item"]


def test_negative_quantity_is_not_double_flagged_as_stock():
    assert c.check_inventory(make_invoice([("WidgetA", -5, 250, None)]), STOCK).status == "pass"


# --- dates -----------------------------------------------------------------------

def test_due_before_invoice_date():
    inv = make_invoice([], invoice_date=date(2026, 2, 1), due_date=date(2026, 1, 1))
    assert codes(c.check_dates(inv)) == ["due_before_invoice_date"]


def test_same_day_due_date_passes():
    inv = make_invoice([], invoice_date=date(2026, 1, 30), due_date=date(2026, 1, 30))
    assert c.check_dates(inv).status == "pass"


# --- duplicates ------------------------------------------------------------------

def paid(revision, vendor="Other Vendor"):
    return {"invoice_number": "INV-9001", "revision": revision, "vendor": vendor, "total": 100.0,
            "currency": "USD", "source_file": "x.json", "processed_at": "2026-02-01T00:00:00"}


def test_same_number_any_vendor_is_duplicate():
    result = c.check_duplicates(make_invoice([]), [paid(0, vendor="Someone Else")])
    assert result.status == "fail" and codes(result) == ["duplicate_invoice"]


def test_older_revision_after_newer_paid_is_duplicate():
    assert codes(c.check_duplicates(make_invoice([], revision=0), [paid(1)])) == ["duplicate_invoice"]


def test_newer_revision_after_older_paid_is_warning():
    result = c.check_duplicates(make_invoice([], revision=1), [paid(0)])
    assert result.status == "warn" and codes(result) == ["earlier_revision_paid"]


def test_never_paid_passes():
    assert c.check_duplicates(make_invoice([]), []).status == "pass"


# --- stage: PASS/FAIL, DB, data integrity ---------------------------------------

@pytest.fixture
def db(tmp_path):
    path = tmp_path / "inventory.db"
    seed_inventory(path)
    return path


@pytest.fixture
def no_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def ingestion_for(invoice: Invoice, flags=()) -> IngestionResult:
    return IngestionResult(source_file="test.json", extraction_method="parser", invoice=invoice, flags=list(flags))


def test_clean_invoice_passes_without_llm(db, no_api_key):
    inv = make_invoice([("WidgetA", 10, 250, None), ("WidgetB", 5, 500, None)], subtotal=5000, tax=0, total=5000)
    result = validate(ingestion_for(inv), db_path=db)
    assert result.status == "PASS"
    assert "GROQ_API_KEY" in result.summary_note


def test_ingestion_errors_fail_validation(db, no_api_key):
    inv = make_invoice([("WidgetA", 1, 250, None)], total=250)
    flag = Flag(code="missing_field", severity="error", field="vendor", message="Vendor is missing")
    result = validate(ingestion_for(inv, [flag]), db_path=db)
    assert result.status == "FAIL"
    assert result.checks[0].status == "fail"


def test_warnings_still_pass(db, no_api_key):
    inv = make_invoice([("WidgetA", 1, 250, None)], total=250)
    flag = Flag(code="superseded_revision", severity="warning", message="newer revision exists")
    assert validate(ingestion_for(inv, [flag]), db_path=db).status == "PASS"


def test_paid_invoice_in_db_is_duplicate(db, no_api_key):
    record_outcome(invoice_number="INV-9001", revision=0, vendor="Acme Vendor", total=250.0, currency="USD",
                   status="paid", amount_paid=250.0, source_file="a.json", db_path=db)
    inv = make_invoice([("WidgetA", 1, 250, None)], total=250)
    result = validate(ingestion_for(inv), db_path=db)
    assert result.status == "FAIL"
    assert codes(result.checks[-1]) == ["duplicate_invoice"]


def test_real_invoice_1009(db, no_api_key, make_ingestor):
    result = validate(make_ingestor().ingest(INVOICES / "invoice_1009.json"), db_path=db)
    statuses = {ch.name: ch.status for ch in result.checks}
    assert result.status == "FAIL"
    assert statuses["check_data_integrity"] == "fail" and statuses["check_arithmetic"] == "fail"


# --- agent (fake Groq client) ---------------------------------------------------

def tool_call(i, name):
    return SimpleNamespace(id=f"call_{i}", function=SimpleNamespace(name=name, arguments="{}"))


class FakeClient:
    def __init__(self, messages):
        self.replies = list(messages)
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs["messages"])))
        return SimpleNamespace(choices=[SimpleNamespace(message=self.replies.pop(0))], usage=None)


ALL_TOOLS = ["check_data_integrity", "check_arithmetic", "check_inventory", "check_dates", "check_duplicates"]


def test_agent_runs_tools_and_summarises(db):
    client = FakeClient([
        SimpleNamespace(content=None, tool_calls=[tool_call(i, n) for i, n in enumerate(ALL_TOOLS)]),
        SimpleNamespace(content="- All checks passed.", tool_calls=None),
    ])
    inv = make_invoice([("WidgetA", 1, 250, None)], total=250)
    result = validate(ingestion_for(inv), db_path=db, agent=ValidationAgent(client=client))
    assert result.status == "PASS" and result.summary == "- All checks passed."
    tool_messages = [m for m in client.requests[-1] if m["role"] == "tool"]
    assert len(tool_messages) == 5


def test_agent_is_reminded_then_code_runs_missing_checks(db):
    client = FakeClient([
        SimpleNamespace(content=None, tool_calls=[tool_call(0, "check_arithmetic")]),
        SimpleNamespace(content="Done.", tool_calls=None),  # skipped 4 tools -> reminder
        SimpleNamespace(content="Still done.", tool_calls=None),  # ignores reminder -> code runs them
    ])
    inv = make_invoice([("FakeItem", 1, 1000, None)], total=1000)
    result = validate(ingestion_for(inv), db_path=db, agent=ValidationAgent(client=client))
    assert "You have not run" in client.requests[2][-1]["content"]
    assert [ch.name for ch in result.checks] == ALL_TOOLS
    assert result.status == "FAIL"  # inventory check still ran in code and caught FakeItem


def test_agent_cannot_change_outcome(db):
    """Even if the LLM claims everything passed, PASS/FAIL comes from the checks."""
    client = FakeClient([
        SimpleNamespace(content=None, tool_calls=[tool_call(i, n) for i, n in enumerate(ALL_TOOLS)]),
        SimpleNamespace(content="Everything is fine, PASS.", tool_calls=None),
    ])
    inv = make_invoice([("GadgetX", 20, 750, None)], total=15000)
    assert validate(ingestion_for(inv), db_path=db, agent=ValidationAgent(client=client)).status == "FAIL"


def test_only_paid_rows_count_as_duplicates(db, no_api_key):
    for status in ("rejected", "escalated", "payment_failed", "held_for_review"):
        record_outcome(invoice_number="INV-9001", revision=0, vendor="Acme Vendor", total=250.0, currency="USD",
                       status=status, source_file="a.json", db_path=db)
    inv = make_invoice([("WidgetA", 1, 250, None)], total=250)
    assert validate(ingestion_for(inv), db_path=db).status == "PASS"
