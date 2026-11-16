import shutil

from conftest import INVOICES, raw

from ingestion.models import RawInvoice
from ingestion.llm import GroqExtractor, grounding_problems


def codes(result, severity=None):
    return {f.code for f in result.flags if severity is None or f.severity == severity}


# --- structured formats (no LLM) -------------------------------------------

def test_clean_json(make_ingestor):
    result = make_ingestor().ingest(INVOICES / "invoice_1016.json")
    inv = result.invoice
    assert result.extraction_method == "parser"
    assert (inv.invoice_number, inv.vendor, inv.total, inv.currency) == ("INV-1016", "Widgets Inc.", 3233.0, "USD")
    assert [li.item for li in inv.line_items] == ["WidgetA", "WidgetB", "WidgetC"]
    assert codes(result, "error") == set()


def test_key_value_csv(make_ingestor):
    inv = make_ingestor().ingest(INVOICES / "invoice_1006.csv").invoice
    assert [(li.item, li.quantity, li.unit_price) for li in inv.line_items] == [("WidgetA", 5, 250.0), ("WidgetB", 3, 500.0)]
    assert inv.total == 2750.0


def test_tabular_csv_summary_rows(make_ingestor):
    inv = make_ingestor().ingest(INVOICES / "invoice_1007.csv").invoice
    assert (inv.subtotal, inv.tax, inv.total) == (14750.0, 885.0, 15525.0)
    assert len(inv.line_items) == 3


def test_xml(make_ingestor):
    inv = make_ingestor().ingest(INVOICES / "invoice_1014.xml").invoice
    assert (inv.invoice_number, inv.vendor, inv.currency, inv.total) == ("INV-1014", "TechParts International", "EUR", 4125.0)


def test_invalid_data_is_flagged(make_ingestor):
    result = make_ingestor().ingest(INVOICES / "invoice_1009.json")
    fields = {(f.code, f.field) for f in result.flags}
    assert ("missing_field", "vendor") in fields  # empty string counts as missing
    assert ("missing_field", "due_date") in fields
    assert ("negative_value", "line_items[1].quantity") in fields
    assert ("negative_value", "total") in fields


def test_repeated_items_are_totalled(make_ingestor):
    responses = {"invoice_1013.pdf": raw("INV-1013", "Atlas Industrial Supply", [], 22562.80)}
    inv = make_ingestor(responses).ingest(INVOICES / "invoice_1013.json").invoice
    assert inv.quantity_by_item() == {"WidgetA": 22, "WidgetB": 18, "GadgetX": 9}


# --- revisions ---------------------------------------------------------------

def test_original_is_flagged_as_superseded(make_ingestor):
    result = make_ingestor().ingest(INVOICES / "invoice_1004.json")
    assert result.invoice.revision == 0
    assert "superseded_revision" in codes(result, "warning")
    assert [(r.path, r.relation) for r in result.related_files] == [("invoice_1004_revised.json", "newer_revision")]


def test_revised_invoice_supersedes_original(make_ingestor):
    result = make_ingestor().ingest(INVOICES / "invoice_1004_revised.json")
    assert result.invoice.revision == 1
    assert "supersedes" in codes(result, "info")
    assert "superseded_revision" not in codes(result)


def test_revision_detected_from_txt_filename(make_ingestor, tmp_path):
    for name in ("invoice_2001.txt", "invoice_2001_rev2.txt"):
        (tmp_path / name).write_text("Invoice: INV-2001")
    responses = {n: raw("INV-2001", "Acme", [("WidgetA", 1, 250)], 250, due_date="2026-02-01")
                 for n in ("invoice_2001.txt", "invoice_2001_rev2.txt")}
    result = make_ingestor(responses).ingest(tmp_path / "invoice_2001.txt")
    assert "superseded_revision" in codes(result)
    assert result.related_files[0].revision == 2


# --- copies in other formats ---------------------------------------------------

def test_copy_fills_missing_fields(make_ingestor, tmp_path):
    for name in ("invoice_1011.txt", "invoice_1011.pdf"):
        shutil.copy(INVOICES / name, tmp_path / name)
    items = [("WidgetA", 6, 250), ("WidgetB", 3, 500)]
    responses = {
        "invoice_1011.pdf": raw("INV-1011", "Summit Manufacturing Co.", items, 3000, due_date="2026-02-20"),
        "invoice_1011.txt": raw("INV-1011", "Summit Manufacturing Co.", items, 3000, due_date="2026-02-20",
                                subtotal=3000, tax=0, payment_terms="Net 30"),
    }
    result = make_ingestor(responses).ingest(tmp_path / "invoice_1011.pdf")
    assert (result.invoice.subtotal, result.invoice.payment_terms) == (3000, "Net 30")
    assert "filled_from_copy" in codes(result, "info")
    assert "copy_conflict" not in codes(result)


def test_copy_conflict_is_flagged(make_ingestor, tmp_path):
    for name in ("invoice_1011.txt", "invoice_1011.pdf"):
        shutil.copy(INVOICES / name, tmp_path / name)
    items = [("WidgetA", 6, 250), ("WidgetB", 3, 500)]
    responses = {
        "invoice_1011.txt": raw("INV-1011", "Summit Manufacturing Co.", items, 3000, due_date="2026-02-20"),
        "invoice_1011.pdf": raw("INV-1011", "Summit Manufacturing Co.", items, 3900, due_date="2026-02-20"),
    }
    result = make_ingestor(responses).ingest(tmp_path / "invoice_1011.txt")
    assert any(f.code == "copy_conflict" and f.field == "total" for f in result.flags)


# --- LLM-extracted invoices: normalisation and flags --------------------------

def test_messy_text_invoice_is_normalised(make_ingestor):
    responses = {"invoice_1012.txt": raw("INV 1012", "QuickShip Distributers",
                                         [("Widget A", 12, 250), ("WidgetB", 7, 500), ("Gadget X", 4, 750)], 9975,
                                         invoice_date="26-Jan-2O26", due_date="25-Feb-2026")}
    responses["invoice_1012.pdf"] = responses["invoice_1012.txt"]
    result = make_ingestor(responses).ingest(INVOICES / "invoice_1012.txt")
    inv = result.invoice
    assert inv.invoice_number == "INV-1012"
    assert str(inv.invoice_date) == "2026-01-26"
    assert [li.item for li in inv.line_items] == ["WidgetA", "WidgetB", "GadgetX"]
    assert [r.relation for r in result.related_files] == ["copy"]


def test_unparseable_due_date_and_zero_values(make_ingestor, tmp_path):
    (tmp_path / "invoice_3001.txt").write_text("x")
    responses = {"invoice_3001.txt": raw("INV-3001", "Fraudster LLC", [("FakeItem", 0, 1000)], 0, due_date="yesterday")}
    result = make_ingestor(responses).ingest(tmp_path / "invoice_3001.txt")
    found = {(f.code, f.field) for f in result.flags}
    assert ("unparseable_date", "due_date") in found
    assert ("zero_value", "line_items[1].quantity") in found
    assert ("zero_value", "total") in found


# --- LLM self-correction loop ------------------------------------------------

DOC = "Vendor: QuickShip Distributers\nWidget A  12  $250  $3,000.00\nTOTAL: $3,000.00"


def test_grounding_catches_corrected_spelling_and_invented_numbers():
    good = raw("1", "QuickShip Distributers", [("Widget A", 12, 250)], 3000)
    bad = raw("1", "QuickShip Distributors", [("Widget A", 12, 250)], 3500)
    assert grounding_problems(good, DOC) == []
    assert len(grounding_problems(bad, DOC)) == 2


class _FakeCompletions:
    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def create(self, **kwargs):
        self.seen.append(kwargs["messages"])
        content = self.replies.pop(0)
        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": content})})]})


def _fake_client(replies):
    completions = _FakeCompletions(replies)
    return type("Client", (), {"chat": type("Chat", (), {"completions": completions})})(), completions


def test_llm_retries_until_grounded():
    bad = raw("1", "QuickShip Distributors", [("Widget A", 12, 250)], 3000).model_dump_json()
    good = raw("1", "QuickShip Distributers", [("Widget A", 12, 250)], 3000).model_dump_json()
    client, completions = _fake_client(["not json", bad, good])
    result, flags = GroqExtractor(client=client).extract(DOC, "doc.txt")
    assert result.vendor == "QuickShip Distributers"
    assert [f.code for f in flags] == ["llm_self_corrected"]
    feedback = [m["content"] for m in completions.seen[-1] if m["role"] == "user"]
    assert any("not valid for the required JSON shape" in m for m in feedback)
    assert any('"QuickShip Distributors" does not appear' in m for m in feedback)


def test_llm_gives_up_with_warning_flags():
    bad = RawInvoice(vendor="Made Up Corp").model_dump_json()
    client, _ = _fake_client([bad] * 3)
    result, flags = GroqExtractor(client=client).extract(DOC, "doc.txt")
    assert flags and all(f.code == "ungrounded_value" for f in flags)


def test_currency_flags(make_ingestor):
    def currency_flags(name):
        return [(f.code, f.severity) for f in make_ingestor().ingest(INVOICES / name).flags if f.field == "currency"]

    assert currency_flags("invoice_1006.csv") == [("missing_currency", "warning")]
    assert currency_flags("invoice_1014.xml") == [("non_usd_currency", "warning")]
    assert currency_flags("invoice_1016.json") == []
