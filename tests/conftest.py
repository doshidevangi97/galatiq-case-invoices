from pathlib import Path

import pytest

from ingestion import Ingestor
from ingestion.models import RawInvoice, RawLineItem

INVOICES = Path(__file__).resolve().parent.parent / "data" / "invoices"
CATALOG = ["WidgetA", "WidgetB", "GadgetX", "FakeItem"]


class FakeLLM:
    """Stands in for GroqExtractor: returns canned extractions keyed by file name."""

    def __init__(self, responses: dict[str, RawInvoice]):
        self.responses = responses
        self.calls: list[str] = []

    def extract(self, text: str, source_name: str):
        self.calls.append(source_name)
        return self.responses[source_name], []


def raw(invoice_number, vendor, items, total, **extra) -> RawInvoice:
    return RawInvoice(
        invoice_number=invoice_number, vendor=vendor, total=total,
        line_items=[RawLineItem(name=n, quantity=q, unit_price=p) for n, q, p in items], **extra,
    )


@pytest.fixture
def make_ingestor():
    def _make(responses: dict[str, RawInvoice] | None = None) -> Ingestor:
        return Ingestor(item_names=CATALOG, llm=FakeLLM(responses or {}))
    return _make
