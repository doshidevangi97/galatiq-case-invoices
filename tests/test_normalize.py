from datetime import date

import pytest

from ingestion.normalize import (
    build_catalog, normalize_invoice_number, parse_date, parse_number,
    parse_revision, revision_from_filename, standardize_item,
)

CATALOG = build_catalog(["WidgetA", "WidgetB", "GadgetX", "FakeItem"])


@pytest.mark.parametrize("raw", ["Widget a", "Widget A", "WidgetA", "widgeta", "WIDGET-A", "  widget   a "])
def test_item_name_variants_map_to_catalog(raw):
    assert standardize_item(raw, CATALOG) == ("WidgetA", None)


def test_item_parenthetical_becomes_note():
    assert standardize_item("WidgetA (rush order)", CATALOG) == ("WidgetA", "rush order")


@pytest.mark.parametrize("raw, expected", [("WidgetC", "WidgetC"), ("widget c", "WidgetC"),
                                           ("SuperGizmo", "SuperGizmo"), ("Gadget X", "GadgetX")])
def test_unknown_items_are_not_fuzzy_matched(raw, expected):
    assert standardize_item(raw, CATALOG)[0] == expected


@pytest.mark.parametrize("raw, expected", [
    ("2026-01-30", date(2026, 1, 30)), ("Jan 30 2026", date(2026, 1, 30)),
    ("01/28/2026", date(2026, 1, 28)), ("January 27, 2026", date(2026, 1, 27)),
    ("26-Jan-2O26", date(2026, 1, 26)), ("25-Feb-2026", date(2026, 2, 25)),
])
def test_parse_date_formats(raw, expected):
    assert parse_date(raw) == expected


@pytest.mark.parametrize("raw", ["yesterday", "ASAP", "2026-13-45"])
def test_unparseable_dates(raw):
    assert parse_date(raw) is None


@pytest.mark.parametrize("raw, expected", [
    ("$3,500.O0", (3500.0, True)), ("$15,000.00", (15000.0, True)), (-5, (-5.0, True)),
    ("(250.00)", (-250.0, True)), (None, (None, True)), ("", (None, True)), ("n/a", (None, False)),
])
def test_parse_number(raw, expected):
    assert parse_number(raw) == expected


@pytest.mark.parametrize("raw", ["INV-1012", "INV 1012", "1012", "inv#1012"])
def test_invoice_number(raw):
    assert normalize_invoice_number(raw) == "INV-1012"


def test_revisions():
    assert [parse_revision(v) for v in (None, "R1", "Rev 2", "Revised", 3)] == [0, 1, 2, 1, 3]
    assert revision_from_filename("invoice_1004_revised") == 1
    assert revision_from_filename("invoice_1004_rev2") == 2
    assert revision_from_filename("invoice_1004") == 0
