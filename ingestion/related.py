"""Find other files for the same invoice (revisions and copies) and merge copies."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from ingestion.models import Flag, IngestionError, Invoice
from ingestion.normalize import fix_ocr_digits, item_key
from ingestion.readers import SUPPORTED_SUFFIXES, read_text

# (field, matching raw field). Copies may fill gaps in these; conflicts are flagged.
_COMPARED_FIELDS = [
    ("vendor", None), ("invoice_date", "invoice_date_raw"), ("due_date", "due_date_raw"),
    ("currency", None), ("subtotal", None), ("tax", None), ("shipping", None), ("total", None),
    ("payment_terms", None),
]
_FILL_ONLY_FIELDS = [("notes", None)]  # free text; wording legitimately differs between formats


def invoice_id_pattern(invoice_number: str) -> re.Pattern | None:
    digits = re.search(r"\d+", invoice_number)
    if not digits:
        return None
    return re.compile(rf"\bINV[\s#:\-]*0*{int(digits.group())}\b", re.I)


def find_candidate_files(primary: Path, invoice_number: str) -> list[Path]:
    """Files in the same folder whose text mentions the same invoice number."""
    pattern = invoice_id_pattern(invoice_number)
    if pattern is None:
        return []
    candidates = []
    for path in sorted(primary.parent.iterdir()):
        if path.resolve() == primary.resolve() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            if pattern.search(fix_ocr_digits(read_text(path))):
                candidates.append(path)
        except IngestionError:
            continue
    return candidates


def _same(a: object, b: object) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 0.005
    if isinstance(a, date) or isinstance(b, date):
        return a == b
    return item_key(str(a)) == item_key(str(b))


def _items_signature(invoice: Invoice) -> list[tuple]:
    return sorted((li.item or "", li.quantity or 0, li.unit_price or 0) for li in invoice.line_items)


def merge_copies(primary: Invoice, copies: list[tuple[str, Invoice]]) -> tuple[Invoice, list[Flag]]:
    """Fill fields missing from the primary file using same-revision copies; flag disagreements."""
    merged = primary.model_copy(deep=True)
    flags: list[Flag] = []
    for name, copy in copies:
        for (field, raw_field), compare in [(f, True) for f in _COMPARED_FIELDS] + [(f, False) for f in _FILL_ONLY_FIELDS]:
            mine, theirs = getattr(merged, field), getattr(copy, field)
            if theirs is None:
                continue
            if mine is None:
                if raw_field and getattr(merged, raw_field):
                    continue  # present but unreadable here; don't hide that behind the copy's value
                setattr(merged, field, theirs)
                if raw_field:
                    setattr(merged, raw_field, getattr(copy, raw_field))
                flags.append(Flag(code="filled_from_copy", severity="info", field=field,
                                  message=f"{field} was missing; filled from {name}: {theirs}"))
            elif compare and not _same(mine, theirs):
                flags.append(Flag(code="copy_conflict", severity="warning", field=field,
                                  message=f"{field} differs from copy {name}: {mine!r} here vs {theirs!r} there"))
        if not merged.line_items and copy.line_items:
            merged.line_items = [li.model_copy() for li in copy.line_items]
            flags.append(Flag(code="filled_from_copy", severity="info", field="line_items",
                              message=f"Line items were missing; filled from {name}"))
        elif copy.line_items and _items_signature(merged) != _items_signature(copy):
            flags.append(Flag(code="copy_conflict", severity="warning", field="line_items",
                              message=f"Line items differ from copy {name}"))
    return merged, flags
