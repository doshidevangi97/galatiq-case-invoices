"""Clean raw extracted values: numbers, dates, invoice IDs, revisions and item names."""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

from ingestion.models import Flag, Invoice, LineItem, RawInvoice

# A letter O used where a zero belongs, e.g. "2O26" or "$3,500.O0".
_OCR_ZERO = re.compile(r"(?<=[\d.,])[oO]|[oO](?=[\d.,]*\d)")

_DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d",
    "%m/%d/%Y", "%m-%d-%Y",  # numeric day/month order assumed US (MM/DD)
    "%d-%b-%Y", "%d-%B-%Y", "%d %b %Y", "%d %B %Y",
    "%b %d %Y", "%B %d %Y", "%b %d, %Y", "%B %d, %Y",
]


def fix_ocr_digits(text: str) -> str:
    return _OCR_ZERO.sub("0", text)


def clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def item_key(name: str) -> str:
    """Comparison key that ignores case, spaces and punctuation: 'Widget a' -> 'widgeta'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def parse_number(value: object) -> tuple[float | None, bool]:
    """Return (number, ok). ok is False when a value was present but unreadable."""
    if value is None:
        return None, True
    if isinstance(value, bool):
        return None, False
    if isinstance(value, (int, float)):
        return round(float(value), 2), True
    text = fix_ocr_digits(str(value)).strip()
    if not text:
        return None, True
    negative = text.startswith("(") and text.endswith(")")  # accounting style (250.00)
    text = re.sub(r"[^\d.\-]", "", text)
    try:
        number = float(text)
    except ValueError:
        return None, False
    return round(-number if negative else number, 2), True


def parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    text = fix_ocr_digits(raw).strip().rstrip(".")
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def normalize_invoice_number(value: object) -> str | None:
    """'INV 1012', '1002', 'INV-1004' -> 'INV-1012', 'INV-1002', 'INV-1004'."""
    text = clean_text(value)
    if text is None:
        return None
    digits = re.search(r"\d+", fix_ocr_digits(text))
    return f"INV-{digits.group()}" if digits else text.upper()


def parse_revision(value: object) -> int:
    """'R1' -> 1, 'Rev 2' -> 2, 'Revised' -> 1, missing -> 0 (original)."""
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if match := re.search(r"\d+", text):
        return int(match.group())
    return 1 if "revis" in text else 0


def revision_from_filename(stem: str) -> int:
    """'invoice_1004_revised' -> 1, 'invoice_1004_rev2' -> 2, 'invoice_1004' -> 0."""
    if match := re.search(r"(?:^|[_\-\s.])(?:revised|revision|rev|r)[_\-\s]?(\d+)(?:$|[_\-\s.])", stem, re.I):
        return int(match.group(1))
    return 1 if re.search(r"revised|revision", stem, re.I) else 0


def build_catalog(item_names: list[str]) -> dict[str, str]:
    return {item_key(name): name for name in item_names}


def standardize_item(raw_name: str | None, catalog: dict[str, str]) -> tuple[str | None, str | None]:
    """Return (standard name, note split off from the name).

    Matches the inventory catalog exactly after ignoring case/spacing, so 'widget a' and
    'WidgetA' both become 'WidgetA', but 'WidgetC' is never loosely matched to 'WidgetA'.
    Unknown items get a consistent CamelCase form ('super gizmo' -> 'SuperGizmo').
    """
    name = clean_text(raw_name)
    if name is None:
        return None, None
    note = None
    if (match := re.match(r"^(.*?)\s*[(\[](.*?)[)\]]\s*$", name)) and match.group(1):
        name, note = match.group(1).strip(), match.group(2).strip() or None
    if (key := item_key(name)) in catalog:
        return catalog[key], note
    tokens = [t for t in re.split(r"[\s_\-]+", name) if t]
    return "".join(t[:1].upper() + t[1:] for t in tokens), note


def to_invoice(raw: RawInvoice, catalog: dict[str, str], source: Path) -> tuple[Invoice, list[Flag]]:
    flags: list[Flag] = []

    def number(value: object, field: str) -> float | None:
        parsed, ok = parse_number(value)
        if not ok:
            flags.append(Flag(code="unparseable_number", severity="error", field=field,
                              message=f"Could not read {field} value {value!r} as a number"))
        return parsed

    items: list[LineItem] = []
    for i, raw_item in enumerate(raw.line_items, start=1):
        raw_name = clean_text(raw_item.name)
        name, name_note = standardize_item(raw_name, catalog)
        if raw_name and name and raw_name != name:
            flags.append(Flag(code="item_name_standardized", severity="info", field=f"line_items[{i}].item",
                              message=f"Line {i}: '{raw_name}' -> '{name}'"))
        quantity = number(raw_item.quantity, f"line_items[{i}].quantity")
        if quantity is not None and float(quantity).is_integer():
            quantity = int(quantity)
        note = "; ".join(n for n in (name_note, clean_text(raw_item.note)) if n) or None
        items.append(LineItem(
            line=i, raw_name=raw_name, item=name, quantity=quantity,
            unit_price=number(raw_item.unit_price, f"line_items[{i}].unit_price"),
            amount=number(raw_item.amount, f"line_items[{i}].amount"),
            note=note,
        ))

    invoice_date_raw = clean_text(raw.invoice_date)
    due_date_raw = clean_text(raw.due_date)
    currency = clean_text(raw.currency)
    invoice = Invoice(
        invoice_number=normalize_invoice_number(raw.invoice_number),
        revision=max(parse_revision(raw.revision), revision_from_filename(source.stem)),
        vendor=clean_text(raw.vendor),
        vendor_address=clean_text(raw.vendor_address),
        invoice_date=parse_date(invoice_date_raw),
        invoice_date_raw=invoice_date_raw,
        due_date=parse_date(due_date_raw),
        due_date_raw=due_date_raw,
        currency=currency.upper() if currency else None,
        line_items=items,
        subtotal=number(raw.subtotal, "subtotal"),
        tax=number(raw.tax, "tax"),
        shipping=number(raw.shipping, "shipping"),
        total=number(raw.total, "total"),
        payment_terms=clean_text(raw.payment_terms),
        notes=clean_text(raw.notes),
    )
    return invoice, flags
