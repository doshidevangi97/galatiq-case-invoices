"""Deterministic parsers for structured formats (JSON, CSV, XML).

Exact, fast and free of hallucination. If a file doesn't fit the expected shapes,
parse_structured returns None and the pipeline falls back to the LLM extractor.
"""

from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from ingestion.models import RawInvoice, RawLineItem

FIELD_ALIASES = {
    "invoice_number": ["invoice_number", "invoice_no", "invoice_num", "invoice_id", "inv_no", "invoice"],
    "revision": ["revision", "revision_number", "rev", "version"],
    "vendor": ["vendor", "vendor_name", "supplier", "seller"],
    "vendor_address": ["vendor_address", "supplier_address", "address"],
    "invoice_date": ["invoice_date", "date", "issue_date"],
    "due_date": ["due_date", "due", "payment_due"],
    "currency": ["currency"],
    "subtotal": ["subtotal", "sub_total"],
    "tax": ["tax_amount", "tax", "sales_tax"],
    "shipping": ["shipping", "shipping_amount", "freight"],
    "total": ["total", "total_amount", "grand_total", "amount_due", "total_due"],
    "payment_terms": ["payment_terms", "terms"],
    "notes": ["notes", "note", "memo"],
}
ITEM_ALIASES = {
    "name": ["item", "name", "item_name", "description", "product"],
    "quantity": ["quantity", "qty"],
    "unit_price": ["unit_price", "price", "rate"],
    "amount": ["amount", "line_total", "line_amount", "total"],
    "note": ["note", "notes"],
}
ITEMS_CONTAINER_KEYS = {"line_items", "items", "lines"}
# Keys that belong to a line item inside a key/value CSV ("total" is excluded: it's the invoice total there).
_KV_ITEM_FIELD_KEYS = {k for f in ("quantity", "unit_price", "note") for k in ITEM_ALIASES[f]} | {"amount", "line_total", "line_amount"}
_TABULAR_ITEM_COLUMNS = {k for keys in ITEM_ALIASES.values() for k in keys} - {"total"}


def _key(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def _pick(data: dict, aliases: list[str]):
    for alias in aliases:
        if alias in data:
            return data[alias]
    return None


def _build(flat: dict, items: list[dict]) -> RawInvoice | None:
    flat = {_key(k): v for k, v in flat.items()}
    values = {field: _pick(flat, aliases) for field, aliases in FIELD_ALIASES.items()}
    if isinstance(values["vendor"], dict):
        vendor = values["vendor"]
        values["vendor"] = vendor.get("name")
        values["vendor_address"] = values["vendor_address"] or vendor.get("address")
    line_items = []
    for item in items:
        item = {_key(k): v for k, v in item.items()}
        line_items.append(RawLineItem(**{f: _pick(item, aliases) for f, aliases in ITEM_ALIASES.items()}))
    invoice = RawInvoice(**values, line_items=line_items)
    if not (invoice.invoice_number or invoice.vendor or invoice.line_items):
        return None  # nothing recognisable; let the LLM try
    return invoice


def _parse_json(text: str) -> RawInvoice | None:
    data = json.loads(text)
    if not isinstance(data, dict):
        return None
    container = next((k for k in data if _key(k) in ITEMS_CONTAINER_KEYS), None)
    items = (data.get(container) or []) if container else []
    flat = {k: v for k, v in data.items() if k != container}
    return _build(flat, [i for i in items if isinstance(i, dict)])


def _parse_xml(text: str) -> RawInvoice | None:
    root = ET.fromstring(text)
    container = next((el for el in root.iter() if _key(el.tag) in ITEMS_CONTAINER_KEYS), None)
    inside = set(container.iter()) if container is not None else set()
    items = [{child.tag: child.text for child in item} for item in container] if container is not None else []
    flat: dict = {}
    for el in root.iter():
        if len(el) == 0 and el not in inside:
            flat.setdefault(el.tag, el.text)
    return _build(flat, items)


def _summary_field(label: str) -> str:
    key = _key(label)
    if key.startswith("sub"):
        return "subtotal"
    if key.startswith("tax") or key.startswith("sales_tax"):
        return "tax"
    if key.startswith(("ship", "freight")):
        return "shipping"
    if "total" in key or key.startswith("amount_due"):
        return "total"
    return key


def _parse_csv(text: str) -> RawInvoice | None:
    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    if not rows:
        return None
    header = [_key(c) for c in rows[0]]
    flat: dict = {}
    items: list[dict] = []

    if header[:2] == ["field", "value"]:
        # Key/value layout (INV-1006): one field per row, items as repeated item/quantity/price rows.
        current: dict | None = None
        for row in rows[1:]:
            key, value = _key(row[0]), (row[1] if len(row) > 1 else None)
            if key in ITEM_ALIASES["name"]:
                current = {"name": value}
                items.append(current)
            elif current is not None and key in _KV_ITEM_FIELD_KEYS and key not in current:
                current[key] = value
            else:
                current = None
                flat.setdefault(key, value)
        return _build(flat, items)

    # Tabular layout (INV-1007): one row per line item, summary rows below with blank leading cells.
    item_cols = [i for i, h in enumerate(header) if h in _TABULAR_ITEM_COLUMNS]
    name_col = next((i for i, h in enumerate(header) if h in ITEM_ALIASES["name"]), None)
    for row in rows[1:]:
        cells = [c.strip() for c in row] + [""] * (len(header) - len(row))
        if name_col is not None and cells[name_col]:
            items.append({header[i]: cells[i] for i in item_cols})
            for i, h in enumerate(header):
                if i not in item_cols and cells[i]:
                    flat.setdefault(h, cells[i])
            continue
        filled = [c for c in cells if c]
        label = next((c for c in filled if re.search(r"[A-Za-z]", c)), None)
        if label and filled.index(label) + 1 < len(filled):
            flat.setdefault(_summary_field(label), filled[filled.index(label) + 1])
    return _build(flat, items)


_PARSERS = {".json": _parse_json, ".xml": _parse_xml, ".csv": _parse_csv}


def parse_structured(path: Path, text: str) -> RawInvoice | None:
    """Parse a JSON/CSV/XML invoice. Raises on malformed files; returns None on unknown shapes."""
    return _PARSERS[path.suffix.lower()](text)
