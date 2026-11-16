"""Deterministic validation checks. These are the source of truth; the LLM agent only calls them."""

from __future__ import annotations

from ingestion.models import Flag, IngestionResult, Invoice
from validation.models import CheckResult

TOLERANCE = 0.01  # rounding tolerance for money comparisons


def _money(value: float) -> str:
    return f"{value:,.2f}"


def _result(name: str, label: str, flags: list[Flag], ok_summary: str, skipped: str | None = None) -> CheckResult:
    if any(f.severity == "error" for f in flags):
        status = "fail"
    elif any(f.severity == "warning" for f in flags):
        status = "warn"
    elif skipped:
        status = "skipped"
    else:
        status = "pass"
    summary = "; ".join(f.message for f in flags if f.severity != "info") or skipped or ok_summary
    return CheckResult(name=name, label=label, status=status, summary=summary, flags=flags)


def check_data_integrity(ingestion: IngestionResult) -> CheckResult:
    """Carry forward problems found while reading the invoice (missing, negative, zero, unreadable values)."""
    carried = [f.model_copy() for f in ingestion.flags if f.severity in ("error", "warning")]
    return _result("check_data_integrity", "Data integrity", carried,
                   "All required fields present; no negative, zero or unreadable values")


def check_arithmetic(invoice: Invoice) -> CheckResult:
    """quantity x unit price per line, lines vs subtotal, subtotal + tax + shipping vs total."""
    flags: list[Flag] = []
    lines_total, complete = 0.0, bool(invoice.line_items)

    for li in invoice.line_items:
        if li.quantity is None or li.unit_price is None:
            complete = False
            continue
        expected = round(li.quantity * li.unit_price, 2)
        lines_total += expected
        if li.amount is not None and abs(expected - li.amount) > TOLERANCE:
            flags.append(Flag(code="line_amount_mismatch", severity="error", field=f"line_items[{li.line}].amount",
                              message=f"Line {li.line} ({li.item}): {li.quantity} × {_money(li.unit_price)} = "
                                      f"{_money(expected)}, but the line amount says {_money(li.amount)}"))
    lines_total = round(lines_total, 2)

    if complete and invoice.subtotal is not None and abs(lines_total - invoice.subtotal) > TOLERANCE:
        flags.append(Flag(code="subtotal_mismatch", severity="error", field="subtotal",
                          message=f"Line items add up to {_money(lines_total)}, but the subtotal says "
                                  f"{_money(invoice.subtotal)}"))

    base = invoice.subtotal if invoice.subtotal is not None else (lines_total if complete else None)
    if invoice.total is None or base is None:
        reason = "total is missing" if invoice.total is None else "some lines have no quantity or unit price"
        return _result("check_arithmetic", "Arithmetic", flags, "", skipped=f"Cannot verify the total: {reason}")

    tax, shipping = invoice.tax or 0, invoice.shipping or 0
    expected_total = round(base + tax + shipping, 2)
    parts = f"{'subtotal' if invoice.subtotal is not None else 'line items'} {_money(base)}"
    parts += f" + tax {_money(tax)}" if invoice.tax is not None else ""
    parts += f" + shipping {_money(shipping)}" if invoice.shipping is not None else ""
    if abs(expected_total - invoice.total) > TOLERANCE:
        diff = invoice.total - expected_total
        flags.append(Flag(code="total_mismatch", severity="error", field="total",
                          message=f"{parts} = {_money(expected_total)}, but the total says {_money(invoice.total)} "
                                  f"({'over' if diff > 0 else 'under'} by {_money(abs(diff))})"))
    return _result("check_arithmetic", "Arithmetic", flags, f"{parts} = total {_money(invoice.total)}")


def check_inventory(invoice: Invoice, stock: dict[str, int]) -> CheckResult:
    """Each item must exist in inventory, have stock, and not exceed it (quantities summed across lines)."""
    stock_by_key = {name.lower(): (name, qty) for name, qty in stock.items()}
    flags: list[Flag] = []
    lines_per_item: dict[str, int] = {}
    for li in invoice.line_items:
        if li.item:
            lines_per_item[li.item] = lines_per_item.get(li.item, 0) + 1

    checked = []
    for item, qty in invoice.quantity_by_item().items():
        across = f" across {lines_per_item[item]} lines" if lines_per_item.get(item, 0) > 1 else ""
        if item.lower() not in stock_by_key:
            flags.append(Flag(code="unknown_item", severity="error", field="line_items",
                              message=f"{item} is not in the inventory database"))
            continue
        _, available = stock_by_key[item.lower()]
        if available == 0:
            flags.append(Flag(code="zero_stock_item", severity="error", field="line_items",
                              message=f"{item} has zero stock (out of stock or suspicious item); invoice bills {qty}{across}"))
        elif qty <= 0:
            continue  # negative/zero quantities are already flagged by data integrity
        elif qty > available:
            flags.append(Flag(code="exceeds_stock", severity="error", field="line_items",
                              message=f"{item}: invoice bills {qty}{across}, but only {available} in stock"))
        checked.append(f"{item} {qty}/{available}")
    return _result("check_inventory", "Inventory", flags,
                   "All items in stock: " + ", ".join(checked) if checked else "No items to check")


def check_dates(invoice: Invoice) -> CheckResult:
    flags: list[Flag] = []
    if invoice.invoice_date is None or invoice.due_date is None:
        return _result("check_dates", "Dates", flags, "", skipped="Cannot compare dates: invoice or due date missing")
    if invoice.due_date < invoice.invoice_date:
        flags.append(Flag(code="due_before_invoice_date", severity="error", field="due_date",
                          message=f"Due date {invoice.due_date} is before the invoice date {invoice.invoice_date}"))
    return _result("check_dates", "Dates", flags,
                   f"Due {invoice.due_date}, {(invoice.due_date - invoice.invoice_date).days} days after invoice date")


def check_duplicates(invoice: Invoice, paid: list[dict]) -> CheckResult:
    """Compare against invoices already paid with the same invoice number (any vendor)."""
    if not invoice.invoice_number:
        return _result("check_duplicates", "Duplicates", [], "", skipped="Cannot check: no invoice number")
    flags: list[Flag] = []
    rev = invoice.revision

    def describe(row: dict) -> str:
        total = f"{row['total']:,.2f} {row['currency'] or ''}".strip() if row["total"] is not None else "unknown total"
        return (f"R{row['revision']} from {row['vendor'] or 'unknown vendor'} ({total}) was paid "
                f"on {row['processed_at'][:10]} via {row['source_file']}")

    same = [r for r in paid if r["revision"] == rev]
    newer = [r for r in paid if r["revision"] > rev]
    older = [r for r in paid if r["revision"] < rev]
    for row in same:
        flags.append(Flag(code="duplicate_invoice", severity="error", field="invoice_number",
                          message=f"Duplicate: {invoice.invoice_number} {describe(row)}"))
    for row in newer:
        flags.append(Flag(code="duplicate_invoice", severity="error", field="invoice_number",
                          message=f"Stale revision: a newer {invoice.invoice_number} {describe(row)}"))
    if older and not (same or newer):
        for row in older:
            flags.append(Flag(code="earlier_revision_paid", severity="warning", field="invoice_number",
                              message=f"Review: earlier {invoice.invoice_number} {describe(row)}; "
                                      f"this is R{rev}, so only the difference may be owed"))
    return _result("check_duplicates", "Duplicates", flags, f"{invoice.invoice_number} has not been paid before")
