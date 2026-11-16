"""Data-quality flags raised at ingestion: missing values, negatives, zeros, unreadable dates.

Ingestion never rejects an invoice; it records flags for the validation and approval stages.
"""

from ingestion.models import Flag, Invoice


def run_checks(invoice: Invoice) -> list[Flag]:
    flags: list[Flag] = []

    def add(code: str, severity: str, field: str, message: str) -> None:
        flags.append(Flag(code=code, severity=severity, field=field, message=message))

    for field, label in (("invoice_number", "Invoice number"), ("vendor", "Vendor"), ("total", "Total amount")):
        if getattr(invoice, field) is None:
            add("missing_field", "error", field, f"{label} is missing")

    for field, label, severity in (("invoice_date", "Invoice date", "warning"), ("due_date", "Due date", "error")):
        raw = getattr(invoice, f"{field}_raw")
        if getattr(invoice, field) is None:
            if raw:
                add("unparseable_date", "error", field, f"{label} '{raw}' is not a real date")
            else:
                add("missing_field", severity, field, f"{label} is missing")

    if invoice.currency is None:
        add("missing_currency", "warning", "currency",
            "Currency is not stated on the invoice; it will be sent for human review at approval (not paid)")
    elif invoice.currency != "USD":
        add("non_usd_currency", "warning", "currency",
            f"Invoice is in {invoice.currency}, not USD; it will be sent for human review at approval")

    if not invoice.line_items:
        add("missing_field", "error", "line_items", "No line items found")

    for li in invoice.line_items:
        prefix, label = f"line_items[{li.line}]", f"Line {li.line} ({li.raw_name or 'unnamed'})"
        if li.item is None:
            add("missing_field", "error", f"{prefix}.item", f"Line {li.line}: item name is missing")
        if li.quantity is None:
            add("missing_field", "error", f"{prefix}.quantity", f"{label}: quantity is missing")
        if li.unit_price is None:
            add("missing_field", "warning", f"{prefix}.unit_price", f"{label}: unit price is missing")
        for field in ("quantity", "unit_price", "amount"):
            value = getattr(li, field)
            if value is not None and value < 0:
                add("negative_value", "error", f"{prefix}.{field}", f"{label}: {field.replace('_', ' ')} is negative ({value})")
        if li.quantity == 0:
            add("zero_value", "error", f"{prefix}.quantity", f"{label}: quantity is zero")

    for field, label in (("subtotal", "Subtotal"), ("tax", "Tax"), ("shipping", "Shipping"), ("total", "Total amount")):
        value = getattr(invoice, field)
        if value is not None and value < 0:
            add("negative_value", "error", field, f"{label} is negative ({value})")
    if invoice.total == 0:
        add("zero_value", "error", "total", "Total amount is zero")

    return flags
