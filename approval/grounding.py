"""Check that the critic's claims are backed by evidence (gap #2).

Every point the critic makes must cite evidence that code can look up:
- document_quote:   the quoted text must appear in the original document (ignoring case, spacing, punctuation)
- extracted_field:  the field must exist in the extracted invoice (e.g. "total", "line_items[2].quantity")
- validation_check: the check must exist (e.g. "Inventory")
And every number in a claim must appear in the document, the extracted data or the validation results
(or be a line's quantity x unit price). Small counts (0-10) are allowed, e.g. "all 5 checks".

This catches a critic that invents facts. It cannot catch one that cites real facts but reasons wrongly;
the code rules and the two-model agreement cover that.
"""

from __future__ import annotations

import re

from approval.models import HIGH_VALUE_THRESHOLD, Claim, Critique
from ingestion.llm import _numbers_in
from ingestion.models import Invoice
from ingestion.normalize import fix_ocr_digits, item_key
from validation.models import ValidationResult

SMALL_NUMBER = 10


def _compact(text: str) -> str:
    return item_key(fix_ocr_digits(text))


def allowed_numbers(document: str, case: str, invoice: Invoice) -> set[float]:
    """Numbers the critic may legitimately mention."""
    numbers = (_numbers_in(document) | _numbers_in(case) | {HIGH_VALUE_THRESHOLD}
               | _numbers_in(invoice.model_dump_json(exclude={"vendor_address"})))
    line_sum = 0.0
    for li in invoice.line_items:
        if li.quantity is not None and li.unit_price is not None:
            product = round(abs(li.quantity * li.unit_price), 2)
            numbers.add(product)
            line_sum += product
    base = invoice.subtotal if invoice.subtotal is not None else line_sum
    numbers |= {round(line_sum, 2), round(base + (invoice.tax or 0), 2),
                round(base + (invoice.tax or 0) + (invoice.shipping or 0), 2)}
    return numbers


def _field_exists(path: str, invoice: dict) -> bool:
    path = path.strip().strip("'\"`").removeprefix("invoice.")
    if match := re.fullmatch(r"line_items\[(\d+)\](?:\.(\w+))?", path):
        index, key = int(match.group(1)), match.group(2)
        items = invoice.get("line_items", [])
        if not 1 <= index <= len(items):
            return False
        return key is None or key in items[index - 1]
    return path in invoice


def _check_claim(claim: Claim, kind: str, compact_doc: str, invoice: dict, check_names: set[str],
                 numbers: set[float]) -> list[str]:
    label = f'{kind} "{claim.statement[:80]}"'
    issues = []
    if claim.evidence_type == "document_quote":
        if not claim.evidence.strip() or _compact(claim.evidence) not in compact_doc:
            issues.append(f'{label}: quoted text not found in the document: "{claim.evidence[:80]}"')
    elif claim.evidence_type == "extracted_field":
        if not _field_exists(claim.evidence, invoice):
            issues.append(f"{label}: no extracted field named '{claim.evidence}'")
    elif claim.evidence_type == "validation_check":
        name = claim.evidence.strip().lower().removesuffix(" check").strip()
        if name not in check_names:
            issues.append(f"{label}: no validation check named '{claim.evidence}'")
    for number in sorted(_numbers_in(claim.statement)):
        if number > SMALL_NUMBER and number not in numbers:
            issues.append(f"{label}: the number {number:g} is not in the document, extracted data or checks")
    return issues


def unverified_claims(critique: Critique, document: str, case: str, invoice: Invoice,
                      validation: ValidationResult) -> list[str]:
    """Problems with the critique's evidence; empty means every claim checked out."""
    compact_doc = _compact(document)
    invoice_dict = invoice.model_dump(mode="json", exclude={"vendor_address"})
    check_names = {c.label.lower() for c in validation.checks} | {c.name.lower() for c in validation.checks}
    numbers = allowed_numbers(document, case, invoice)
    issues = []
    for kind, claims in (("verified", critique.verified), ("problem", critique.problems)):
        for claim in claims:
            issues += _check_claim(claim, kind, compact_doc, invoice_dict, check_names, numbers)
    return issues
