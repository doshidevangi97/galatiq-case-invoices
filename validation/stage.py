"""Validation stage: run all checks (via the LLM agent when available) and decide PASS/FAIL."""

from __future__ import annotations

import logging
from pathlib import Path

from db.inventory import find_processed, load_stock
from db.seed_inventory import DB_PATH
from ingestion.models import IngestionError, IngestionResult, LLMUsage
from llm_utils import has_api_key
from validation import checks as c
from validation.agent import ValidationAgent
from validation.models import ValidationResult

log = logging.getLogger("invoice.validation")


def validate(ingestion: IngestionResult, db_path: Path = DB_PATH,
             agent: ValidationAgent | None = None) -> ValidationResult:
    invoice = ingestion.invoice
    log.info("Validating %s", invoice.invoice_number or ingestion.source_file,
             extra={"data": {"invoice_number": invoice.invoice_number}})
    stock = load_stock(db_path)
    paid = find_processed(invoice.invoice_number, db_path) if invoice.invoice_number else []

    checks = {
        "check_data_integrity": lambda: c.check_data_integrity(ingestion),
        "check_arithmetic": lambda: c.check_arithmetic(invoice),
        "check_inventory": lambda: c.check_inventory(invoice, stock),
        "check_dates": lambda: c.check_dates(invoice),
        "check_duplicates": lambda: c.check_duplicates(invoice, paid),
    }

    summary, note = None, None
    if agent is None and not has_api_key():
        note = "No LLM summary: GROQ_API_KEY is not set (checks ran in code)"
        results = {name: run() for name, run in checks.items()}
    else:
        agent = agent or ValidationAgent()
        try:
            results, summary = agent.run(ingestion, checks)
            if summary is None:
                note = "The LLM agent did not return a summary"
        except IngestionError as exc:
            log.warning("Validation agent failed, running checks in code: %s", exc)
            note = f"No LLM summary: {exc}"
            results = {name: run() for name, run in checks.items()}

    ordered = [results[name] for name in checks]
    status = "FAIL" if any(r.status == "fail" for r in ordered) else "PASS"
    result = ValidationResult(
        invoice_number=invoice.invoice_number, status=status, checks=ordered, summary=summary, summary_note=note,
        llm_usage=agent.usage.model_copy(deep=True) if agent else LLMUsage(),
    )
    log.info("Validation %s: %d error(s), %d warning(s)", status, result.count("error"), result.count("warning"),
             extra={"data": {"invoice_number": invoice.invoice_number, "status": status,
                             "checks": {r.name: r.status for r in ordered},
                             "llm_usage": result.llm_usage.summary()}})
    return result
