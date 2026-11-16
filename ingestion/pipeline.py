"""Ingestion stage: file -> extracted, normalised invoice with data-quality flags."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ingestion.checks import run_checks
from ingestion.llm import GroqExtractor
from ingestion.models import Flag, IngestionError, IngestionResult, Invoice, LLMUsage, RelatedFile
from ingestion.normalize import build_catalog, to_invoice
from ingestion.parsers import parse_structured
from ingestion.readers import STRUCTURED_SUFFIXES, SUPPORTED_SUFFIXES, read_text
from ingestion.related import find_candidate_files, merge_copies

log = logging.getLogger("invoice.ingestion")


@dataclass
class ExtractedFile:
    path: Path
    method: str
    invoice: Invoice
    flags: list[Flag]


class Ingestor:
    def __init__(self, item_names: list[str], llm: GroqExtractor | None = None):
        self.catalog = build_catalog(item_names)
        self.llm = llm or GroqExtractor()
        self._cache: dict[Path, ExtractedFile] = {}

    def extract_file(self, path: Path) -> ExtractedFile:
        path = path.resolve()
        if path in self._cache:
            return self._cache[path]
        if not path.is_file():
            raise IngestionError(f"File not found: {path}")
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise IngestionError(f"Unsupported file type '{path.suffix}'. Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}")

        text = read_text(path)
        if not text.strip():
            raise IngestionError(f"{path.name} contains no readable text (a scanned PDF would need OCR)")

        flags: list[Flag] = []
        raw, method = None, "parser"
        if path.suffix.lower() in STRUCTURED_SUFFIXES:
            try:
                raw = parse_structured(path, text)
            except Exception as exc:  # malformed JSON/XML/CSV
                log.warning("Parser failed for %s: %s", path.name, exc, extra={"data": {"file": path.name}})
            if raw is None:
                flags.append(Flag(code="parser_fallback", severity="info",
                                  message=f"{path.suffix} file did not match a known layout; used the LLM instead"))
        if raw is None:
            method = "llm"
            raw, llm_flags = self.llm.extract(text, path.name)
            flags += llm_flags

        invoice, norm_flags = to_invoice(raw, self.catalog, path)
        log.info("Extracted %s via %s: %s, %d line items", path.name, method, invoice.invoice_number,
                 len(invoice.line_items),
                 extra={"data": {"file": path.name, "method": method, "invoice_number": invoice.invoice_number}})
        extracted = ExtractedFile(path, method, invoice, flags + norm_flags)
        self._cache[path] = extracted
        return extracted

    def ingest(self, path: Path) -> IngestionResult:
        log.info("Ingesting %s", path, extra={"data": {"file": str(path)}})
        primary = self.extract_file(path)
        invoice, flags = primary.invoice, list(primary.flags)
        related: list[RelatedFile] = []

        if invoice.invoice_number:
            copies: list[tuple[str, Invoice]] = []
            for candidate in find_candidate_files(primary.path, invoice.invoice_number):
                try:
                    other = self.extract_file(candidate)
                except IngestionError as exc:
                    flags.append(Flag(code="related_file_unreadable", severity="warning",
                                      message=f"Could not check {candidate.name} for the same invoice: {exc}"))
                    continue
                if other.invoice.invoice_number != invoice.invoice_number:
                    continue
                rev = other.invoice.revision
                relation = "copy" if rev == invoice.revision else "newer_revision" if rev > invoice.revision else "older_revision"
                related.append(RelatedFile(path=candidate.name, revision=rev, relation=relation))
                if relation == "copy":
                    copies.append((candidate.name, other.invoice))
            log.info("Found %d related file(s) for %s", len(related), invoice.invoice_number,
                     extra={"data": {"related": [r.model_dump() for r in related]}})

            newer = [r for r in related if r.relation == "newer_revision"]
            if newer:
                latest = max(newer, key=lambda r: r.revision)
                flags.append(Flag(code="superseded_revision", severity="warning", field="revision",
                                  message=f"This is revision R{invoice.revision}; a newer revision exists: "
                                          f"{latest.path} (R{latest.revision})"))
            older = [r.path for r in related if r.relation == "older_revision"]
            if older:
                flags.append(Flag(code="supersedes", severity="info", field="revision",
                                  message=f"Revision R{invoice.revision} supersedes: {', '.join(older)}"))
            if copies:
                invoice, merge_flags = merge_copies(invoice, copies)
                flags += merge_flags

        unreadable = {f.field for f in flags if f.code == "unparseable_number"}
        flags += [f for f in run_checks(invoice) if not (f.code == "missing_field" and f.field in unreadable)]

        usage = getattr(self.llm, "usage", None) or LLMUsage()
        result = IngestionResult(source_file=str(path), extraction_method=primary.method, invoice=invoice,
                                 related_files=related, flags=flags, llm_usage=usage.model_copy(deep=True))
        log.info("Ingestion finished: %d error(s), %d warning(s), %d LLM tokens", result.count("error"),
                 result.count("warning"), usage.total_tokens,
                 extra={"data": {"invoice_number": invoice.invoice_number,
                                 "flags": [f.code for f in flags if f.severity != "info"],
                                 "llm_usage": usage.summary()}})
        return result
