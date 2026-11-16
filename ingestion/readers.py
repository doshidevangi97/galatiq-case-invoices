"""Turn an invoice file into text."""

from pathlib import Path

import pdfplumber

from ingestion.models import IngestionError

STRUCTURED_SUFFIXES = {".json", ".csv", ".xml"}
TEXT_SUFFIXES = {".txt", ".pdf"}
SUPPORTED_SUFFIXES = STRUCTURED_SUFFIXES | TEXT_SUFFIXES


def read_text(path: Path) -> str:
    try:
        if path.suffix.lower() == ".pdf":
            with pdfplumber.open(path) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # pdfplumber raises several unrelated types
        raise IngestionError(f"Could not read {path.name}: {exc}") from exc
