"""Data models for the ingestion stage.

RawInvoice is what an extractor (file parser or LLM) returns: loosely typed values
exactly as found. Invoice is the cleaned, typed result after normalisation.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["error", "warning", "info"]
RawValue = float | str | None


class IngestionError(Exception):
    """The file could not be ingested at all (unsupported, unreadable, LLM unavailable)."""


class Flag(BaseModel):
    code: str
    severity: Severity
    message: str
    field: str | None = None


# --- Raw extraction output -------------------------------------------------


class RawLineItem(BaseModel):
    name: str | None = None
    quantity: RawValue = None
    unit_price: RawValue = None
    amount: RawValue = None
    note: str | None = None


class RawInvoice(BaseModel):
    invoice_number: str | int | None = None
    revision: str | int | None = None
    vendor: str | None = None
    vendor_address: str | None = None
    invoice_date: str | None = None
    due_date: str | None = None
    currency: str | None = None
    line_items: list[RawLineItem] = Field(default_factory=list)
    subtotal: RawValue = None
    tax: RawValue = None
    shipping: RawValue = None
    total: RawValue = None
    payment_terms: str | None = None
    notes: str | None = None


# --- Normalised output ------------------------------------------------------


class LineItem(BaseModel):
    line: int
    raw_name: str | None
    item: str | None  # standardised name, e.g. "Widget a" -> "WidgetA"
    quantity: int | float | None
    unit_price: float | None
    amount: float | None
    note: str | None = None


class Invoice(BaseModel):
    invoice_number: str | None
    revision: int = 0  # 0 = original, 1 = R1, ...
    vendor: str | None
    vendor_address: str | None = None  # stored only; not judged or shown to agents (roadmap)
    invoice_date: date | None
    invoice_date_raw: str | None = None
    due_date: date | None
    due_date_raw: str | None = None
    currency: str | None
    line_items: list[LineItem]
    subtotal: float | None
    tax: float | None
    shipping: float | None
    total: float | None
    payment_terms: str | None
    notes: str | None

    def quantity_by_item(self) -> dict[str, int | float]:
        """Total quantity per item across all lines (items can repeat, e.g. INV-1013)."""
        totals: dict[str, int | float] = {}
        for li in self.line_items:
            if li.item and li.quantity is not None:
                totals[li.item] = totals.get(li.item, 0) + li.quantity
        return totals


class RelatedFile(BaseModel):
    path: str
    revision: int
    relation: Literal["newer_revision", "older_revision", "copy"]


class LLMCall(BaseModel):
    file: str
    attempt: int
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0  # part of completion_tokens, for reasoning models
    total_tokens: int = 0
    latency_ms: int = 0


class LLMUsage(BaseModel):
    calls: list[LLMCall] = Field(default_factory=list)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    def summary(self) -> dict:
        return {"calls": len(self.calls), "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "total_tokens": self.total_tokens}


class IngestionResult(BaseModel):
    source_file: str
    extraction_method: Literal["parser", "llm"]
    invoice: Invoice
    related_files: list[RelatedFile] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)
    llm_usage: LLMUsage = Field(default_factory=LLMUsage)

    def count(self, severity: Severity) -> int:
        return sum(1 for f in self.flags if f.severity == severity)
