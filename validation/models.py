from typing import Literal

from pydantic import BaseModel, Field

from ingestion.models import Flag, LLMUsage

CheckStatus = Literal["pass", "fail", "warn", "skipped"]


class CheckResult(BaseModel):
    name: str  # tool name, e.g. "check_arithmetic"
    label: str  # human label, e.g. "Arithmetic"
    status: CheckStatus
    summary: str
    flags: list[Flag] = Field(default_factory=list)


class ValidationResult(BaseModel):
    invoice_number: str | None
    status: Literal["PASS", "FAIL"]
    checks: list[CheckResult]
    summary: str | None = None  # plain-language summary written by the LLM agent
    summary_note: str | None = None  # why there is no summary, if there isn't one
    llm_usage: LLMUsage = Field(default_factory=LLMUsage)

    @property
    def flags(self) -> list[Flag]:
        return [f for c in self.checks for f in c.flags]

    def count(self, severity: str) -> int:
        return sum(1 for f in self.flags if f.severity == severity)
