from typing import Literal

from pydantic import BaseModel, Field

from ingestion.models import LLMUsage

HIGH_VALUE_THRESHOLD = 10_000.0


class Decision(BaseModel):
    """What the main approval agent returns."""

    decision: Literal["APPROVE", "REJECT"]
    reasons: list[str] = Field(description="Specific, checkable reasons citing invoice facts or validation results")
    scrutiny_findings: list[str] = Field(
        default_factory=list,
        description="Invoices over $10K only: findings for each extra-scrutiny area, including 'none found'",
    )
    response_to_critic: str | None = Field(
        default=None, description="On a revision round: how the critique was addressed")


class Claim(BaseModel):
    """One point in a critique, with evidence that code can check."""

    statement: str
    evidence_type: Literal["document_quote", "extracted_field", "validation_check"]
    evidence: str = Field(description="document_quote: exact short quote from the original document; "
                                      "extracted_field: field name such as 'total' or 'line_items[2].quantity'; "
                                      "validation_check: check name such as 'Inventory'")


class Critique(BaseModel):
    """What the critic agent returns."""

    verdict: Literal["AGREE", "DISAGREE"]
    recommended_decision: Literal["APPROVE", "REJECT"]
    verified: list[Claim] = Field(description="Reasons and extracted facts confirmed against the document and checks")
    problems: list[Claim] = Field(description="Reasons that are wrong or unsupported, extraction errors, missed red flags")
    explanation: str


class ApprovalRound(BaseModel):
    round: int
    decision: Decision
    decision_overridden: str | None = None  # set when code overrode the agent (e.g. tried to approve a FAIL)
    critique: Critique | None = None
    critic_model: str | None = None
    critique_unverified: list[str] = Field(default_factory=list)  # citations code could not verify


class ApprovalResult(BaseModel):
    invoice_number: str | None
    status: Literal["APPROVED", "REJECTED", "ESCALATED"]
    high_value: bool  # total over $10K (or unknown) -> extra scrutiny
    rounds: list[ApprovalRound] = Field(default_factory=list)
    final_reasons: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)  # fallbacks, overrides, skipped steps
    llm_usage: LLMUsage = Field(default_factory=LLMUsage)
