"""Approval stage: main agent decides, critic double-checks, up to 2 revision rounds, then escalate.

The loop itself is a LangGraph subgraph (approval/graph.py); this module holds the steps it runs.

Hard rules enforced in code (the LLMs cannot override them):
- A validation FAIL can never be approved.
- If the agent and critic still disagree after the revision rounds, the invoice is ESCALATED to a human VP.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from approval.agents import ApproverAgent, ClaudeCritic, GroqCritic
from approval.grounding import unverified_claims
from approval.models import HIGH_VALUE_THRESHOLD, ApprovalResult, ApprovalRound, Critique, Decision
from ingestion.models import IngestionError, IngestionResult, LLMUsage
from ingestion.readers import read_text
from llm_utils import has_api_key
from validation.models import ValidationResult

log = logging.getLogger("invoice.approval")

MAX_REVISIONS = 2


@dataclass
class ApprovalContext:
    ingestion: IngestionResult
    validation: ValidationResult
    high_value: bool
    failed: bool
    case: str
    document: str = ""
    skipped: bool = False  # no LLM available


def _case(ingestion: IngestionResult, validation: ValidationResult, high_value: bool) -> str:
    inv = ingestion.invoice
    checks = [{"check": c.label, "status": c.status,
               "details": [f.message for f in c.flags if f.severity != "info"] or [c.summary]}
              for c in validation.checks]
    total = f"{inv.total:,.2f} {inv.currency or ''}".strip() if inv.total is not None else "unknown"
    scrutiny = (f"YES - total {total} is over ${HIGH_VALUE_THRESHOLD:,.0f}" if inv.total is not None
                else "YES - total is unknown")
    return (
        f"Extra scrutiny applies: {scrutiny if high_value else 'NO'}\n"
        f"Validation status: {validation.status}\n"
        f"Validation checks:\n{json.dumps(checks, indent=1)}\n"
        f"Extracted invoice:\n<invoice>\n{inv.model_dump_json(exclude_none=True, exclude={"vendor_address"})}\n</invoice>"
    )


def _history(rounds: list[ApprovalRound]) -> str:
    last = rounds[-1]
    return ("Your previous decision and the critic's review:\n"
            + json.dumps({"your_decision": last.decision.model_dump(),
                          "critic_review": last.critique.model_dump() if last.critique else None}, indent=1))


def _review_input(case: str, document: str, decision: Decision, round_no: int) -> str:
    return (f"Review round {round_no}.\n\n{case}\n\n"
            f"Original invoice document:\n<document>\n{document}\n</document>\n\n"
            f"Approval agent's decision:\n{decision.model_dump_json(indent=1)}")


class ApprovalStage:
    def __init__(self, approver: ApproverAgent | None = None, critic=None, fallback_critic=None):
        self.approver = approver
        self.critic = critic
        self.fallback_critic = fallback_critic
        self.notes: list[str] = []
        self._critics_used: list = []

    def _use(self, critic):
        if critic not in self._critics_used:
            self._critics_used.append(critic)
        self.critic = critic
        return critic

    def _get_critic(self):
        if self.critic is None:
            if ClaudeCritic.available():
                return self._use(ClaudeCritic())
            self.notes.append("ANTHROPIC_API_KEY not set: critic used Qwen on Groq instead of Claude")
            return self._use(self.fallback_critic or GroqCritic())
        return self._use(self.critic)

    def _critique(self, review_input: str, round_no: int) -> tuple[Critique, str]:
        critic = self._get_critic()
        try:
            return critic.review(review_input, round_no), critic.model
        except IngestionError as exc:
            if isinstance(critic, ClaudeCritic):
                log.warning("Claude critic failed, falling back to Qwen: %s", exc)
                self.notes.append(f"Claude critic failed ({exc}); used Qwen on Groq instead")
                fallback = self._use(self.fallback_critic or GroqCritic())
                return fallback.review(review_input, round_no), fallback.model
            raise

    def _decide(self, case: str, history: str | None, round_no: int, failed: bool,
                high_value: bool) -> tuple[Decision, str | None]:
        decision = self.approver.decide(case, history, round_no)

        problems = []
        if failed and decision.decision == "APPROVE":
            problems.append("POLICY VIOLATION: you approved an invoice whose validation status is FAIL. "
                            "A FAIL can never be approved. REJECT it, citing the failed checks.")
        if high_value and not decision.scrutiny_findings:
            problems.append("MISSING SCRUTINY: this invoice is over $10,000, so you must report findings for each "
                            "of the 3 extra-scrutiny areas in scrutiny_findings (say 'none found' where relevant).")
        if problems:
            log.warning("Approver output broke policy; asking it to correct: %s", " | ".join(problems))
            decision = self.approver.decide(case, history, round_no, policy_note="\n".join(problems))

        overrides = []
        if failed and decision.decision == "APPROVE":
            decision = decision.model_copy(update={"decision": "REJECT"})
            overrides.append("Agent approved a validation FAIL twice; code changed the decision to REJECT")
        if high_value and not decision.scrutiny_findings:
            overrides.append("Agent did not document the required extra scrutiny")
        return decision, "; ".join(overrides) or None

    # --- steps of the approval loop (each one is a node in approval/graph.py) ---------

    def start(self, ingestion: IngestionResult, validation: ValidationResult) -> ApprovalContext:
        inv = ingestion.invoice
        high_value = inv.total is None or inv.total > HIGH_VALUE_THRESHOLD
        failed = validation.status == "FAIL"
        log.info("Approval for %s: validation %s, total %s, extra scrutiny %s", inv.invoice_number,
                 validation.status, inv.total, "yes" if high_value else "no",
                 extra={"data": {"invoice_number": inv.invoice_number, "high_value": high_value}})
        ctx = ApprovalContext(ingestion=ingestion, validation=validation, high_value=high_value, failed=failed,
                              case=_case(ingestion, validation, high_value))
        if self.approver is None and not has_api_key():
            ctx.skipped = True
            return ctx
        self.approver = self.approver or ApproverAgent()
        try:
            ctx.document = read_text(Path(ingestion.source_file))
        except IngestionError:
            ctx.document = "(original document could not be re-read)"
            self.notes.append("Critic could not see the original document")
        return ctx

    def decide(self, ctx: ApprovalContext, rounds: list[ApprovalRound]) -> ApprovalRound:
        """Main agent decides (or revises after a critique). Raises IngestionError if the LLM fails."""
        round_no = len(rounds) + 1
        decision, override = self._decide(ctx.case, _history(rounds) if rounds else None, round_no,
                                          ctx.failed, ctx.high_value)
        log.info("Round %d: approver says %s", round_no, decision.decision,
                 extra={"data": {"round": round_no, "decision": decision.decision, "reasons": decision.reasons}})
        return ApprovalRound(round=round_no, decision=decision, decision_overridden=override)

    def review(self, ctx: ApprovalContext, rnd: ApprovalRound) -> ApprovalRound:
        """Critic double-checks the round's decision, citing evidence that code then verifies.

        Unverifiable claims are sent back once; if they remain, the round is marked unverified (-> ESCALATED).
        Raises IngestionError if no critic is available.
        """
        review_input = _review_input(ctx.case, ctx.document, rnd.decision, rnd.round)
        invoice = ctx.ingestion.invoice
        critique, critic_model = self._critique(review_input, rnd.round)
        issues = unverified_claims(critique, ctx.document, ctx.case, invoice, ctx.validation)
        if issues:
            log.warning("Round %d: critic made %d unverifiable claim(s); asking it to correct them", rnd.round,
                        len(issues), extra={"data": {"round": rnd.round, "unverified": issues}})
            feedback = ("\n\nYour previous review had claims that could not be verified:\n- " + "\n- ".join(issues)
                        + "\nReturn a corrected review. Quote the document exactly, use real field and check names, "
                          "and only state numbers that appear in the document, extracted data or validation results.")
            critique, critic_model = self._critique(review_input + feedback, rnd.round)
            issues = unverified_claims(critique, ctx.document, ctx.case, invoice, ctx.validation)
            if issues:
                log.warning("Round %d: critic's review still unverifiable", rnd.round,
                            extra={"data": {"round": rnd.round, "unverified": issues}})
        log.info("Round %d: critic (%s) %s, recommends %s%s", rnd.round, critic_model, critique.verdict,
                 critique.recommended_decision, " (UNVERIFIED)" if issues else "",
                 extra={"data": {"round": rnd.round, "verdict": critique.verdict,
                                 "problems": [c.statement for c in critique.problems], "unverified": issues}})
        return rnd.model_copy(update={"critique": critique, "critic_model": critic_model,
                                      "critique_unverified": issues})

    def finish(self, ctx: ApprovalContext, rounds: list[ApprovalRound], outcome: str) -> ApprovalResult:
        """outcome: agreed | disagreed | critic_unverified | critic_unavailable | approver_failed | skipped"""
        inv = ctx.ingestion.invoice
        if outcome == "skipped":
            status = "REJECTED" if ctx.failed else "ESCALATED"
            reasons = (["Validation FAIL"] if ctx.failed
                       else ["No LLM available (GROQ_API_KEY not set); needs a human VP review"])
            self.notes.append("Approval agents skipped: GROQ_API_KEY not set")
        elif outcome == "approver_failed":
            status, reasons = "ESCALATED", ["Approval agent failed; needs a human VP review"]
        elif outcome == "critic_unavailable":
            final = rounds[-1]
            status = "ESCALATED"
            reasons = ([f"Critic unavailable, so the agent's {final.decision.decision} was not double-checked; "
                        "needs a human VP review"]
                       + [f"Agent ({final.decision.decision}): {r}" for r in final.decision.reasons])
        elif outcome == "critic_unverified":
            final = rounds[-1]
            status = "ESCALATED"
            reasons = ([f"The critic's review could not be verified against the document, so the agent's "
                        f"{final.decision.decision} was not reliably double-checked; needs a human VP review"]
                       + [f"Unverified: {i}" for i in final.critique_unverified[:5]]
                       + [f"Agent ({final.decision.decision}): {r}" for r in final.decision.reasons])
        elif outcome == "disagreed":
            final = rounds[-1]
            status = "ESCALATED"
            reasons = ([f"Agent and critic still disagree after {MAX_REVISIONS} revision rounds"]
                       + [f"Agent ({final.decision.decision}): {r}" for r in final.decision.reasons]
                       + [f"Critic (recommends {final.critique.recommended_decision}): {p.statement}"
                          for p in final.critique.problems])
        else:  # agreed
            final = rounds[-1]
            status = "APPROVED" if final.decision.decision == "APPROVE" else "REJECTED"
            reasons = final.decision.reasons

        # A missing or non-USD currency is always a human decision, whatever the agents concluded
        # (their reasoning is kept for the reviewer). Never paid automatically.
        if inv.currency != "USD":
            outcome_note = (f"Agents' outcome would have been {status}" if status != "ESCALATED"
                            else "Agents' outcome was also ESCALATED")
            if inv.currency is None:
                reason = "Currency not stated on the invoice: needs human VP review before any payment"
                note = "Status forced to ESCALATED: invoice currency is not stated"
            else:
                reason = (f"Non-USD currency ({inv.currency}): needs human VP review, since how to handle "
                          f"non-USD invoices is not defined")
                note = f"Status forced to ESCALATED: invoice currency is {inv.currency}"
            reasons = [reason, outcome_note] + reasons
            self.notes.append(note)
            status = "ESCALATED"

        calls = (self.approver.usage.calls if self.approver else []) + [
            c for critic in self._critics_used for c in critic.usage.calls]
        usage = LLMUsage(calls=calls)
        result = ApprovalResult(invoice_number=inv.invoice_number, status=status, high_value=ctx.high_value,
                                rounds=rounds, final_reasons=reasons, notes=self.notes, llm_usage=usage)
        log.info("Approval %s after %d round(s)", status, len(rounds),
                 extra={"data": {"invoice_number": inv.invoice_number, "status": status, "rounds": len(rounds),
                                 "outcome": outcome, "llm_usage": usage.summary()}})
        return result

    def run(self, ingestion: IngestionResult, validation: ValidationResult) -> ApprovalResult:
        """Run the approval loop on its own (the same LangGraph subgraph the full pipeline uses)."""
        from approval.graph import build_approval_graph

        state = build_approval_graph(self).invoke({"ingestion": ingestion, "validation": validation})
        return state["approval"]


def approve(ingestion: IngestionResult, validation: ValidationResult, **kwargs) -> ApprovalResult:
    return ApprovalStage(**kwargs).run(ingestion, validation)
