"""The approval loop as a LangGraph subgraph.

    approval_start ──► approver_decide ──► critic_review ──► approval_finish
          │                  ▲      │            │
          │                  └──────┼── DISAGREE ┘ (up to 2 revision rounds)
          └── no LLM ───────────────┴──────────────────────► approval_finish

The same compiled subgraph is used on its own (ApprovalStage.run) and as a node in the full pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from approval.models import ApprovalResult, ApprovalRound
from approval.stage import MAX_REVISIONS, ApprovalContext, ApprovalStage
from ingestion.models import IngestionError, IngestionResult
from validation.models import ValidationResult

log = logging.getLogger("invoice.approval")


class ApprovalState(TypedDict, total=False):
    # shared with the parent pipeline graph
    ingestion: IngestionResult
    validation: ValidationResult
    approval: ApprovalResult
    # private to the approval loop
    approval_ctx: ApprovalContext
    rounds: list[ApprovalRound]
    approval_outcome: str | None


def build_approval_graph(stage: ApprovalStage):
    def approval_start(state: ApprovalState) -> dict[str, Any]:
        ctx = stage.start(state["ingestion"], state["validation"])
        return {"approval_ctx": ctx, "rounds": [], "approval_outcome": "skipped" if ctx.skipped else None}

    def approver_decide(state: ApprovalState) -> dict[str, Any]:
        try:
            rnd = stage.decide(state["approval_ctx"], state["rounds"])
        except IngestionError as exc:
            log.error("Approval agent failed: %s", exc)
            stage.notes.append(f"Approval agent failed: {exc}")
            return {"approval_outcome": "approver_failed"}
        return {"rounds": state["rounds"] + [rnd]}

    def critic_review(state: ApprovalState) -> dict[str, Any]:
        rounds = state["rounds"]
        try:
            reviewed = stage.review(state["approval_ctx"], rounds[-1])
        except IngestionError as exc:
            log.error("Critic unavailable: %s", exc)
            stage.notes.append(f"Critic unavailable: {exc}")
            return {"approval_outcome": "critic_unavailable"}
        rounds = rounds[:-1] + [reviewed]
        if reviewed.critique_unverified:
            outcome = "critic_unverified"  # can't trust this review either way -> human decides
        elif reviewed.critique.verdict == "AGREE":
            outcome = "agreed"
        elif len(rounds) > MAX_REVISIONS:
            outcome = "disagreed"
        else:
            outcome = None  # go round again: the approver revises with the critique
        return {"rounds": rounds, "approval_outcome": outcome}

    def approval_finish(state: ApprovalState) -> dict[str, Any]:
        return {"approval": stage.finish(state["approval_ctx"], state.get("rounds", []), state["approval_outcome"])}

    graph = StateGraph(ApprovalState)
    graph.add_node("approval_start", approval_start)
    graph.add_node("approver_decide", approver_decide)
    graph.add_node("critic_review", critic_review)
    graph.add_node("approval_finish", approval_finish)

    graph.add_edge(START, "approval_start")
    graph.add_conditional_edges("approval_start",
                                lambda s: "finish" if s["approval_outcome"] == "skipped" else "decide",
                                {"decide": "approver_decide", "finish": "approval_finish"})
    graph.add_conditional_edges("approver_decide",
                                lambda s: "finish" if s.get("approval_outcome") == "approver_failed" else "review",
                                {"review": "critic_review", "finish": "approval_finish"})
    graph.add_conditional_edges("critic_review",
                                lambda s: "revise" if s.get("approval_outcome") is None else "finish",
                                {"revise": "approver_decide", "finish": "approval_finish"})
    graph.add_edge("approval_finish", END)
    return graph.compile()
