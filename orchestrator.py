"""LangGraph orchestration of the whole invoice pipeline.

    START → ingest ─┬─ error ────────────────────────────────────────────────────► END
                    └─► validate ─► approval (subgraph: decide ⇄ critique) ─► pay ─► record ─► END

Each stage is a node; the approval loop is a nested subgraph (approval/graph.py) so the
agent ⇄ critic cycle is an explicit part of the graph. `python main.py --graph` prints it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from approval.graph import build_approval_graph
from approval.models import ApprovalResult
from approval.stage import ApprovalStage
from db.inventory import record_invoice_run
from db.seed_inventory import DB_PATH
from ingestion import IngestionError, Ingestor
from ingestion.models import IngestionResult
from observability import LOG_DIR
from payment.bank import mock_payment
from payment.models import PaymentResult
from payment.stage import process_payment
from validation import validate
from validation.models import ValidationResult

log = logging.getLogger("invoice.orchestrator")


class PipelineState(TypedDict, total=False):
    invoice_path: str
    run_id: str
    error: str | None
    ingestion: IngestionResult
    validation: ValidationResult
    approval: ApprovalResult
    payment: PaymentResult
    invoice_record_id: int


def build_pipeline(item_names: list[str], *, ingestor: Ingestor | None = None,
                   approval_stage: ApprovalStage | None = None, validation_agent=None,
                   pay: Callable[..., dict] = mock_payment, db_path: Path = DB_PATH, log_dir: Path = LOG_DIR):
    """Compile the pipeline graph. Every dependency can be injected for tests."""
    ingestor = ingestor or Ingestor(item_names=item_names)
    approval_stage = approval_stage or ApprovalStage()

    def ingest(state: PipelineState) -> dict[str, Any]:
        log.info("Node ingest: %s", state["invoice_path"])
        try:
            return {"ingestion": ingestor.ingest(Path(state["invoice_path"])), "error": None}
        except IngestionError as exc:
            log.error("Ingestion failed: %s", exc, extra={"data": {"invoice_path": state["invoice_path"]}})
            return {"error": str(exc)}

    def validate_node(state: PipelineState) -> dict[str, Any]:
        log.info("Node validate")
        return {"validation": validate(state["ingestion"], db_path=db_path, agent=validation_agent)}

    def pay_node(state: PipelineState) -> dict[str, Any]:
        log.info("Node pay: approval %s", state["approval"].status)
        return {"payment": process_payment(state["ingestion"], state["validation"], state["approval"],
                                           run_id=state.get("run_id"), db_path=db_path, log_dir=log_dir, pay=pay)}

    def record(state: PipelineState) -> dict[str, Any]:
        ingestion, payment = state["ingestion"], state["payment"]
        record_id = record_invoice_run(
            run_id=state.get("run_id"), source_file=ingestion.source_file,
            extraction_method=ingestion.extraction_method, invoice=ingestion.invoice, flags=ingestion.flags,
            validation_status=state["validation"].status, approval_status=state["approval"].status,
            payment_status=payment.status, amount_paid=payment.amount_paid, payment_reference=payment.reference,
            db_path=db_path)
        log.info("Node record: stored invoice record %d", record_id, extra={"data": {"invoice_record_id": record_id}})
        return {"invoice_record_id": record_id}

    graph = StateGraph(PipelineState)
    graph.add_node("ingest", ingest)
    graph.add_node("validate", validate_node)
    graph.add_node("approval", build_approval_graph(approval_stage))  # nested subgraph
    graph.add_node("pay", pay_node)
    graph.add_node("record", record)

    graph.add_edge(START, "ingest")
    graph.add_conditional_edges("ingest", lambda s: "failed" if s.get("error") else "ok",
                                {"ok": "validate", "failed": END})
    graph.add_edge("validate", "approval")
    graph.add_edge("approval", "pay")
    graph.add_edge("pay", "record")
    graph.add_edge("record", END)
    return graph.compile()


def run_pipeline(invoice_path: str, run_id: str, item_names: list[str], **deps) -> PipelineState:
    return build_pipeline(item_names, **deps).invoke({"invoice_path": invoice_path, "run_id": run_id})


def mermaid_diagram() -> str:
    """Mermaid source for the full graph, with the approval subgraph expanded."""
    return build_pipeline([], ingestor=object(), approval_stage=ApprovalStage()).get_graph(xray=True).draw_mermaid()
