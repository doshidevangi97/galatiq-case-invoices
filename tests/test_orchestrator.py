import pytest

from approval.models import Claim, Critique, Decision
from approval.stage import ApprovalStage
from conftest import CATALOG, INVOICES, FakeLLM
from db.seed_inventory import seed_inventory
from ingestion import Ingestor
from ingestion.models import IngestionError, LLMUsage
from orchestrator import build_pipeline, mermaid_diagram
from test_approval import FakeApprover, FakeCritic


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)  # validation runs its checks in code
    db = tmp_path / "inventory.db"
    seed_inventory(db)
    return {"db_path": db, "log_dir": tmp_path / "logs"}


def decision(d):
    return Decision(decision=d, reasons=[f"reason {d}"], scrutiny_findings=["none found"] * 3)


def critique(verdict, rec):
    return Critique(verdict=verdict, recommended_decision=rec, verified=[Claim(statement="ok", evidence_type="validation_check", evidence="Inventory")],
                    problems=[] if verdict == "AGREE" else [Claim(statement="disagree", evidence_type="validation_check", evidence="Inventory")], explanation="...")


class Bank:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"status": "success"}


def run(env, path, decisions, critiques, bank=None, approver=None):
    stage = ApprovalStage(approver=approver or FakeApprover(decisions), critic=FakeCritic(critiques))
    graph = build_pipeline(CATALOG, ingestor=Ingestor(item_names=CATALOG, llm=FakeLLM({})),
                           approval_stage=stage, pay=bank or Bank(), **env)
    return graph.invoke({"invoice_path": str(path), "run_id": "test-run"})


def test_clean_invoice_flows_through_every_node_and_is_paid(env):
    bank = Bank()
    state = run(env, INVOICES / "invoice_1004_revised.json", [decision("APPROVE")], [critique("AGREE", "APPROVE")],
                bank)
    assert state["validation"].status == "PASS"
    assert state["approval"].status == "APPROVED"
    assert state["payment"].status == "PAID" and bank.calls[0]["amount"] == 5940.0
    assert not {"approval_ctx", "rounds", "approval_outcome"} & set(state)  # subgraph keys stay private


def test_failed_invoice_is_rejected_and_not_paid(env):
    bank = Bank()
    state = run(env, INVOICES / "invoice_1009.json", [decision("REJECT")], [critique("AGREE", "REJECT")], bank)
    assert state["validation"].status == "FAIL"
    assert state["payment"].status == "REJECTED" and bank.calls == []


def test_revision_cycle_runs_inside_the_graph(env):
    state = run(env, INVOICES / "invoice_1006.csv", [decision("APPROVE")] * 3,
                [critique("DISAGREE", "REJECT")] * 3)
    assert len(state["approval"].rounds) == 3
    assert state["approval"].status == "ESCALATED"
    assert state["payment"].status == "PENDING_REVIEW"


def test_ingestion_error_ends_the_graph(env, tmp_path):
    state = run(env, tmp_path / "missing.txt", [], [])
    assert "File not found" in state["error"]
    assert "payment" not in state


def test_approver_failure_escalates(env):
    class BrokenApprover:
        usage = LLMUsage()

        def decide(self, *args, **kwargs):
            raise IngestionError("Groq down")

    state = run(env, INVOICES / "invoice_1006.csv", [], [], approver=BrokenApprover())
    assert state["approval"].status == "ESCALATED"
    assert state["payment"].status == "PENDING_REVIEW"


def test_mermaid_diagram_shows_nested_approval_loop():
    diagram = mermaid_diagram()
    assert "subgraph approval" in diagram
    assert "revise" in diagram and "critic_review" in diagram


def test_missing_currency_invoice_is_escalated_and_not_paid(env):
    bank = Bank()
    state = run(env, INVOICES / "invoice_1006.csv", [decision("APPROVE")], [critique("AGREE", "APPROVE")], bank)
    assert state["validation"].status == "PASS"
    assert state["approval"].status == "ESCALATED"
    assert state["payment"].status == "PENDING_REVIEW" and bank.calls == []


def test_every_run_is_recorded_with_line_items(env):
    import sqlite3

    state = run(env, INVOICES / "invoice_1016.json", [decision("REJECT")], [critique("AGREE", "REJECT")])
    with sqlite3.connect(env["db_path"]) as conn:
        record = conn.execute("SELECT invoice_number, vendor_address, validation_status, approval_status,"
                              " payment_status FROM invoice_records WHERE id = ?",
                              (state["invoice_record_id"],)).fetchone()
        lines = conn.execute("SELECT COUNT(*) FROM invoice_line_items WHERE invoice_record_id = ?",
                             (state["invoice_record_id"],)).fetchone()[0]
    assert record == ("INV-1016", "100 Main St, Chicago, IL 60601", "FAIL", "REJECTED", "REJECTED")
    assert lines == 3
    run(env, INVOICES / "invoice_1016.json", [decision("REJECT")], [critique("AGREE", "REJECT")])
    with sqlite3.connect(env["db_path"]) as conn:
        assert conn.execute("SELECT COUNT(*) FROM invoice_records").fetchone()[0] == 2  # one per run
