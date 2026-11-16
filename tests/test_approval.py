from datetime import date

import pytest

from approval.agents import ClaudeCritic
from approval.models import Claim, Critique, Decision
from approval.stage import ApprovalStage
from ingestion.models import IngestionError, IngestionResult, Invoice, LineItem, LLMUsage
from validation.models import CheckResult, ValidationResult


def make_case(total: float, status: str = "PASS", tmp_path=None):
    inv = Invoice(invoice_number="INV-9001", vendor="Acme Vendor", invoice_date=date(2026, 1, 1),
                  due_date=date(2026, 2, 1), currency="USD",
                  line_items=[LineItem(line=1, raw_name="WidgetA", item="WidgetA", quantity=1, unit_price=total,
                                       amount=None)],
                  subtotal=None, tax=None, shipping=None, total=total, payment_terms=None, notes=None)
    source = tmp_path / "invoice.txt"
    source.write_text(f"Invoice INV-9001 total {total}")
    ingestion = IngestionResult(source_file=str(source), extraction_method="llm", invoice=inv)
    check = CheckResult(name="check_inventory", label="Inventory", status="pass" if status == "PASS" else "fail",
                        summary="ok")
    validation = ValidationResult(invoice_number="INV-9001", status=status, checks=[check])
    return ingestion, validation


def decision(d="APPROVE", scrutiny=()):
    return Decision(decision=d, reasons=[f"reason for {d}"], scrutiny_findings=list(scrutiny))


def critique(verdict="AGREE", rec="APPROVE"):
    return Critique(verdict=verdict, recommended_decision=rec, verified=[Claim(statement="ok", evidence_type="validation_check", evidence="Inventory")],
                    problems=[] if verdict == "AGREE" else [Claim(statement="wrong", evidence_type="validation_check", evidence="Inventory")], explanation="...")


class FakeApprover:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = []
        self.usage = LLMUsage()

    def decide(self, case, history, round_no, policy_note=None):
        self.calls.append({"case": case, "history": history, "policy_note": policy_note})
        return self.decisions.pop(0)


class FakeCritic:
    def __init__(self, critiques, model="fake-critic"):
        self.critiques = list(critiques)
        self.model = model
        self.inputs = []
        self.usage = LLMUsage()

    def review(self, review_input, round_no):
        self.inputs.append(review_input)
        item = self.critiques.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def run(tmp_path, total, status, decisions, critiques):
    approver, critic = FakeApprover(decisions), FakeCritic(critiques)
    result = ApprovalStage(approver=approver, critic=critic).run(*make_case(total, status, tmp_path))
    return result, approver, critic


def test_agree_first_round_approves(tmp_path):
    result, _, critic = run(tmp_path, 5000, "PASS", [decision()], [critique()])
    assert result.status == "APPROVED" and len(result.rounds) == 1 and not result.high_value
    assert "Invoice INV-9001 total 5000" in critic.inputs[0]  # critic sees the original document


def test_disagree_then_revise_then_agree(tmp_path):
    result, approver, _ = run(tmp_path, 5000, "PASS", [decision("APPROVE"), decision("REJECT")],
                              [critique("DISAGREE", "REJECT"), critique("AGREE", "REJECT")])
    assert result.status == "REJECTED" and len(result.rounds) == 2
    assert "critic_review" in approver.calls[1]["history"]  # revision round sees the critique


def test_persistent_disagreement_escalates(tmp_path):
    result, _, _ = run(tmp_path, 5000, "PASS", [decision()] * 3, [critique("DISAGREE", "REJECT")] * 3)
    assert result.status == "ESCALATED" and len(result.rounds) == 3
    assert any("still disagree" in r for r in result.final_reasons)


def test_fail_can_never_be_approved(tmp_path):
    result, approver, _ = run(tmp_path, 5000, "FAIL", [decision("APPROVE"), decision("APPROVE")],
                              [critique("AGREE", "APPROVE")])
    assert "POLICY VIOLATION" in approver.calls[1]["policy_note"]
    assert result.rounds[0].decision.decision == "REJECT"
    assert "code changed the decision to REJECT" in result.rounds[0].decision_overridden
    assert result.status == "REJECTED"


def test_high_value_requires_scrutiny(tmp_path):
    result, approver, _ = run(tmp_path, 15000, "PASS",
                              [decision("APPROVE"), decision("APPROVE", scrutiny=["no red flags"] * 3)],
                              [critique()])
    assert result.high_value
    assert "MISSING SCRUTINY" in approver.calls[1]["policy_note"]
    assert "Extra scrutiny applies: YES" in approver.calls[0]["case"]
    assert result.status == "APPROVED" and result.rounds[0].decision_overridden is None


def test_exactly_10k_is_standard_review(tmp_path):
    result, _, _ = run(tmp_path, 10000, "PASS", [decision()], [critique()])
    assert not result.high_value


def test_critic_unavailable_escalates_and_keeps_decision(tmp_path):
    result, _, _ = run(tmp_path, 5000, "PASS", [decision("APPROVE")], [IngestionError("rate limited")])
    assert result.status == "ESCALATED"
    assert result.rounds[0].decision.decision == "APPROVE" and result.rounds[0].critique is None
    assert "not double-checked" in result.final_reasons[0]


def test_claude_failure_falls_back_to_qwen(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    class FailingClaude(ClaudeCritic):
        def review(self, review_input, round_no):
            raise IngestionError("Anthropic API error 500")

    claude, qwen = FailingClaude(client=object()), FakeCritic([critique()], model="qwen")
    stage = ApprovalStage(approver=FakeApprover([decision()]), critic=claude, fallback_critic=qwen)
    result = stage.run(*make_case(5000, "PASS", tmp_path))
    assert result.status == "APPROVED" and result.rounds[0].critic_model == "qwen"
    assert any("used Qwen" in n for n in result.notes)


def test_no_anthropic_key_uses_fallback_critic(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    qwen = FakeCritic([critique()], model="qwen")
    stage = ApprovalStage(approver=FakeApprover([decision()]), fallback_critic=qwen)
    result = stage.run(*make_case(5000, "PASS", tmp_path))
    assert result.rounds[0].critic_model == "qwen"
    assert any("ANTHROPIC_API_KEY not set" in n for n in result.notes)


def test_no_groq_key_rejects_fail_and_escalates_pass(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert ApprovalStage().run(*make_case(5000, "FAIL", tmp_path)).status == "REJECTED"
    assert ApprovalStage().run(*make_case(5000, "PASS", tmp_path)).status == "ESCALATED"


def make_currency_case(tmp_path, currency, status="PASS"):
    ingestion, validation = make_case(5000, status, tmp_path)
    ingestion.invoice.currency = currency
    return ingestion, validation


def test_non_usd_is_escalated_even_when_agents_agree(tmp_path):
    stage = ApprovalStage(approver=FakeApprover([decision("APPROVE")]), critic=FakeCritic([critique()]))
    result = stage.run(*make_currency_case(tmp_path, "EUR"))
    assert result.status == "ESCALATED"
    assert result.final_reasons[0].startswith("Non-USD currency (EUR)")
    assert "would have been APPROVED" in result.final_reasons[1]
    assert len(result.rounds) == 1  # the agents still ran


def test_non_usd_fail_is_escalated_not_rejected(tmp_path):
    stage = ApprovalStage(approver=FakeApprover([decision("REJECT")]), critic=FakeCritic([critique("AGREE", "REJECT")]))
    assert stage.run(*make_currency_case(tmp_path, "EUR", status="FAIL")).status == "ESCALATED"


def test_non_usd_without_llm_is_escalated(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert ApprovalStage().run(*make_currency_case(tmp_path, "GBP", status="FAIL")).status == "ESCALATED"


def test_usd_follows_normal_rules(tmp_path):
    stage = ApprovalStage(approver=FakeApprover([decision("APPROVE")]), critic=FakeCritic([critique()]))
    assert stage.run(*make_currency_case(tmp_path, "USD")).status == "APPROVED"


@pytest.mark.parametrize("validation_status", ["PASS", "FAIL"])
def test_missing_currency_is_escalated(tmp_path, validation_status):
    d = "APPROVE" if validation_status == "PASS" else "REJECT"
    stage = ApprovalStage(approver=FakeApprover([decision(d)]), critic=FakeCritic([critique("AGREE", d)]))
    result = stage.run(*make_currency_case(tmp_path, None, status=validation_status))
    assert result.status == "ESCALATED"
    assert result.final_reasons[0].startswith("Currency not stated")


# --- critic citations (gap #2) -----------------------------------------------------

from approval.grounding import unverified_claims


def claim(statement, kind, evidence):
    return Claim(statement=statement, evidence_type=kind, evidence=evidence)


def grounded(*claims, tmp_path):
    ingestion, validation = make_case(5000, "PASS", tmp_path)
    ingestion.invoice.line_items[0].quantity = 2
    ingestion.invoice.line_items[0].unit_price = 2500.0
    c = Critique(verdict="AGREE", recommended_decision="APPROVE", verified=list(claims), problems=[], explanation="")
    return unverified_claims(c, "Invoice INV-9001 from Acme Vendor total 5000", "case text", ingestion.invoice,
                             validation)


def test_grounded_claims_pass(tmp_path):
    assert grounded(claim("Vendor matches", "document_quote", "from acme  VENDOR"),
                    claim("Quantity is 2 at 2,500 each = 5,000", "extracted_field", "line_items[1].quantity"),
                    claim("All 5 checks passed", "validation_check", "Inventory check"),
                    claim("Under the $10,000 threshold", "extracted_field", "total"), tmp_path=tmp_path) == []


@pytest.mark.parametrize("bad, expected", [
    (("Vendor matches", "document_quote", "Totally Real Vendor Ltd"), "quoted text not found"),
    (("Total is fine", "extracted_field", "grand_total"), "no extracted field named 'grand_total'"),
    (("Line 3 ok", "extracted_field", "line_items[3].quantity"), "no extracted field"),
    (("Stock ok", "validation_check", "Fraud screening"), "no validation check named"),
    (("Total is 5,400", "extracted_field", "total"), "the number 5400 is not in the document"),
])
def test_ungrounded_claims_are_caught(tmp_path, bad, expected):
    issues = grounded(claim(*bad), tmp_path=tmp_path)
    assert len(issues) == 1 and expected in issues[0]


def bad_critique():
    return Critique(verdict="AGREE", recommended_decision="APPROVE", explanation="",
                    verified=[claim("Vendor is Fake Co", "document_quote", "Fake Co")], problems=[])


def test_unverified_critique_gets_one_retry(tmp_path):
    critic = FakeCritic([bad_critique(), critique()])
    result = ApprovalStage(approver=FakeApprover([decision()]), critic=critic).run(*make_case(5000, "PASS", tmp_path))
    assert result.status == "APPROVED" and len(critic.inputs) == 2
    assert "could not be verified" in critic.inputs[1]


def test_still_unverified_critique_escalates(tmp_path):
    critic = FakeCritic([bad_critique(), bad_critique()])
    result = ApprovalStage(approver=FakeApprover([decision()]), critic=critic).run(*make_case(5000, "PASS", tmp_path))
    assert result.status == "ESCALATED"
    assert result.final_reasons[0].startswith("The critic's review could not be verified")
    assert result.rounds[0].critique_unverified
