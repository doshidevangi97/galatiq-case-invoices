"""Terminal rendering of pipeline results."""

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from ingestion.models import IngestionResult, LLMUsage
from validation.models import ValidationResult

SEVERITY_STYLE = {"error": "bold red", "warning": "yellow", "info": "dim"}


def _money(value, currency: str | None) -> str:
    if value is None:
        return "[red]—[/]"
    return f"{value:,.2f} {currency or ''}".strip()


def _plain(value) -> str:
    return "[red]—[/]" if value is None else str(value)


def render_llm_usage(usage: LLMUsage, console: Console) -> None:
    if not usage.calls:
        console.print("[dim]LLM usage: none (no LLM calls this run)[/]")
        return
    models = ", ".join(sorted({c.model for c in usage.calls}))
    table = Table(title=f"LLM token usage [dim]({models})[/]", title_justify="left")
    for col, justify in (("File", "left"), ("Try", "right"), ("Prompt", "right"),
                         ("Completion (reasoning)", "right"), ("Total", "right"), ("Time", "right")):
        table.add_column(col, justify=justify, no_wrap=True)
    for c in usage.calls:
        reasoning = f" [dim]({c.reasoning_tokens:,})[/]" if c.reasoning_tokens else ""
        table.add_row(c.file, str(c.attempt), f"{c.prompt_tokens:,}", f"{c.completion_tokens:,}{reasoning}",
                      f"{c.total_tokens:,}", f"{c.latency_ms / 1000:.1f}s")
    table.add_section()
    table.add_row(f"[bold]Total ({len(usage.calls)} call{'s' if len(usage.calls) != 1 else ''})[/]", "",
                  f"{usage.prompt_tokens:,}", f"{usage.completion_tokens:,}", f"[bold]{usage.total_tokens:,}[/]",
                  f"{sum(c.latency_ms for c in usage.calls) / 1000:.1f}s")
    console.print(table)


def render_ingestion(result: IngestionResult, console: Console | None = None) -> None:
    console = console or Console()
    inv = result.invoice
    console.rule("[bold cyan]Stage 1 · Ingestion")

    header = Table.grid(padding=(0, 2))
    header.add_column(style="bold")
    header.add_column()
    header.add_row("Source", f"{result.source_file}  [dim](via {result.extraction_method})[/]")
    header.add_row("Invoice", f"{_plain(inv.invoice_number)}  [dim]revision R{inv.revision}[/]")
    header.add_row("Vendor", _plain(inv.vendor))
    header.add_row("Invoice date", _plain(inv.invoice_date) if inv.invoice_date or not inv.invoice_date_raw
                   else f"[red]unreadable: {inv.invoice_date_raw!r}[/]")
    header.add_row("Due date", _plain(inv.due_date) if inv.due_date or not inv.due_date_raw
                   else f"[red]unreadable: {inv.due_date_raw!r}[/]")
    header.add_row("Currency", _plain(inv.currency))
    header.add_row("Payment terms", _plain(inv.payment_terms))
    if inv.notes:
        header.add_row("Notes", inv.notes)
    console.print(Panel(header, title="Ingestion — extracted invoice", border_style="cyan"))

    items = Table(title="Line items", title_justify="left")
    for col, justify in (("#", "right"), ("As written", "left"), ("Item", "left"), ("Qty", "right"),
                         ("Unit price", "right"), ("Amount", "right"), ("Note", "left")):
        items.add_column(col, justify=justify)
    for li in inv.line_items:
        items.add_row(str(li.line), _plain(li.raw_name), _plain(li.item), _plain(li.quantity),
                      _money(li.unit_price, None), "" if li.amount is None else f"{li.amount:,.2f}", li.note or "")
    items.add_section()
    for label, value in (("Subtotal", inv.subtotal), ("Tax", inv.tax), ("Shipping", inv.shipping)):
        if value is not None:
            items.add_row("", "", "", "", "", f"{label}: {value:,.2f}", "")
    items.add_row("", "", "", "", "", f"[bold]Total: {_money(inv.total, inv.currency)}[/]", "")
    console.print(items)

    totals = inv.quantity_by_item()
    if len(totals) < sum(1 for li in inv.line_items if li.item and li.quantity is not None):
        console.print("[bold]Quantity per item[/] (item appears on several lines): "
                      + ", ".join(f"{k} × {v}" for k, v in totals.items()))

    if result.related_files:
        related = Table(title="Related files (same invoice number)", title_justify="left")
        related.add_column("File")
        related.add_column("Revision")
        related.add_column("Relation")
        for r in result.related_files:
            related.add_row(r.path, f"R{r.revision}", r.relation.replace("_", " "))
        console.print(related)

    if result.flags:
        flags = Table(title="Flags", title_justify="left")
        flags.add_column("Severity")
        flags.add_column("Code")
        flags.add_column("Detail")
        order = {"error": 0, "warning": 1, "info": 2}
        for f in sorted(result.flags, key=lambda f: order[f.severity]):
            style = SEVERITY_STYLE[f.severity]
            flags.add_row(f"[{style}]{f.severity.upper()}[/]", f.code, f.message)
        console.print(flags)

    errors, warnings = result.count("error"), result.count("warning")
    colour = "red" if errors else "yellow" if warnings else "green"
    console.print(f"[bold {colour}]Ingestion complete — {errors} error(s), {warnings} warning(s)[/]")


CHECK_ICON = {"pass": "[green]✔ pass[/]", "fail": "[bold red]✘ fail[/]", "warn": "[yellow]! warn[/]",
              "skipped": "[dim]– skipped[/]"}


def render_validation(result: ValidationResult, console: Console | None = None) -> None:
    console = console or Console()
    console.rule("[bold cyan]Stage 2 · Validation")

    table = Table(title="Checks", title_justify="left", show_lines=True)
    table.add_column("Check", style="bold", no_wrap=True)
    table.add_column("Result", no_wrap=True)
    table.add_column("Details")
    for check in result.checks:
        details = "\n".join(escape(f.message) for f in check.flags if f.severity != "info") or escape(check.summary)
        table.add_row(check.label, CHECK_ICON[check.status], details)
    console.print(table)

    if result.summary:
        console.print(Panel(Markdown(result.summary), title="Validation agent summary", title_align="left",
                            border_style="blue"))
    elif result.summary_note:
        console.print(f"[dim]{escape(result.summary_note)}[/]")

    colour = "green" if result.status == "PASS" else "red"
    console.print(Panel(f"[bold {colour}]{result.status}[/]  ·  {result.count('error')} error(s), "
                        f"{result.count('warning')} warning(s)", border_style=colour, expand=False))


DECISION_STYLE = {"APPROVE": "green", "REJECT": "red", "APPROVED": "green", "REJECTED": "red", "ESCALATED": "yellow",
                  "AGREE": "green", "DISAGREE": "yellow"}


EVIDENCE_LABEL = {"document_quote": "doc", "extracted_field": "field", "validation_check": "check"}


def _evidence(claim) -> str:
    return f"[dim]({EVIDENCE_LABEL[claim.evidence_type]}: {escape(claim.evidence)})[/]"


def render_approval(result: "ApprovalResult", console: Console | None = None) -> None:
    console = console or Console()
    console.rule("[bold cyan]Stage 3 · Approval")
    scrutiny = "[bold yellow]extra scrutiny (over $10K)[/]" if result.high_value else "standard review (≤ $10K)"
    console.print(f"Review level: {scrutiny}")

    for rnd in result.rounds:
        d, c = rnd.decision, rnd.critique
        body = Table.grid(padding=(0, 1))
        body.add_column(style="bold", no_wrap=True)
        body.add_column()
        style = DECISION_STYLE[d.decision]
        body.add_row("Agent", f"[bold {style}]{d.decision}[/]")
        for reason in d.reasons:
            body.add_row("", f"• {escape(reason)}")
        if d.scrutiny_findings:
            body.add_row("Scrutiny", "\n".join(f"• {escape(s)}" for s in d.scrutiny_findings))
        if d.response_to_critic:
            body.add_row("To critic", escape(d.response_to_critic))
        if rnd.decision_overridden:
            body.add_row("Override", f"[red]{escape(rnd.decision_overridden)}[/]")
        if c:
            body.add_row("", "")
            body.add_row("Critic", f"[bold {DECISION_STYLE[c.verdict]}]{c.verdict}[/] [dim]({escape(rnd.critic_model or '')}, "
                                   f"recommends {c.recommended_decision})[/]")
            for v in c.verified:
                body.add_row("", f"[green]✔[/] {escape(v.statement)} {_evidence(v)}")
            for p in c.problems:
                body.add_row("", f"[yellow]✘[/] {escape(p.statement)} {_evidence(p)}")
            body.add_row("", f"[dim]{escape(c.explanation)}[/]")
            if rnd.critique_unverified:
                body.add_row("Unverified", "\n".join(f"[red]• {escape(i)}[/]" for i in rnd.critique_unverified))
        console.print(Panel(body, title=f"Round {rnd.round}", title_align="left", border_style="blue"))

    for note in result.notes:
        console.print(f"[dim]Note: {escape(note)}[/]")
    colour = DECISION_STYLE[result.status]
    reasons = "\n".join(f"• {escape(r)}" for r in result.final_reasons)
    console.print(Panel(f"[bold {colour}]{result.status}[/]\n{reasons}", border_style=colour,
                        title="Approval decision", title_align="left"))


PAYMENT_STYLE = {"PAID": "green", "PAYMENT_FAILED": "red", "REJECTED": "red", "PENDING_REVIEW": "yellow"}


def render_payment(result: "PaymentResult", console: Console | None = None) -> None:
    console = console or Console()
    console.rule("[bold cyan]Stage 4 · Payment")
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column()
    colour = PAYMENT_STYLE[result.status]
    grid.add_row("Outcome", f"[bold {colour}]{result.status.replace('_', ' ')}[/]  {escape(result.message)}")
    if result.amount_paid is not None:
        currency = result.currency or "(currency not stated)"
        grid.add_row("Amount paid", f"{result.amount_paid:,.2f} {currency}")
    if result.payment_kind == "difference" or result.already_paid:
        grid.add_row("Invoice total", f"{result.invoice_total:,.2f}  [dim](already paid for earlier revisions: "
                                      f"{result.already_paid:,.2f})[/]")
    if result.reference:
        grid.add_row("Reference", result.reference)
    if len(result.attempts) > 1 or (result.attempts and result.status != "PAID"):
        grid.add_row("Attempts", "\n".join(f"{a.attempt}: {escape(a.error or str(a.response))}" for a in result.attempts))
    for reason in result.reasons:
        grid.add_row("Reason", escape(reason))
    if result.log_file:
        grid.add_row("Logged to", f"{result.log_file}  [dim](ledger row {result.ledger_row_id})[/]")
    elif result.ledger_row_id:
        grid.add_row("Ledger row", str(result.ledger_row_id))
    console.print(Panel(grid, border_style=colour, title="Payment", title_align="left"))
