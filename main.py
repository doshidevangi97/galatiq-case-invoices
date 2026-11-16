"""Acme invoice pipeline CLI.

Usage:
    python main.py --invoice_path=data/invoices/invoice_1001.txt
    python main.py --invoice_path=data/invoices/invoice_1004.json --json
    python main.py --reset-ledger      # clear paid/rejected/escalated records (asks to confirm)
    python main.py --graph             # print the LangGraph pipeline as a Mermaid diagram

The stages are orchestrated as a LangGraph state graph (orchestrator.py).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from db.inventory import ensure_inventory_db, load_item_names, reset_ledger
from display import render_approval, render_ingestion, render_llm_usage, render_payment, render_validation
from ingestion.models import LLMUsage
from observability import setup_logging
from orchestrator import mermaid_diagram, run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Process an invoice through the Acme invoice pipeline.")
    parser.add_argument("--invoice_path", help="Path to an invoice file (.txt, .pdf, .json, .csv, .xml)")
    parser.add_argument("--reset-ledger", action="store_true",
                        help="Delete all paid/rejected/escalated records (for repeat demos); asks to confirm")
    parser.add_argument("--graph", action="store_true", help="Print the pipeline graph (Mermaid) and exit")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON instead of tables")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    if args.reset_ledger:
        return _reset_ledger()
    if args.graph:
        print(mermaid_diagram())
        return 0
    if not args.invoice_path:
        parser.error("--invoice_path is required")

    load_dotenv()
    run_id = setup_logging(args.log_level)
    log = logging.getLogger("invoice")
    log.info("Run %s started", run_id, extra={"data": {"invoice_path": args.invoice_path}})

    ensure_inventory_db()
    state = run_pipeline(args.invoice_path, run_id, load_item_names())
    if state.get("error"):
        Console(stderr=True).print(f"[bold red]Ingestion failed:[/] {state['error']}")
        return 1
    ingestion, validation, approval, payment = (state["ingestion"], state["validation"], state["approval"],
                                                state["payment"])

    usage = LLMUsage(calls=ingestion.llm_usage.calls + validation.llm_usage.calls + approval.llm_usage.calls)
    log.info("Run %s finished: validation %s, approval %s, payment %s, %d LLM tokens", run_id, validation.status,
             approval.status, payment.status, usage.total_tokens,
             extra={"data": {"validation": validation.status, "approval": approval.status,
                             "payment": payment.status, "llm_usage": usage.summary()}})

    if args.json:
        print(json.dumps({
            "ingestion": ingestion.model_dump(mode="json"),
            "validation": validation.model_dump(mode="json"),
            "approval": approval.model_dump(mode="json"),
            "payment": payment.model_dump(mode="json"),
            "llm_usage_total": usage.summary(),
        }, indent=2))
    else:
        console = Console()
        render_ingestion(ingestion, console)
        render_validation(validation, console)
        render_approval(approval, console)
        render_payment(payment, console)
        console.rule("[bold cyan]Run summary")
        render_llm_usage(usage, console)
    return 0


def _reset_ledger() -> int:
    console = Console()
    console.print("[bold yellow]This deletes every paid, rejected, escalated and failed record[/] "
                  "(inventory is kept). Paid invoices could then be paid again.")
    try:
        answer = input("Type 'reset' to confirm: ")
    except EOFError:
        answer = ""
    if answer.strip().lower() != "reset":
        console.print("Cancelled; nothing deleted.")
        return 1
    console.print(f"Deleted {reset_ledger()} record(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
