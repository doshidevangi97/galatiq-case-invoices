"""Access to the mock inventory database: stock levels and the invoice outcome ledger."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from db.seed_inventory import DB_PATH, create_schema, seed_inventory


def ensure_inventory_db(db_path: Path = DB_PATH) -> Path:
    """Create and seed the database on first run; upgrade tables in an older file."""
    if not db_path.exists():
        seed_inventory(db_path)
    else:
        with sqlite3.connect(db_path) as conn:
            create_schema(conn)
    return db_path


def load_item_names(db_path: Path = DB_PATH) -> list[str]:
    """Canonical item names, used to standardise spellings like 'Widget a' -> 'WidgetA'."""
    return list(load_stock(db_path))


def load_stock(db_path: Path = DB_PATH) -> dict[str, int]:
    with sqlite3.connect(ensure_inventory_db(db_path)) as conn:
        return {name: qty for name, qty in conn.execute("SELECT item_name, quantity FROM inventory")}


def find_processed(invoice_number: str, db_path: Path = DB_PATH, status: str = "paid") -> list[dict]:
    """Ledger rows with this invoice number (any vendor, any revision) and status (default: paid)."""
    with sqlite3.connect(ensure_inventory_db(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM processed_invoices WHERE invoice_number = ? AND status = ? ORDER BY revision, id",
            (invoice_number, status),
        ).fetchall()
    return [dict(r) for r in rows]


def record_outcome(*, invoice_number: str | None, revision: int, vendor: str | None, total: float | None,
                   currency: str | None, status: str, source_file: str, amount_paid: float | None = None,
                   payment_reference: str | None = None, reasons: list[str] | None = None,
                   run_id: str | None = None, db_path: Path = DB_PATH) -> int:
    """Write one outcome row (paid, payment_failed, rejected, escalated, held_for_review)."""
    with sqlite3.connect(ensure_inventory_db(db_path)) as conn:
        cursor = conn.execute(
            "INSERT INTO processed_invoices (invoice_number, revision, vendor, total, currency, status, amount_paid,"
            " payment_reference, reasons, source_file, run_id, processed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (invoice_number, revision, vendor, total, currency, status, amount_paid, payment_reference,
             json.dumps(reasons or []), source_file, run_id, datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        return cursor.lastrowid


def record_invoice_run(*, run_id: str | None, source_file: str, extraction_method: str, invoice,
                       flags: list, validation_status: str | None, approval_status: str | None,
                       payment_status: str | None, amount_paid: float | None, payment_reference: str | None,
                       db_path: Path = DB_PATH) -> int:
    """Store one run's extracted invoice (header + line items) and stage outcomes. Returns the record id."""
    inv = invoice
    with sqlite3.connect(ensure_inventory_db(db_path)) as conn:
        cursor = conn.execute(
            "INSERT INTO invoice_records (run_id, source_file, extraction_method, invoice_number, revision, vendor,"
            " vendor_address, invoice_date, invoice_date_raw, due_date, due_date_raw, currency, subtotal, tax,"
            " shipping, total, payment_terms, notes, validation_status, approval_status, payment_status,"
            " amount_paid, payment_reference, flags, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, source_file, extraction_method, inv.invoice_number, inv.revision, inv.vendor,
             inv.vendor_address, inv.invoice_date and inv.invoice_date.isoformat(), inv.invoice_date_raw,
             inv.due_date and inv.due_date.isoformat(), inv.due_date_raw, inv.currency, inv.subtotal, inv.tax,
             inv.shipping, inv.total, inv.payment_terms, inv.notes, validation_status, approval_status,
             payment_status, amount_paid, payment_reference,
             json.dumps([{"code": f.code, "severity": f.severity, "message": f.message} for f in flags]),
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        record_id = cursor.lastrowid
        conn.executemany(
            "INSERT INTO invoice_line_items (invoice_record_id, line, raw_name, item, quantity, unit_price, amount,"
            " note) VALUES (?,?,?,?,?,?,?,?)",
            [(record_id, li.line, li.raw_name, li.item, li.quantity, li.unit_price, li.amount, li.note)
             for li in inv.line_items],
        )
    return record_id


def reset_ledger(db_path: Path = DB_PATH) -> int:
    """Delete every outcome row (paid, rejected, ...). Inventory is untouched. Returns rows deleted."""
    with sqlite3.connect(ensure_inventory_db(db_path)) as conn:
        return conn.execute("DELETE FROM processed_invoices").rowcount
