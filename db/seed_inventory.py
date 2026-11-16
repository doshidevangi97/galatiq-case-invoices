"""Create and seed the mock inventory database (inventory.db at the project root).

Safe to run repeatedly: existing rows are overwritten with the seed values.
Usage: python db/seed_inventory.py
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "inventory.db"

# Seed data from the case README.
SEED_ITEMS = [
    ("WidgetA", 15),
    ("WidgetB", 10),
    ("GadgetX", 5),
    ("FakeItem", 0),
]


SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS inventory (
        item_name TEXT PRIMARY KEY COLLATE NOCASE,
        quantity  INTEGER NOT NULL CHECK (quantity >= 0)
    )
    """,
    # Outcome of every invoice that reached the payment stage. One row per outcome, so an
    # invoice can be rejected first and paid later. Validation counts only 'paid' rows as
    # duplicates (same invoice number, any vendor).
    """
    CREATE TABLE IF NOT EXISTS processed_invoices (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        invoice_number    TEXT    COLLATE NOCASE,
        revision          INTEGER NOT NULL DEFAULT 0,
        vendor            TEXT,
        total             REAL,              -- invoice total
        currency          TEXT,
        status            TEXT    NOT NULL CHECK (status IN
                              ('paid', 'payment_failed', 'rejected', 'escalated', 'held_for_review')),
        amount_paid       REAL,              -- what was actually sent (the difference, for a later revision)
        payment_reference TEXT,
        reasons           TEXT,              -- JSON list
        source_file       TEXT,
        run_id            TEXT,
        processed_at      TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_processed_invoice_status ON processed_invoices (invoice_number, status)",
    # One record per pipeline run: what was extracted and how each stage ended. Re-runs of the same
    # file add new records (full history), linked to logs/pipeline.jsonl by run_id.
    """
    CREATE TABLE IF NOT EXISTS invoice_records (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id            TEXT,
        source_file       TEXT    NOT NULL,
        extraction_method TEXT,
        invoice_number    TEXT    COLLATE NOCASE,
        revision          INTEGER NOT NULL DEFAULT 0,
        vendor            TEXT,
        vendor_address    TEXT,
        invoice_date      TEXT,
        invoice_date_raw  TEXT,
        due_date          TEXT,
        due_date_raw      TEXT,
        currency          TEXT,
        subtotal          REAL,
        tax               REAL,
        shipping          REAL,
        total             REAL,
        payment_terms     TEXT,
        notes             TEXT,
        validation_status TEXT,
        approval_status   TEXT,
        payment_status    TEXT,
        amount_paid       REAL,
        payment_reference TEXT,
        flags             TEXT,               -- JSON list of {code, severity, message}
        created_at        TEXT    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS invoice_line_items (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        invoice_record_id INTEGER NOT NULL REFERENCES invoice_records (id) ON DELETE CASCADE,
        line              INTEGER NOT NULL,
        raw_name          TEXT,
        item              TEXT,
        quantity          REAL,
        unit_price        REAL,
        amount            REAL,
        note              TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_invoice_records_number ON invoice_records (invoice_number)",
    "CREATE INDEX IF NOT EXISTS idx_line_items_record ON invoice_line_items (invoice_record_id)",
]


def _migrate_old_ledger(conn: sqlite3.Connection) -> None:
    """The first version of processed_invoices allowed one row per invoice+revision. Replace it."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(processed_invoices)")}
    if columns and "id" not in columns:
        if conn.execute("SELECT COUNT(*) FROM processed_invoices").fetchone()[0]:
            conn.execute("ALTER TABLE processed_invoices RENAME TO processed_invoices_legacy")
        else:
            conn.execute("DROP TABLE processed_invoices")


def create_schema(conn: sqlite3.Connection) -> None:
    _migrate_old_ledger(conn)
    for statement in SCHEMA:
        conn.execute(statement)


def seed_inventory(db_path: Path = DB_PATH) -> Path:
    """Create all tables and (re)load the inventory seed rows. Paid-invoice history is kept."""
    with sqlite3.connect(db_path) as conn:
        create_schema(conn)
        conn.executemany(
            "INSERT OR REPLACE INTO inventory (item_name, quantity) VALUES (?, ?)",
            SEED_ITEMS,
        )
    return db_path


if __name__ == "__main__":
    path = seed_inventory()
    with sqlite3.connect(path) as conn:
        rows = conn.execute("SELECT item_name, quantity FROM inventory ORDER BY item_name").fetchall()
    print(f"Seeded {path}")
    for name, qty in rows:
        print(f"  {name:<10} {qty}")
