"""Logging: readable console logs (stderr) plus structured JSON-lines logs in logs/pipeline.jsonl."""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

LOG_DIR = Path(__file__).resolve().parent / "logs"


class _JsonLinesFormatter(logging.Formatter):
    def __init__(self, run_id: str):
        super().__init__()
        self.run_id = run_id

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if data := getattr(record, "data", None):
            entry["data"] = data
        return json.dumps(entry, default=str)


def setup_logging(level: str = "INFO") -> str:
    run_id = uuid.uuid4().hex[:8]
    LOG_DIR.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(LOG_DIR / "pipeline.jsonl", encoding="utf-8")
    file_handler.setFormatter(_JsonLinesFormatter(run_id))
    console_handler = RichHandler(console=Console(stderr=True), show_path=False, markup=False)

    root = logging.getLogger("invoice")
    root.handlers.clear()
    root.setLevel(level.upper())
    root.addHandler(console_handler)
    root.addHandler(file_handler)
    root.propagate = False
    return run_id
