"""Structured logging with mandatory redaction.

Every log record - message *and* formatted arguments - passes through
:func:`security.redact_text` before it reaches a handler. This is enforced by
a ``logging.Filter`` installed on the root logger, so a module that forgets to
redact still cannot leak a credential to stdout or to the run log file.

The formatter emits a compact single-line format for the console and JSON
lines for the per-run file handler, which keeps ``reports/runs/<id>/agent.log``
machine-greppable.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from security import redact_text

_CONFIGURED = False
_RUN_HANDLERS: dict[str, logging.Handler] = {}


class RedactingFilter(logging.Filter):
    """Scrub registered secret values from the rendered message.

    We redact the *rendered* message rather than the format string so that
    ``log.info("user=%s", username)`` is covered too.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
        except Exception:  # pragma: no cover - broken %-args
            rendered = str(record.msg)
        cleaned = redact_text(rendered)
        if cleaned != rendered or record.args:
            record.msg = cleaned
            record.args = ()
        if record.exc_info:
            # Exception text can contain a credential (e.g. a failed login
            # request body). Render it now, redact, and drop the raw tuple.
            import traceback

            formatted = "".join(traceback.format_exception(*record.exc_info))
            record.exc_text = redact_text(formatted)
            record.exc_info = None
        return True


class ConsoleFormatter(logging.Formatter):
    """Human-readable single line: ``12:04:31 INFO  planner  message``."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%H:%M:%S")
        base = f"{ts} {record.levelname:<7} {record.name:<22} {record.getMessage()}"
        if record.exc_text:
            base += "\n" + record.exc_text
        return base


class JsonLinesFormatter(logging.Formatter):
    """One JSON object per line, for the per-run log file."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        run_id = getattr(record, "run_id", None)
        if run_id:
            payload["run_id"] = run_id
        stage = getattr(record, "stage", None)
        if stage:
            payload["stage"] = stage
        if record.exc_text:
            payload["exception"] = record.exc_text
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Install the console handler and the redaction filter exactly once."""
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
        return

    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(ConsoleFormatter())
    console.addFilter(RedactingFilter())
    root.addHandler(console)

    # Third-party noise we do not want in a demo transcript.
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio", "PIL", "chromadb", "watchdog"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def attach_run_log(run_id: str, path: Path) -> None:
    """Add a JSON-lines file handler that captures everything for one run."""
    if run_id in _RUN_HANDLERS:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(JsonLinesFormatter())
    handler.addFilter(RedactingFilter())
    logging.getLogger().addHandler(handler)
    _RUN_HANDLERS[run_id] = handler


def detach_run_log(run_id: str) -> None:
    """Remove and close the per-run file handler."""
    handler = _RUN_HANDLERS.pop(run_id, None)
    if handler is not None:
        logging.getLogger().removeHandler(handler)
        handler.close()


def get_logger(name: str) -> logging.Logger:
    """Convenience accessor so callers do not import ``logging`` directly."""
    configure_logging()
    return logging.getLogger(name)
