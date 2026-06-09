"""Structured JSON logging configuration for Emergo.

Provides a JsonFormatter that emits machine-readable log records suitable
for log aggregation systems (Datadog, Loki, Splunk, CloudWatch, etc.).

Usage::

    from emergo.logging_config import configure_structured_logging

    configure_structured_logging(level="INFO", include_emergo_context=True)
    # All emergo.* loggers now emit JSON lines to stderr
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import sys
from typing import Any

__all__ = [
    "JsonFormatter",
    "configure_structured_logging",
    "get_emergo_logger",
]


class JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects.

    Each line contains at minimum:
      timestamp  — ISO-8601 UTC (e.g. "2026-06-09T12:34:56.789Z")
      level      — "DEBUG" / "INFO" / "WARNING" / "ERROR" / "CRITICAL"
      logger     — logger name (e.g. "emergo.kernel")
      message    — the log message

    Plus any extra fields passed via logger.info("msg", extra={"key": value}).
    """

    _RESERVED = frozenset(
        {
            "args",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
        }
    )

    def __init__(self, extra_fields: dict[str, Any] | None = None) -> None:
        super().__init__()
        self._extra_fields: dict[str, Any] = dict(extra_fields) if extra_fields else {}

    def format(self, record: logging.LogRecord) -> str:
        # Build the base record
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc)
        # ISO-8601 with milliseconds, ending in Z
        timestamp = ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"

        payload: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add any static extra_fields baked into this formatter
        payload.update(self._extra_fields)

        # Add dynamic extra fields from the log call (exclude stdlib internals)
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value

        # Attach exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_structured_logging(
    level: str = "INFO",
    stream: Any = None,
    include_emergo_context: bool = True,
    extra_fields: dict[str, Any] | None = None,
) -> logging.Handler:
    """Configure structured JSON logging.

    Returns the installed handler (useful for testing).
    If include_emergo_context=True, also sets up the root emergo logger.
    """
    _stream = stream if stream is not None else sys.stderr
    handler = logging.StreamHandler(_stream)
    handler.setFormatter(JsonFormatter(extra_fields=extra_fields))
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    if include_emergo_context:
        emergo_logger = logging.getLogger("emergo")
        emergo_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        # Avoid duplicate handlers if called multiple times
        if not any(
            isinstance(h, logging.StreamHandler) and h.stream is _stream
            for h in emergo_logger.handlers
        ):
            emergo_logger.addHandler(handler)

    return handler


def get_emergo_logger(name: str, **context: Any) -> logging.LoggerAdapter[logging.Logger]:
    """Return a LoggerAdapter with extra context fields baked in.

    Usage::

        log = get_emergo_logger("emergo.kernel", run_id="exp_001", n_agents=50)
        log.info("Kernel started")  # → {"message": "Kernel started", "run_id": "exp_001", ...}
    """
    base_logger = logging.getLogger(name)
    return logging.LoggerAdapter(base_logger, extra=context)
