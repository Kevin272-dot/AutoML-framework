"""Structured logging.

Library code never calls ``print``. Logs are JSON when ``AUTOML_LOG_JSON=1`` (or when
stdout is not a TTY), which is what a container/collector expects, and human-readable
otherwise.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

_LOGGER_NAME = "rl_automl"
_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter with stable keys."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extra = getattr(record, "context", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} {record.getMessage()}"
        extra = getattr(record, "context", None)
        if isinstance(extra, dict) and extra:
            rendered = " ".join(f"{k}={v}" for k, v in extra.items())
            base = f"{base}  [{rendered}]"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


def configure_logging(level: str | int | None = None, json_output: bool | None = None) -> None:
    """Idempotently configure the package logger. Call once at an entry point."""
    global _CONFIGURED

    if level is None:
        level = os.environ.get("AUTOML_LOG_LEVEL", "INFO")
    if json_output is None:
        env_flag = os.environ.get("AUTOML_LOG_JSON")
        json_output = env_flag == "1" if env_flag is not None else not sys.stderr.isatty()

    root = logging.getLogger(_LOGGER_NAME)
    root.setLevel(level)
    root.propagate = False
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter() if json_output else HumanFormatter())
        root.handlers = [handler]
        _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger, e.g. ``get_logger("execution.executor")``."""
    configure_logging()
    return logging.getLogger(_LOGGER_NAME if not name else f"{_LOGGER_NAME}.{name}")


def log_context(**context: Any) -> dict[str, Any]:
    """Build the ``extra=`` payload for a structured log call."""
    return {"context": context}


__all__ = ["configure_logging", "get_logger", "log_context", "JsonFormatter", "HumanFormatter"]
