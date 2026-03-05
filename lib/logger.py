import logging
import os
import sys
import json
import contextvars
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

# ── Context var for per-request tracing ──────────────────────
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


# ── JSON Formatter (production / file handler) ───────────────
class JSONFormatter(logging.Formatter):
    """Structured JSON log lines — easy to ingest by ELK / CloudWatch / Loki."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get("-"),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


# ── Human-readable Formatter (console / dev) ─────────────────
class ConsoleFormatter(logging.Formatter):
    """Coloured, concise output for local development."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        rid = request_id_ctx.get("-")
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        base = f"{color}{ts} [{record.levelname:<8}]{self.RESET} [{rid}] {record.getMessage()}"
        if record.exc_info and record.exc_info[0] is not None:
            base += "\n" + self.formatException(record.exc_info)
        return base


# ── Setup ─────────────────────────────────────────────────────
def setup_logger() -> logging.Logger:
    env = os.getenv("ENV", "production").lower()
    log_level_name = os.getenv("LOG_LEVEL", "DEBUG" if env == "development" else "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    _logger = logging.getLogger("winner_bike")
    _logger.setLevel(log_level)
    _logger.handlers.clear()
    _logger.propagate = False

    # ── Console handler (always) ──────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    if env == "development":
        console_handler.setFormatter(ConsoleFormatter())
    else:
        console_handler.setFormatter(JSONFormatter())
    _logger.addHandler(console_handler)

    # ── File handler (production only) ────────────────────────
    if env != "development":
        log_dir = os.getenv("LOG_DIR", "logs")
        os.makedirs(log_dir, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=os.path.join(log_dir, "app.log"),
            maxBytes=10 * 1024 * 1024,   # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(JSONFormatter())
        _logger.addHandler(file_handler)

        # ── Separate error log ────────────────────────────────
        error_handler = RotatingFileHandler(
            filename=os.path.join(log_dir, "error.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        _logger.addHandler(error_handler)

    return _logger


logger = setup_logger()
