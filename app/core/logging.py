"""
Structured Logging Configuration with Sensitive Data Sanitization.
Prevents leaking absolute filesystem paths, authentication tokens, or PII into logs.
"""
import logging
import re
import sys
from app.core.config import settings


class SensitiveDataLogFilter(logging.Filter):
    """Filter that masks sensitive tokens, passwords, and absolute host paths in log output."""

    # Patterns to mask
    PATTERNS = [
        (re.compile(r"(Bearer\s+)[A-Za-z0-9\-\._~\+\/]+=*", re.IGNORECASE), r"\1[FILTERED_TOKEN]"),
        (re.compile(r"(password[\"':\s=]+)[^\s,\"'}]+", re.IGNORECASE), r"\1[FILTERED_PASSWORD]"),
        (re.compile(r"(api[_-]?key[\"':\s=]+)[^\s,\"'}]+", re.IGNORECASE), r"\1[FILTERED_KEY]"),
        (re.compile(r"[a-zA-Z]:\\[Uu]sers\\[^\\]+\\", re.IGNORECASE), r"C:\\Users\\[USER]\\"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if not settings.ENABLE_LOG_MASKING:
            return True
        if isinstance(record.msg, str):
            for pattern, replacement in self.PATTERNS:
                record.msg = pattern.sub(replacement, record.msg)
        return True


def setup_logging() -> None:
    """Configures structured, sanitized logging for the application."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(SensitiveDataLogFilter())

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if reconfigured
    if not root_logger.handlers:
        root_logger.addHandler(handler)
    else:
        root_logger.handlers[0] = handler
