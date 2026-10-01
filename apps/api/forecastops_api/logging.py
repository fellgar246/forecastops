"""Structured logging for the API process."""

import logging

import structlog


def configure_logging() -> None:
    """Send one JSON object per log event to stderr."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger().setLevel(logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
