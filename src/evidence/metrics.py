from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def log_event(event: str, **fields) -> None:
    logger.info("evidence.%s %s", event, fields)
