from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("recommendation_service")


def configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    safe: dict[str, Any] = {"event": event}
    for key, value in fields.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            safe[key] = value
        else:
            safe[key] = type(value).__name__
    logger.log(level, json.dumps(safe, sort_keys=True, separators=(",", ":")))
