from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    try:
        value = int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name)
    try:
        value = float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def _application_name() -> str:
    value = os.environ.get("DB_APPLICATION_NAME", "achilles-recommendations")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "-", value).strip("-")
    return (cleaned or "achilles-recommendations")[:64]


@dataclass(frozen=True)
class Settings:
    database_url: str = field(repr=False)
    recommendation_service_key: str = field(repr=False)
    redis_url: str = field(repr=False)
    db_pool_min_size: int
    db_pool_max_size: int
    db_pool_acquire_timeout: float
    db_connect_timeout: int
    db_statement_timeout_ms: int
    db_pool_max_idle: int
    db_pool_max_lifetime: int
    db_application_name: str
    db_sslmode: str
    catalog_ttl_seconds: int
    catalog_refresh_retry_seconds: int
    catalog_initial_wait_seconds: float
    activity_history_days: int
    recommendation_rate_limit_per_minute: int
    refresh_rate_limit_per_minute: int
    recommendation_max_concurrency: int
    recommendation_slot_timeout: float
    max_request_bytes: int
    trusted_proxies: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "Settings":
        sslmode = os.environ.get("DB_SSLMODE", "prefer").lower()
        if sslmode not in {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}:
            sslmode = "prefer"

        trusted = tuple(
            value.strip()
            for value in os.environ.get("TRUSTED_PROXIES", "").split(",")
            if value.strip()
        )

        minimum = _bounded_int("DB_POOL_MIN_SIZE", 1, 0, 10)
        maximum = _bounded_int("DB_POOL_MAX_SIZE", 5, 1, 32)
        minimum = min(minimum, maximum)

        return cls(
            database_url=os.environ.get("DATABASE_URL", ""),
            recommendation_service_key=os.environ.get("RECOMMENDATION_SERVICE_KEY", ""),
            redis_url=os.environ.get("REDIS_URL", ""),
            db_pool_min_size=minimum,
            db_pool_max_size=maximum,
            db_pool_acquire_timeout=_bounded_float("DB_POOL_ACQUIRE_TIMEOUT", 2.0, 0.1, 30.0),
            db_connect_timeout=_bounded_int("DB_CONNECT_TIMEOUT", 3, 1, 30),
            db_statement_timeout_ms=_bounded_int("DB_STATEMENT_TIMEOUT_MS", 3000, 250, 60000),
            db_pool_max_idle=_bounded_int("DB_POOL_MAX_IDLE_SECONDS", 300, 30, 3600),
            db_pool_max_lifetime=_bounded_int("DB_POOL_MAX_LIFETIME_SECONDS", 1800, 60, 86400),
            db_application_name=_application_name(),
            db_sslmode=sslmode,
            catalog_ttl_seconds=_bounded_int("CATALOG_CACHE_TTL_SECONDS", 600, 30, 3600),
            catalog_refresh_retry_seconds=_bounded_int("CATALOG_REFRESH_RETRY_SECONDS", 30, 5, 600),
            catalog_initial_wait_seconds=_bounded_float("CATALOG_INITIAL_WAIT_SECONDS", 3.0, 0.1, 30.0),
            activity_history_days=_bounded_int("ACTIVITY_HISTORY_DAYS", 0, 0, 3650),
            recommendation_rate_limit_per_minute=_bounded_int(
                "RECOMMENDATION_RATE_LIMIT_PER_MINUTE", 120, 1, 10000
            ),
            refresh_rate_limit_per_minute=_bounded_int(
                "REFRESH_RATE_LIMIT_PER_MINUTE", 2, 1, 60
            ),
            recommendation_max_concurrency=_bounded_int(
                "RECOMMENDATION_MAX_CONCURRENCY", 4, 1, 32
            ),
            recommendation_slot_timeout=_bounded_float(
                "RECOMMENDATION_SLOT_TIMEOUT_SECONDS", 0.25, 0.0, 5.0
            ),
            max_request_bytes=_bounded_int("MAX_REQUEST_BYTES", 16384, 1024, 1048576),
            trusted_proxies=trusted,
        )


settings = Settings.from_env()
