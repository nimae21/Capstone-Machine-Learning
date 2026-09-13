from __future__ import annotations

import atexit
import threading
import time
from typing import Any, Callable

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from config import Settings, settings as default_settings
from observability import log_event

_settings = default_settings
_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()
_pool_factory: Callable[..., ConnectionPool] = ConnectionPool


def configure(settings: Settings) -> None:
    global _settings
    close_pool()
    _settings = settings


def set_pool_factory(factory: Callable[..., ConnectionPool]) -> None:
    global _pool_factory
    close_pool()
    _pool_factory = factory


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is not None:
        return _pool

    with _pool_lock:
        if _pool is None:
            if not _settings.database_url:
                raise RuntimeError("DATABASE_URL is not configured")
            pool = _pool_factory(
                conninfo=_settings.database_url,
                min_size=_settings.db_pool_min_size,
                max_size=_settings.db_pool_max_size,
                timeout=_settings.db_pool_acquire_timeout,
                max_idle=_settings.db_pool_max_idle,
                max_lifetime=_settings.db_pool_max_lifetime,
                kwargs={
                    "connect_timeout": _settings.db_connect_timeout,
                    "options": f"-c statement_timeout={_settings.db_statement_timeout_ms}",
                    "application_name": _settings.db_application_name,
                    "sslmode": _settings.db_sslmode,
                },
                open=False,
                name="recommendations",
            )
            pool.open(wait=False)
            _pool = pool
            log_event(
                "db_pool_opened",
                min_size=_settings.db_pool_min_size,
                max_size=_settings.db_pool_max_size,
            )
    return _pool


def fetch_all(query: str, params: tuple[Any, ...] | None = None, *, operation: str = "query") -> list[dict]:
    acquire_started = time.perf_counter()
    pool = get_pool()
    acquired = False
    try:
        with pool.connection(timeout=_settings.db_pool_acquire_timeout) as connection:
            acquired = True
            acquire_ms = (time.perf_counter() - acquire_started) * 1000
            query_started = time.perf_counter()
            try:
                with connection.cursor(row_factory=dict_row) as cursor:
                    cursor.execute(query, params or ())
                    rows = cursor.fetchall()
            except Exception:
                log_event(
                    "db_query_failed",
                    operation=operation,
                    acquire_ms=round(acquire_ms, 2),
                )
                raise
            log_event(
                "db_query_completed",
                operation=operation,
                acquire_ms=round(acquire_ms, 2),
                query_ms=round((time.perf_counter() - query_started) * 1000, 2),
                row_count=len(rows),
            )
            return list(rows)
    except Exception as exc:
        if not acquired:
            log_event(
                "db_pool_acquire_failed",
                operation=operation,
                error_type=type(exc).__name__,
                duration_ms=round((time.perf_counter() - acquire_started) * 1000, 2),
            )
        raise


def check_database() -> bool:
    try:
        rows = fetch_all("SELECT 1 AS ok", operation="readiness")
        return bool(rows and int(rows[0]["ok"]) == 1)
    except Exception as exc:
        log_event("db_readiness_failed", error_type=type(exc).__name__)
        return False


def pool_stats() -> dict[str, int]:
    pool = _pool
    if pool is None:
        return {}
    try:
        return {key: int(value) for key, value in pool.get_stats().items()}
    except Exception:
        return {}


def close_pool() -> None:
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
    if pool is not None:
        try:
            pool.close(timeout=5.0)
        except Exception as exc:
            log_event("db_pool_close_failed", error_type=type(exc).__name__)


atexit.register(close_pool)
