from __future__ import annotations

import hashlib
import threading
import time

from observability import log_event

MAX_LOCAL_BUCKETS = 10_000

try:
    import redis
except ImportError:
    redis = None


class RateLimiter:
    def __init__(self, redis_url: str = "") -> None:
        self._redis_url = redis_url
        self._redis = None
        self._redis_disabled_until = 0.0
        self._lock = threading.Lock()
        self._counts: dict[tuple[str, str, int], int] = {}

    def allow(self, scope: str, identifier: str, limit: int) -> bool:
        digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:32]
        window = int(time.time() // 60)
        if self._redis_url and redis is not None and time.monotonic() >= self._redis_disabled_until:
            try:
                client = self._redis
                if client is None:
                    client = redis.Redis.from_url(
                        self._redis_url,
                        decode_responses=True,
                        socket_connect_timeout=0.25,
                        socket_timeout=0.25,
                    )
                    self._redis = client
                key = f"recommendation-rate:{scope}:{digest}:{window}"
                pipeline = client.pipeline(transaction=True)
                pipeline.incr(key)
                pipeline.expire(key, 120)
                count, _ = pipeline.execute()
                return int(count) <= limit
            except Exception as exc:
                self._redis_disabled_until = time.monotonic() + 30
                log_event("rate_limit_redis_fallback", error_type=type(exc).__name__)

        key = (scope, digest, window)
        with self._lock:
            if len(self._counts) >= MAX_LOCAL_BUCKETS:
                self._counts = {
                    existing: count
                    for existing, count in self._counts.items()
                    if existing[2] >= window - 1
                }
                if key not in self._counts and len(self._counts) >= MAX_LOCAL_BUCKETS:
                    log_event("rate_limit_local_capacity_reached")
                    return False
            count = self._counts.get(key, 0) + 1
            self._counts[key] = count
            return count <= limit

    def clear(self) -> None:
        with self._lock:
            self._counts.clear()
        self._redis = None
        self._redis_disabled_until = 0.0
