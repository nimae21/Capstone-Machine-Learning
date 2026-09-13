from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from modeling import FEATURE_COLUMNS, build_product_feature_matrix, cluster_products
from observability import log_event


class CatalogUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class CatalogSnapshot:
    product_ids: tuple[int, ...]
    products_df: pd.DataFrame
    feature_matrix: pd.DataFrame
    column_groups: Mapping[str, tuple[str, ...]]
    cluster_labels: np.ndarray
    product_to_cluster: Mapping[int, int]
    model: object | None
    fingerprint: str
    built_at: str
    built_monotonic: float
    build_ms: float

    @property
    def product_count(self) -> int:
        return len(self.product_ids)

    @property
    def total_clusters(self) -> int:
        if self.product_count == 0:
            return 0
        return int(getattr(self.model, "n_clusters", 1))


def create_catalog_snapshot(rows: list[dict], n_clusters: int) -> CatalogSnapshot:
    started = time.perf_counter()
    products_df = pd.DataFrame(rows, columns=["product_id", *FEATURE_COLUMNS])
    if not products_df.empty:
        products_df["product_id"] = products_df["product_id"].map(int)
    feature_matrix, column_groups = build_product_feature_matrix(products_df)
    labels, model = cluster_products(feature_matrix.values, n_clusters)
    products_df = products_df.copy()
    products_df["cluster"] = labels

    product_ids = tuple(int(value) for value in products_df["product_id"].tolist())
    mapping = MappingProxyType({
        product_id: int(cluster)
        for product_id, cluster in zip(product_ids, labels)
    })
    fingerprint_payload = [
        [
            int(row["product_id"]),
            row.get("category_id"),
            row.get("brand_id"),
            row.get("shoe_type_id"),
        ]
        for row in rows
    ]
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    build_ms = (time.perf_counter() - started) * 1000

    snapshot = CatalogSnapshot(
        product_ids=product_ids,
        products_df=products_df,
        feature_matrix=feature_matrix,
        column_groups=MappingProxyType(dict(column_groups)),
        cluster_labels=labels,
        product_to_cluster=mapping,
        model=model,
        fingerprint=fingerprint,
        built_at=datetime.now(timezone.utc).isoformat(),
        built_monotonic=time.monotonic(),
        build_ms=build_ms,
    )
    log_event(
        "catalog_snapshot_built",
        build_ms=round(build_ms, 2),
        product_count=snapshot.product_count,
        cluster_count=snapshot.total_clusters,
        fingerprint=fingerprint[:12],
    )
    return snapshot


class CatalogCache:
    def __init__(
        self,
        builder: Callable[[], CatalogSnapshot],
        ttl_seconds: int,
        retry_seconds: int,
        initial_wait_seconds: float,
    ) -> None:
        self._builder = builder
        self._ttl_seconds = ttl_seconds
        self._retry_seconds = retry_seconds
        self._initial_wait_seconds = initial_wait_seconds
        self._condition = threading.Condition()
        self._snapshot: CatalogSnapshot | None = None
        self._expires_at = 0.0
        self._next_refresh_at = 0.0
        self._refreshing = False

    def get(self, *, force: bool = False) -> tuple[CatalogSnapshot, str]:
        now = time.monotonic()
        with self._condition:
            if self._snapshot is not None and not force and now < self._expires_at:
                log_event("catalog_cache_access", cache_status="hit")
                return self._snapshot, "hit"
            if self._refreshing:
                if self._snapshot is not None:
                    log_event("catalog_cache_access", cache_status="stale_refresh_in_progress")
                    return self._snapshot, "stale_refresh_in_progress"
                self._condition.wait_for(
                    lambda: not self._refreshing,
                    timeout=self._initial_wait_seconds,
                )
                if self._snapshot is not None:
                    return self._snapshot, "waited"
                raise CatalogUnavailable("catalog snapshot is not available")
            if self._snapshot is not None and not force and now < self._next_refresh_at:
                log_event("catalog_cache_access", cache_status="stale_retry_delay")
                return self._snapshot, "stale_retry_delay"
            previous = self._snapshot
            self._refreshing = True

        try:
            snapshot = self._builder()
        except Exception as exc:
            with self._condition:
                self._refreshing = False
                self._next_refresh_at = time.monotonic() + self._retry_seconds
                self._condition.notify_all()
            log_event(
                "catalog_refresh_failed",
                level=logging.ERROR,
                error_type=type(exc).__name__,
                stale_available=previous is not None,
            )
            if previous is not None:
                return previous, "stale"
            raise CatalogUnavailable("catalog snapshot build failed") from exc

        with self._condition:
            self._snapshot = snapshot
            self._expires_at = time.monotonic() + self._ttl_seconds
            self._next_refresh_at = 0.0
            self._refreshing = False
            self._condition.notify_all()
        log_event("catalog_cache_access", cache_status="miss")
        return snapshot, "miss"

    def peek(self) -> CatalogSnapshot | None:
        with self._condition:
            return self._snapshot

    def invalidate(self) -> None:
        with self._condition:
            self._expires_at = 0.0
            self._next_refresh_at = 0.0

    def clear(self) -> None:
        with self._condition:
            self._snapshot = None
            self._expires_at = 0.0
            self._next_refresh_at = 0.0
            self._refreshing = False
            self._condition.notify_all()
