from __future__ import annotations

import hmac
import ipaddress
import logging
import threading
import time

import pandas as pd
from flask import Flask, jsonify, request

import db
from catalog_cache import CatalogCache, CatalogUnavailable, create_catalog_snapshot
from config import settings
from modeling import (
    ACTIVITY_WEIGHTS,
    FEATURE_COLUMNS,
    N_CLUSTERS,
    build_product_feature_matrix,
    build_user_preference_vector,
    cluster_products,
    get_user_dominant_clusters,
    rank_candidates_by_tier,
    weighted_activity_strength,
)
from observability import configure_logging, log_event
from rate_limit import RateLimiter

configure_logging()
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = settings.max_request_bytes
RECOMMENDATION_SERVICE_KEY = settings.recommendation_service_key
fetch_all = db.fetch_all
check_database = db.check_database


def build_catalog_snapshot():
    rows = fetch_all(
        """
        SELECT product_id, category_id, brand_id, shoe_type_id
        FROM products
        WHERE is_active = true
        ORDER BY product_id
        """,
        operation="catalog",
    )
    return create_catalog_snapshot(rows, N_CLUSTERS)


catalog_cache = CatalogCache(
    build_catalog_snapshot,
    settings.catalog_ttl_seconds,
    settings.catalog_refresh_retry_seconds,
    settings.catalog_initial_wait_seconds,
)
rate_limiter = RateLimiter(settings.redis_url)
recommendation_slots = threading.BoundedSemaphore(settings.recommendation_max_concurrency)


def _trusted_client_ip() -> str:
    remote = request.remote_addr or "unknown"
    try:
        remote_ip = ipaddress.ip_address(remote)
        trusted = any(
            remote_ip in ipaddress.ip_network(value, strict=False)
            for value in settings.trusted_proxies
        )
    except ValueError:
        trusted = False
    if trusted:
        forwarded = request.headers.get("X-Forwarded-For", "").split(",", 1)[0].strip()
        try:
            return str(ipaddress.ip_address(forwarded))
        except ValueError:
            pass
    return remote


def _authenticate():
    if not RECOMMENDATION_SERVICE_KEY:
        return jsonify({"error": "service_not_configured"}), 503
    supplied_key = request.headers.get("X-Recommendation-Key", "")
    if not hmac.compare_digest(supplied_key, RECOMMENDATION_SERVICE_KEY):
        return jsonify({"error": "unauthorized"}), 401
    return None


def _activity_query(user_id: int):
    history_clause = ""
    params: list[int] = [user_id]
    if settings.activity_history_days > 0:
        history_clause = "AND ua.created_at >= CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')"
        params.append(settings.activity_history_days)
    return fetch_all(
        f"""
        SELECT
            ua.product_id,
            ua.activity_type,
            SUM(COALESCE(ua.activity_count, 1))::bigint AS activity_count
        FROM user_activities ua
        WHERE ua.user_id = %s
          {history_clause}
        GROUP BY ua.product_id, ua.activity_type
        """,
        tuple(params),
        operation="activity",
    )


def _completed(
    payload: dict,
    status: int,
    started: float,
    log_name: str = "recommendation_request_completed",
    **fields,
):
    log_event(
        log_name,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
        status_code=status,
        result_count=len(payload.get("product_ids", [])),
        **fields,
    )
    return jsonify(payload), status


@app.route("/recommendations/<int:user_id>", methods=["GET"])
def get_recommendations(user_id: int):
    started = time.perf_counter()
    if user_id < 1:
        return _completed({"error": "invalid_user"}, 422, started)
    authentication = _authenticate()
    if authentication is not None:
        log_event(
            "recommendation_request_completed",
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            status_code=authentication[1],
            result_count=0,
        )
        return authentication

    try:
        limit = int(request.args.get("limit", 8))
    except (TypeError, ValueError):
        return _completed({"error": "invalid_limit"}, 422, started)
    if not 1 <= limit <= 20:
        return _completed({"error": "invalid_limit"}, 422, started)

    client_ip = _trusted_client_ip()
    if not rate_limiter.allow(
        "recommendations",
        client_ip,
        settings.recommendation_rate_limit_per_minute,
    ):
        return _completed({"error": "rate_limited"}, 429, started)

    acquired = recommendation_slots.acquire(timeout=settings.recommendation_slot_timeout)
    if not acquired:
        return _completed({"error": "service_busy"}, 503, started)

    cache_status = None
    try:
        activities = _activity_query(user_id)
        if not activities:
            return _completed(
                {"product_ids": [], "reason": "no_activity_history"},
                200,
                started,
            )

        activity_df = pd.DataFrame(
            activities,
            columns=["product_id", "activity_type", "activity_count"],
        )
        activity_df["product_id"] = activity_df["product_id"].map(int)
        already_seen_ids = set(int(value) for value in activity_df["product_id"].unique())

        try:
            snapshot, cache_status = catalog_cache.get()
        except CatalogUnavailable:
            return _completed({"error": "catalog_unavailable"}, 503, started)

        if snapshot.product_count < 2:
            return _completed(
                {"product_ids": [], "reason": "insufficient_catalog"},
                200,
                started,
                cache_status=cache_status,
            )

        dominant_clusters = get_user_dominant_clusters(
            activity_df,
            snapshot.product_ids,
            snapshot.cluster_labels,
            top_n=2,
        )
        if not dominant_clusters:
            return _completed(
                {"product_ids": [], "reason": "no_cluster_signal"},
                200,
                started,
                cache_status=cache_status,
            )

        products_df = snapshot.products_df
        candidates = products_df[
            products_df["cluster"].isin(dominant_clusters)
            & ~products_df["product_id"].isin(already_seen_ids)
        ]
        if candidates.empty:
            candidates = products_df[~products_df["product_id"].isin(already_seen_ids)]
        if candidates.empty:
            return _completed(
                {"product_ids": [], "reason": "catalog_exhausted"},
                200,
                started,
                cache_status=cache_status,
            )

        user_vector = build_user_preference_vector(
            snapshot.feature_matrix.values,
            snapshot.product_ids,
            activity_df,
        )
        ranked = rank_candidates_by_tier(
            candidates,
            snapshot.feature_matrix,
            snapshot.column_groups,
            user_vector,
        )
        product_ids = [int(value) for value in ranked.head(limit)["product_id"].tolist()]
        return _completed(
            {
                "product_ids": product_ids,
                "reason": "unsupervised_clustering",
                "clusters_assigned": [int(value) for value in dominant_clusters],
                "total_clusters": snapshot.total_clusters,
            },
            200,
            started,
            cache_status=cache_status,
        )
    except Exception as exc:
        log_event(
            "recommendation_request_failed",
            level=logging.ERROR,
            error_type=type(exc).__name__,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return jsonify({"error": "recommendation_unavailable"}), 503
    finally:
        recommendation_slots.release()


@app.route("/refresh", methods=["POST"])
def refresh_catalog():
    started = time.perf_counter()
    authentication = _authenticate()
    if authentication is not None:
        log_event(
            "catalog_refresh_request_completed",
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            status_code=authentication[1],
        )
        return authentication
    if not rate_limiter.allow(
        "refresh",
        _trusted_client_ip(),
        settings.refresh_rate_limit_per_minute,
    ):
        return _completed({"error": "rate_limited"}, 429, started, log_name="catalog_refresh_request_completed")

    try:
        snapshot, cache_status = catalog_cache.get(force=True)
    except CatalogUnavailable:
        return _completed({"error": "catalog_unavailable"}, 503, started, log_name="catalog_refresh_request_completed")
    if cache_status == "stale":
        return _completed(
            {"error": "refresh_failed", "stale_available": True},
            503,
            started,
            log_name="catalog_refresh_request_completed",
            cache_status=cache_status,
        )
    status = 202 if cache_status == "stale_refresh_in_progress" else 200
    return _completed(
        {
            "status": "ready",
            "product_count": snapshot.product_count,
            "fingerprint": snapshot.fingerprint,
            "built_at": snapshot.built_at,
        },
        status,
        started,
        log_name="catalog_refresh_request_completed",
        cache_status=cache_status,
    )


@app.route("/health", methods=["GET"])
@app.route("/health/live", methods=["GET"])
def liveness():
    return jsonify({"status": "ok"})


@app.route("/health/ready", methods=["GET"])
def readiness():
    database_ready = check_database()
    snapshot = catalog_cache.peek()
    catalog_ready = snapshot is not None and snapshot.product_count >= 2
    status = 200 if database_ready and catalog_ready else 503
    return jsonify(
        {
            "status": "ready" if status == 200 else "unavailable",
            "database": "ready" if database_ready else "unavailable",
            "catalog": "ready" if catalog_ready else "unavailable",
        }
    ), status


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
