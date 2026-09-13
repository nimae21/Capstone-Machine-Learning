import json
import logging
import os
import runpy
import threading
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd

import app as recommendation_app
import catalog_cache as catalog_module
from catalog_cache import CatalogCache, CatalogUnavailable, create_catalog_snapshot
from config import Settings
import rate_limit as rate_module
from rate_limit import RateLimiter


PRODUCTS = [
    {"product_id": 1, "category_id": 1, "brand_id": 1, "shoe_type_id": 1},
    {"product_id": 2, "category_id": 1, "brand_id": 1, "shoe_type_id": 1},
    {"product_id": 3, "category_id": 1, "brand_id": 2, "shoe_type_id": 1},
    {"product_id": 4, "category_id": 2, "brand_id": 3, "shoe_type_id": 2},
]
ACTIVITIES = [
    {"product_id": 1, "activity_type": "view", "activity_count": 2},
    {"product_id": 1, "activity_type": "search", "activity_count": None},
]


class CatalogCacheTests(unittest.TestCase):
    def test_cache_hit_and_ttl_refresh(self):
        calls = 0

        def builder():
            nonlocal calls
            calls += 1
            return create_catalog_snapshot(PRODUCTS, 10)

        cache = CatalogCache(builder, 0.01, 0.01, 1)
        first, first_status = cache.get()
        second, second_status = cache.get()
        time.sleep(0.02)
        third, third_status = cache.get()

        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertEqual(("miss", "hit", "miss"), (first_status, second_status, third_status))
        self.assertEqual(2, calls)

    def test_only_one_initial_rebuild_runs_for_concurrent_requests(self):
        entered = threading.Event()
        release = threading.Event()
        calls = 0
        results = []

        def builder():
            nonlocal calls
            calls += 1
            entered.set()
            release.wait(2)
            return create_catalog_snapshot(PRODUCTS, 10)

        cache = CatalogCache(builder, 60, 1, 2)
        threads = [
            threading.Thread(target=lambda: results.append(cache.get()[0]))
            for _ in range(5)
        ]
        for thread in threads:
            thread.start()
        self.assertTrue(entered.wait(1))
        release.set()
        for thread in threads:
            thread.join(3)

        self.assertEqual(1, calls)
        self.assertEqual(5, len(results))
        self.assertTrue(all(result is results[0] for result in results))

    def test_concurrent_refresh_uses_last_valid_snapshot(self):
        entered = threading.Event()
        release = threading.Event()
        should_block = False

        def builder():
            if should_block:
                entered.set()
                release.wait(2)
            return create_catalog_snapshot(PRODUCTS, 10)

        cache = CatalogCache(builder, 60, 1, 1)
        original, _ = cache.get()
        cache.invalidate()
        should_block = True
        thread = threading.Thread(target=lambda: cache.get())
        thread.start()
        self.assertTrue(entered.wait(1))
        stale, status = cache.get()
        release.set()
        thread.join(3)

        self.assertIs(original, stale)
        self.assertEqual("stale_refresh_in_progress", status)

    def test_failed_refresh_serves_stale_but_initial_failure_is_controlled(self):
        fail = False

        def builder():
            if fail:
                raise RuntimeError("database URL with secret")
            return create_catalog_snapshot(PRODUCTS, 10)

        cache = CatalogCache(builder, 60, 1, 1)
        original, _ = cache.get()
        cache.invalidate()
        fail = True
        stale, status = cache.get()
        self.assertIs(original, stale)
        self.assertEqual("stale", status)

        empty_cache = CatalogCache(lambda: (_ for _ in ()).throw(RuntimeError("no catalog")), 60, 1, 0.1)
        with self.assertRaises(CatalogUnavailable):
            empty_cache.get()

    def test_empty_single_and_low_distinct_catalogs_are_safe(self):
        empty = create_catalog_snapshot([], 10)
        single = create_catalog_snapshot([PRODUCTS[0]], 10)
        repeated = create_catalog_snapshot([PRODUCTS[0], dict(PRODUCTS[0], product_id=9)], 10)

        self.assertEqual(0, empty.product_count)
        self.assertEqual(0, empty.total_clusters)
        self.assertEqual(1, single.total_clusters)
        self.assertEqual(1, repeated.total_clusters)


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.original_settings = recommendation_app.settings
        self.original_key = recommendation_app.RECOMMENDATION_SERVICE_KEY
        self.original_cache = recommendation_app.catalog_cache
        self.original_limiter = recommendation_app.rate_limiter
        self.original_slots = recommendation_app.recommendation_slots
        recommendation_app.settings = replace(
            recommendation_app.settings,
            recommendation_rate_limit_per_minute=120,
            refresh_rate_limit_per_minute=2,
            activity_history_days=0,
        )
        recommendation_app.RECOMMENDATION_SERVICE_KEY = "shared-test-key"
        recommendation_app.rate_limiter = RateLimiter()
        recommendation_app.recommendation_slots = threading.BoundedSemaphore(4)
        recommendation_app.catalog_cache = CatalogCache(
            recommendation_app.build_catalog_snapshot, 600, 30, 1
        )
        self.client = recommendation_app.app.test_client()
        self.headers = {"X-Recommendation-Key": "shared-test-key"}

    def tearDown(self):
        recommendation_app.settings = self.original_settings
        recommendation_app.RECOMMENDATION_SERVICE_KEY = self.original_key
        recommendation_app.catalog_cache = self.original_cache
        recommendation_app.rate_limiter = self.original_limiter
        recommendation_app.recommendation_slots = self.original_slots

    @staticmethod
    def fake_fetch(query, params=(), *, operation="query"):
        if operation == "activity":
            return list(ACTIVITIES)
        if operation == "catalog":
            return list(PRODUCTS)
        raise AssertionError(operation)

    def test_warm_requests_reuse_catalog_and_do_not_refit_kmeans(self):
        operations = []
        original_cluster = catalog_module.cluster_products

        def fetch(query, params=(), *, operation="query"):
            operations.append(operation)
            return self.fake_fetch(query, params, operation=operation)

        with patch.object(recommendation_app, "fetch_all", side_effect=fetch), patch.object(
            catalog_module, "cluster_products", wraps=original_cluster
        ) as cluster:
            first = self.client.get("/recommendations/42?limit=2", headers=self.headers)
            second = self.client.get("/recommendations/42?limit=2", headers=self.headers)

        self.assertEqual(200, first.status_code)
        self.assertEqual(200, second.status_code)
        self.assertEqual(1, cluster.call_count)
        self.assertEqual(1, operations.count("catalog"))
        self.assertEqual(2, operations.count("activity"))
        self.assertNotIn(1, first.get_json()["product_ids"])

    def test_activity_is_aggregated_in_postgres_and_history_is_unbounded_by_default(self):
        captured = {}

        def fetch(query, params=(), *, operation="query"):
            captured.update(query=query, params=params, operation=operation)
            return []

        with patch.object(recommendation_app, "fetch_all", side_effect=fetch):
            response = self.client.get("/recommendations/42", headers=self.headers)

        self.assertEqual(200, response.status_code)
        self.assertIn("SUM(COALESCE(ua.activity_count, 1))", captured["query"])
        self.assertIn("GROUP BY ua.product_id, ua.activity_type", captured["query"])
        self.assertNotIn("created_at >=", captured["query"])
        self.assertEqual((42,), captured["params"])

    def test_optional_activity_history_window_is_parameterized(self):
        recommendation_app.settings = replace(
            recommendation_app.settings, activity_history_days=30
        )
        captured = {}

        def fetch(query, params=(), *, operation="query"):
            captured.update(query=query, params=params)
            return []

        with patch.object(recommendation_app, "fetch_all", side_effect=fetch):
            self.client.get("/recommendations/42", headers=self.headers)

        self.assertIn("CURRENT_TIMESTAMP", captured["query"])
        self.assertEqual((42, 30), captured["params"])

    def test_no_snapshot_failure_is_controlled_and_stale_snapshot_remains_available(self):
        calls = 0
        fail_catalog = False

        def fetch(query, params=(), *, operation="query"):
            nonlocal calls
            if operation == "activity":
                return list(ACTIVITIES)
            calls += 1
            if fail_catalog:
                raise RuntimeError("catalog failed")
            return list(PRODUCTS)

        with patch.object(recommendation_app, "fetch_all", side_effect=fetch):
            first = self.client.get("/recommendations/42", headers=self.headers)
            recommendation_app.catalog_cache.invalidate()
            fail_catalog = True
            stale = self.client.get("/recommendations/42", headers=self.headers)

        self.assertEqual(200, first.status_code)
        self.assertEqual(200, stale.status_code)
        self.assertEqual(2, calls)

        recommendation_app.catalog_cache = CatalogCache(
            lambda: (_ for _ in ()).throw(RuntimeError("catalog failed")), 60, 30, 0.1
        )
        with patch.object(recommendation_app, "fetch_all", return_value=list(ACTIVITIES)):
            unavailable = self.client.get("/recommendations/42", headers=self.headers)
        self.assertEqual(503, unavailable.status_code)
        self.assertEqual("catalog_unavailable", unavailable.get_json()["error"])

    def test_response_values_are_native_json_integers_and_seen_products_are_excluded(self):
        with patch.object(recommendation_app, "fetch_all", side_effect=self.fake_fetch):
            response = self.client.get("/recommendations/42?limit=3", headers=self.headers)

        payload = response.get_json()
        self.assertTrue(all(type(value) is int for value in payload["product_ids"]))
        self.assertTrue(all(type(value) is int for value in payload["clusters_assigned"]))
        self.assertIs(type(payload["total_clusters"]), int)
        self.assertNotIn(1, payload["product_ids"])

    def test_rate_limit_authentication_and_validation_precede_database_work(self):
        recommendation_app.settings = replace(
            recommendation_app.settings, recommendation_rate_limit_per_minute=1
        )
        with patch.object(recommendation_app, "fetch_all", return_value=[]) as fetch:
            self.assertEqual(401, self.client.get("/recommendations/42").status_code)
            self.assertEqual(
                422,
                self.client.get(
                    "/recommendations/42?limit=200", headers=self.headers
                ).status_code,
            )
            self.assertEqual(
                200, self.client.get("/recommendations/42", headers=self.headers).status_code
            )
            self.assertEqual(
                429, self.client.get("/recommendations/42", headers=self.headers).status_code
            )
        fetch.assert_called_once()

    def test_liveness_and_readiness_are_distinct_and_readiness_never_builds(self):
        builds = 0

        def builder():
            nonlocal builds
            builds += 1
            return create_catalog_snapshot(PRODUCTS, 10)

        recommendation_app.catalog_cache = CatalogCache(builder, 60, 30, 1)
        self.assertEqual(200, self.client.get("/health").status_code)
        self.assertEqual(200, self.client.get("/health/live").status_code)
        with patch.object(recommendation_app, "check_database", return_value=True):
            cold = self.client.get("/health/ready")
        self.assertEqual(503, cold.status_code)
        self.assertEqual(0, builds)

        recommendation_app.catalog_cache.get()
        with patch.object(recommendation_app, "check_database", return_value=True):
            ready = self.client.get("/health/ready")
        self.assertEqual(200, ready.status_code)
        self.assertEqual(1, builds)

    def test_refresh_requires_authentication_and_is_rate_limited(self):
        with patch.object(recommendation_app, "fetch_all", return_value=list(PRODUCTS)):
            self.assertEqual(401, self.client.post("/refresh").status_code)
            self.assertEqual(200, self.client.post("/refresh", headers=self.headers).status_code)
            self.assertEqual(200, self.client.post("/refresh", headers=self.headers).status_code)
            self.assertEqual(429, self.client.post("/refresh", headers=self.headers).status_code)

    def test_failure_logs_never_include_secret_values(self):
        secret = "postgresql://user:password@example.invalid/db?key=shared-test-key"
        events = []

        def capture(event, **fields):
            events.append((event, fields))

        with patch.object(recommendation_app, "fetch_all", side_effect=RuntimeError(secret)), patch.object(
            recommendation_app, "log_event", side_effect=capture
        ):
            response = self.client.get("/recommendations/42", headers=self.headers)

        self.assertEqual(503, response.status_code)
        serialized = json.dumps(events)
        self.assertNotIn("password", serialized)
        self.assertNotIn("shared-test-key", serialized)
        self.assertIn("RuntimeError", serialized)

    def test_expensive_work_is_rejected_when_all_slots_are_busy(self):
        recommendation_app.recommendation_slots = threading.BoundedSemaphore(1)
        recommendation_app.recommendation_slots.acquire()
        try:
            with patch.object(recommendation_app, "fetch_all") as fetch:
                response = self.client.get("/recommendations/42", headers=self.headers)
            self.assertEqual(503, response.status_code)
            self.assertEqual("service_busy", response.get_json()["error"])
            fetch.assert_not_called()
        finally:
            recommendation_app.recommendation_slots.release()

    def test_untrusted_forwarded_address_does_not_bypass_rate_limit(self):
        recommendation_app.settings = replace(
            recommendation_app.settings,
            recommendation_rate_limit_per_minute=1,
            trusted_proxies=(),
        )
        with patch.object(recommendation_app, "fetch_all", return_value=[]):
            first = self.client.get(
                "/recommendations/42",
                headers={**self.headers, "X-Forwarded-For": "198.51.100.1"},
            )
            second = self.client.get(
                "/recommendations/42",
                headers={**self.headers, "X-Forwarded-For": "198.51.100.2"},
            )
        self.assertEqual(200, first.status_code)
        self.assertEqual(429, second.status_code)
    def test_gunicorn_config_is_bounded_and_does_not_preload_the_app(self):
        config = runpy.run_path("gunicorn.conf.py")
        self.assertFalse(config["preload_app"])
        self.assertGreaterEqual(config["workers"], 1)
        self.assertLessEqual(config["workers"], 4)
        self.assertLessEqual(config["threads"], 4)


class ConfigurationAndLimiterTests(unittest.TestCase):
    def test_invalid_numeric_configuration_uses_defaults_and_safe_bounds(self):
        with patch.dict(
            os.environ,
            {
                "DB_POOL_MIN_SIZE": "invalid",
                "DB_POOL_MAX_SIZE": "999",
                "DB_CONNECT_TIMEOUT": "0",
                "CATALOG_CACHE_TTL_SECONDS": "99999",
                "RECOMMENDATION_MAX_CONCURRENCY": "0",
                "DB_SSLMODE": "not-a-mode",
            },
            clear=False,
        ):
            loaded = Settings.from_env()

        self.assertEqual(1, loaded.db_pool_min_size)
        self.assertEqual(32, loaded.db_pool_max_size)
        self.assertEqual(1, loaded.db_connect_timeout)
        self.assertEqual(3600, loaded.catalog_ttl_seconds)
        self.assertEqual(1, loaded.recommendation_max_concurrency)
        self.assertEqual("prefer", loaded.db_sslmode)

    def test_local_rate_limit_fallback_has_a_hard_bucket_bound(self):
        with patch.object(rate_module, "MAX_LOCAL_BUCKETS", 2):
            limiter = RateLimiter("")
            self.assertTrue(limiter.allow("test", "first", 5))
            self.assertTrue(limiter.allow("test", "second", 5))
            self.assertFalse(limiter.allow("test", "third", 5))
            self.assertEqual(2, len(limiter._counts))

    def test_configured_redis_limiter_uses_shared_counter(self):
        class Pipeline:
            count = 0

            def incr(self, _):
                return self

            def expire(self, *_):
                return self

            def execute(self):
                Pipeline.count += 1
                return [Pipeline.count, True]

        class Client:
            def pipeline(self, transaction=True):
                self.transaction = transaction
                return Pipeline()

        class RedisFactory:
            @staticmethod
            def from_url(*_, **__):
                return Client()

        fake_redis = type("RedisModule", (), {"Redis": RedisFactory})
        with patch.object(rate_module, "redis", fake_redis):
            limiter = RateLimiter("redis://example.invalid")
            self.assertTrue(limiter.allow("test", "client", 1))
            self.assertFalse(limiter.allow("test", "client", 1))

class ActivityCompatibilityTests(unittest.TestCase):
    def test_existing_weights_and_legacy_null_counts_are_preserved(self):
        frame = pd.DataFrame(
            [
                {"activity_type": "view", "activity_count": 1},
                {"activity_type": "search", "activity_count": None},
                {"activity_type": "add_to_cart", "activity_count": 2},
            ]
        )
        self.assertEqual(
            [1.0, 2.0, 10.0],
            recommendation_app.weighted_activity_strength(frame).tolist(),
        )


if __name__ == "__main__":
    unittest.main()
