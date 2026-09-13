import unittest
from dataclasses import replace
from unittest.mock import patch

import db
from config import settings


class FakeCursor:
    def __init__(self, rows, error=None):
        self.rows = rows
        self.error = error
        self.executions = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, query, params):
        self.executions.append((query, params))
        if self.error:
            raise self.error

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, cursor):
        self.cursor_instance = cursor

    def cursor(self, **_):
        return self.cursor_instance


class ConnectionLease:
    def __init__(self, connection, owner):
        self.connection = connection
        self.owner = owner

    def __enter__(self):
        self.owner.active += 1
        self.owner.peak_active = max(self.owner.peak_active, self.owner.active)
        return self.connection

    def __exit__(self, *_):
        self.owner.active -= 1
        self.owner.releases += 1
        return False


class FakePool:
    def __init__(self, rows=None, query_error=None, acquire_error=None, **kwargs):
        self.kwargs = kwargs
        self.cursor = FakeCursor(rows or [{"value": 1}], query_error)
        self.connection_instance = FakeConnection(self.cursor)
        self.acquire_error = acquire_error
        self.open_calls = 0
        self.connection_calls = 0
        self.releases = 0
        self.active = 0
        self.peak_active = 0
        self.closed = False

    def open(self, wait=False):
        self.open_calls += 1
        self.wait = wait

    def connection(self, timeout):
        self.connection_calls += 1
        self.timeout = timeout
        if self.acquire_error:
            raise self.acquire_error
        return ConnectionLease(self.connection_instance, self)

    def close(self, timeout=5):
        self.closed = True

    def get_stats(self):
        return {"pool_size": 1, "pool_available": 1}


class PoolTests(unittest.TestCase):
    def setUp(self):
        db.close_pool()
        self.original_factory = db._pool_factory
        self.original_settings = db._settings
        db.configure(
            replace(
                settings,
                database_url="postgresql://example.invalid/database",
                db_pool_min_size=1,
                db_pool_max_size=2,
                db_pool_acquire_timeout=0.2,
                db_sslmode="require",
            )
        )

    def tearDown(self):
        db.close_pool()
        db._pool_factory = self.original_factory
        db._settings = self.original_settings

    def test_pool_is_lazy_and_reuses_the_same_physical_pool(self):
        created = []

        def factory(**kwargs):
            pool = FakePool(**kwargs)
            created.append(pool)
            return pool

        db.set_pool_factory(factory)
        self.assertIsNone(db._pool)
        self.assertEqual([{"value": 1}], db.fetch_all("SELECT 1"))
        self.assertEqual([{"value": 1}], db.fetch_all("SELECT 1"))

        self.assertEqual(1, len(created))
        self.assertEqual(1, created[0].open_calls)
        self.assertEqual(2, created[0].connection_calls)
        self.assertEqual(2, created[0].releases)
        self.assertEqual("require", created[0].kwargs["kwargs"]["sslmode"])
        self.assertEqual(2, created[0].kwargs["max_size"])

    def test_query_failure_releases_the_connection(self):
        pool = FakePool(query_error=RuntimeError("query failed"))
        db.set_pool_factory(lambda **_: pool)

        with self.assertRaisesRegex(RuntimeError, "query failed"):
            db.fetch_all("SELECT broken")

        self.assertEqual(1, pool.releases)
        self.assertEqual(0, pool.active)

    def test_pool_exhaustion_is_bounded_and_does_not_leak(self):
        pool = FakePool(acquire_error=TimeoutError("pool timeout"))
        db.set_pool_factory(lambda **_: pool)

        with self.assertRaisesRegex(TimeoutError, "pool timeout"):
            db.fetch_all("SELECT 1")

        self.assertEqual(0, pool.releases)
        self.assertEqual(0, pool.active)
        self.assertEqual(0.2, pool.timeout)

    def test_importing_flask_does_not_create_a_pool(self):
        db.close_pool()
        with patch.object(db, "_pool", None):
            import app  # noqa: F401

            self.assertIsNone(db._pool)


if __name__ == "__main__":
    unittest.main()
