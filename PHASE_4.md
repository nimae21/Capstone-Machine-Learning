# Phase 4 - recommendation service performance and reliability

Date: 2026-09-13. This checkpoint changes only the recommendation service. It does not deploy, migrate a production database, modify Laravel or mobile code, or begin Phase 5.

## Root cause and behavior

Previously, every active recommendation request ran one activity query and one catalog query through two new physical psycopg connections, rebuilt the catalog DataFrame and feature matrix, and fitted KMeans. The request path now leases connections from one bounded lazy pool per process and stores a prepared catalog snapshot for a bounded TTL.

The endpoint and successful JSON fields remain unchanged. Authentication still uses constant-time comparison of the shared X-Recommendation-Key. Limits remain 1-20 results. Activity weights remain view=1, search=2, and add_to_cart=5, multiplied by activity_count.

## PostgreSQL pool

The pool is created on first database work rather than module import. Defaults are:

- Minimum 1 and maximum 5 connections per Gunicorn worker.
- 2 second acquisition timeout.
- 3 second connect timeout.
- 3000 ms PostgreSQL statement timeout.
- 300 second maximum idle time and 1800 second maximum lifetime.
- Application name achilles-recommendations.
- Explicit sslmode, defaulting to prefer for development. Production must set require or a verified CA mode.

All numeric settings have bounds. Connection and cursor contexts release leases after success or exceptions. Shutdown closes the process pool. Structured events include acquisition and query duration, operation name, and row count without SQL parameters, URLs, or credentials.

With two default Gunicorn workers, DB_POOL_MAX_SIZE=5 permits at most ten service connections. Set this against the PostgreSQL plan's total connection budget.

## Catalog snapshot

Each process keeps a thread-safe snapshot containing product IDs and categorical fields, the one-hot feature matrix, feature column groups, cluster labels, product-to-cluster mapping, KMeans model, fingerprint, and build timestamp.

The default TTL is 600 seconds and is bounded between 30 and 3600 seconds. One thread rebuilds at a time. Concurrent requests use the previous snapshot during refresh. A failed refresh serves the prior snapshot and waits 30 seconds before another automatic attempt. With a healthy database, maximum normal staleness is the configured TTL. During a database or refresh outage, stale service can exceed the TTL until a refresh succeeds; logs identify every stale response.

Gunicorn uses preload_app=false so pools and mutable synchronization state are created after fork. Each worker has its own snapshot. This is not a globally shared model cache.

POST /refresh uses the existing service key, a default two-per-minute rate limit, and the same single-flight cache. It is intended for deployment warming or an explicit catalog refresh outside Laravel product transactions.

Empty and one-product catalogs return the existing insufficient_catalog behavior. Low-distinct catalogs reduce the cluster count safely. Missing category, brand, or shoe type values use a stable missing-value feature. random_state=42 and n_init=10 remain unchanged.

## Activity and request work

PostgreSQL now returns one row per product and activity type using SUM(COALESCE(activity_count, 1)). Only product_id, activity_type, and the aggregate are selected. The default ACTIVITY_HISTORY_DAYS=0 preserves all history. A positive configured value adds a parameterized created_at range.

A warm request builds only the user's small activity DataFrame and ranking vector. It does not query the catalog, rebuild its features, or fit KMeans. Seen products remain excluded and result IDs are converted to native Python integers.

The current Laravel unique aggregation index begins with user_id, product_id, and activity_type, so no new Laravel index or migration was added.

## Availability and workload protection

- GET /health and GET /health/live are liveness probes and do no database/model work.
- GET /health/ready performs a bounded pooled SELECT 1 and checks for an already-built snapshot. It never builds the model. It returns 503 until both are ready.
- POST /refresh authenticates and builds or refreshes the snapshot.
- Recommendation and refresh limits use Redis when REDIS_URL is configured. If Redis is absent or temporarily fails, a per-process minute counter capped at 10,000 active buckets remains available; new identifiers are rejected at that cap until old buckets expire.
- Forwarded client addresses are accepted only when request.remote_addr matches an explicitly configured TRUSTED_PROXIES IP/CIDR.
- Expensive recommendation work is bounded by RECOMMENDATION_MAX_CONCURRENCY per worker and a short slot-acquisition timeout.
- Flask caps request bodies and the endpoint caps responses at 20 product IDs.

Laravel continues using its deferred endpoint, per-user cache, short timeout, exclusions, and empty-section fallback. No synchronous Laravel-to-service invalidation was added.

## Structured events

Machine-readable JSON events cover request duration/status/result count, pool creation/acquisition/query time, cache hit/miss/stale use, snapshot build time/product/cluster count, Redis fallback, and controlled error types. Logs exclude user IDs, activity histories, request keys, database URLs, passwords, and SQL parameters.

## Local measurements

These use mocked database rows on Windows/Python 3.14 and a 500-product synthetic catalog. They are not production latency, PostgreSQL, CPU, or memory measurements.

- The former rebuild path fitted five models for five builds. Median build time was 178.56 ms; range 173.69-3102.22 ms, including first-use library initialization.
- The cached cold endpoint took 308.88 ms with one catalog query and one fit.
- Ten warm endpoint calls had a 73.60 ms median and 71.75-94.83 ms range, with zero catalog queries and zero fits.
- Eight concurrent calls during an expired snapshot all returned 200, completed in 865.43 ms total, and caused one fit.
- Python-tracked cached-workload memory was 1,892,903 bytes current and 3,039,375 bytes peak. The repeated-build loop peaked at 1,075,017 bytes. The snapshot intentionally retains memory to avoid repeated CPU/database work; native NumPy/BLAS and process RSS are not fully represented.
- Unit instrumentation constructed one pool for two queries and released both leases. No real PostgreSQL server was available, so physical connection reuse and pool contention still require deployment-like measurement.

## Production configuration

Use the platform secret store. Required values are documented in .env.example. For Railway or Render, set at minimum:

DATABASE_URL
RECOMMENDATION_SERVICE_KEY
DB_SSLMODE=require
DB_POOL_MIN_SIZE=1
DB_POOL_MAX_SIZE=5
DB_POOL_ACQUIRE_TIMEOUT=2
DB_CONNECT_TIMEOUT=3
DB_STATEMENT_TIMEOUT_MS=3000
CATALOG_CACHE_TTL_SECONDS=600
CATALOG_REFRESH_RETRY_SECONDS=30
REDIS_URL
RECOMMENDATION_RATE_LIMIT_PER_MINUTE=120
REFRESH_RATE_LIMIT_PER_MINUTE=2
RECOMMENDATION_MAX_CONCURRENCY=4
TRUSTED_PROXIES=<verified ingress CIDRs>
WEB_CONCURRENCY=2
GUNICORN_THREADS=2
PORT=<platform supplied>

Start command:

gunicorn -c gunicorn.conf.py app:app

Begin with two workers and two threads. Each worker owns a KMeans snapshot and a pool, so measure RSS, CPU, PostgreSQL connection use, and request concurrency before increasing either number. Gunicorn uses a 60 second request timeout, 30 second graceful timeout, 5 second keep-alive, bounded request recycling, stdout/stderr logs, and no preload.

After deployment, call POST /refresh once with X-Recommendation-Key, then use /health/ready for readiness. Do not configure a platform liveness check against readiness until the warm-up sequence is established.

## Remaining validation

- Real PostgreSQL TLS, statement timeout, physical connection reuse, pool exhaustion, and failure recovery.
- Redis-backed limiting across multiple workers and Redis outage behavior.
- Multi-worker snapshot memory, simultaneous cold starts, BLAS thread usage, and CPU saturation.
- Production catalog build/warm latency and real activity cardinality.
- Maximum stale-data behavior during a prolonged database outage.
- Railway/Render trusted ingress CIDRs and health-check sequencing.
- Dependency vulnerability audit; pip-audit was not installed in the local environment.
