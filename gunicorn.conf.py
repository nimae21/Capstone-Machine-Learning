import os


def bounded_int(name, default, minimum, maximum):
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


bind = f"0.0.0.0:{bounded_int('PORT', 5000, 1, 65535)}"
workers = bounded_int("WEB_CONCURRENCY", 2, 1, 4)
threads = bounded_int("GUNICORN_THREADS", 2, 1, 4)
worker_class = "gthread"
timeout = bounded_int("GUNICORN_TIMEOUT", 60, 15, 300)
graceful_timeout = bounded_int("GUNICORN_GRACEFUL_TIMEOUT", 30, 5, 120)
keepalive = bounded_int("GUNICORN_KEEPALIVE", 5, 1, 30)
max_requests = bounded_int("GUNICORN_MAX_REQUESTS", 1000, 100, 10000)
max_requests_jitter = bounded_int("GUNICORN_MAX_REQUESTS_JITTER", 100, 0, 1000)
preload_app = False
accesslog = "-"
errorlog = "-"
capture_output = True
access_log_format = '{"event":"http_access","method":"%(m)s","path":"%(U)s","status":%(s)s,"duration_us":%(D)s}'
