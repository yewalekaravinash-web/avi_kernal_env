"""FastAPI app for DataCleaner OpenEnv environment."""
import os
import sys

# ── Ensure /app is on sys.path so absolute imports always resolve ──────────────
_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core required. Install with: pip install openenv-core[core]"
    ) from e

try:
    from models import DataCleanerAction, DataCleanerObservation
    from server.environment import DataCleanerEnvironment
except ImportError:
    from ..models import DataCleanerAction, DataCleanerObservation  # type: ignore[import]
    from .environment import DataCleanerEnvironment  # type: ignore[import]

# ── Build the openenv-managed FastAPI app ─────────────────────────────────────
app = create_app(
    DataCleanerEnvironment,
    DataCleanerAction,
    DataCleanerObservation,
    env_name="data_cleaner_env",
    max_concurrent_envs=10,
)

# ── Health & root endpoints ────────────────────────────────────────────────────
# IMPORTANT: These are added via Starlette middleware instead of @app.get routes.
#
# Why middleware and not route decorators?
# ----------------------------------------
# create_app() from openenv-core may register its own /health route first.
# FastAPI uses FIRST-MATCH routing — any route openenv registers before ours
# would shadow @app.get("/health"), making it unreachable.
#
# Starlette middleware runs BEFORE the router, so it intercepts /health
# regardless of what routes openenv-core registers. This guarantees the Docker
# HEALTHCHECK and HuggingFace Spaces external probe always get a 200 response
# as soon as the process is alive, even if the openenv pool is still warming up.
#
# Without a reachable /health:
#   - Docker HEALTHCHECK marks the container unhealthy → container stops
#   - HuggingFace Spaces external probe (port 8000, README app_port) gets no
#     response → Space stays in "Building" state indefinitely

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Intercepts /health and / before FastAPI routing to guarantee reachability."""

    async def dispatch(self, request, call_next):
        path = request.url.path

        if path == "/health" or path == "/healthz":
            return JSONResponse(
                {"status": "healthy", "service": "data_cleaner_env"},
                status_code=200,
            )

        if path == "/":
            return JSONResponse(
                {
                    "status": "ok",
                    "service": "DataCleaner OpenEnv Environment",
                    "tasks": ["field-extraction", "dedup-merge", "multi-source-reconciliation"],
                },
                status_code=200,
            )

        return await call_next(request)


app.add_middleware(HealthCheckMiddleware)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info", access_log=True)


if __name__ == "__main__":
    main()
