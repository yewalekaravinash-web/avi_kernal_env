FROM python:3.11-slim

WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

# ── Python dependencies (own layer for cache efficiency) ──────────────────────
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ── Application source ────────────────────────────────────────────────────────
# .dockerignore excludes: uv.lock, *.egg-info/, __pycache__, inference.py
COPY models.py        ./models.py
COPY __init__.py      ./__init__.py
COPY client.py        ./client.py
COPY openenv.yaml     ./openenv.yaml
COPY pyproject.toml   ./pyproject.toml
COPY README.md        ./README.md
COPY server/          ./server/

# NOTE: pip install . intentionally removed — caused build hangs due to stale
# openenv_avi_kernal_env.egg-info conflicting with pyproject.toml package name.
# PYTHONPATH=/app below makes all imports resolve without a registered package.

EXPOSE 8000

# ── Environment variables ──────────────────────────────────────────────────────
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
ENV HF_TOKEN=""
ENV ENABLE_WEB_INTERFACE=true

# ── Import path ───────────────────────────────────────────────────────────────
ENV PYTHONPATH="/app"

# ── Health check ──────────────────────────────────────────────────────────────
# Two-layer strategy:
#   1. Python urllib (no external binary dependency, always available)
#   2. start-period=90s accounts for openenv-core WebSocket pool cold-start
#
# NOTE: The /health endpoint is served by Starlette middleware in server/app.py
# (not a FastAPI route), guaranteeing it responds before the router is ready
# and can never be shadowed by openenv-core's own route registrations.
HEALTHCHECK --interval=15s --timeout=10s --start-period=90s --retries=5 \
    CMD python3 -c \
        "import urllib.request, sys; \
         r = urllib.request.urlopen('http://localhost:8000/health', timeout=8); \
         sys.exit(0 if r.status == 200 else 1)"

CMD ["uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--log-level", "info", \
     "--access-log"]
