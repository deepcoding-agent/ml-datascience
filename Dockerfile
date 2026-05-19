# Production image for Fly.io / any container host.
# For local dev with hot reload, use Dockerfile.dev (referenced by docker-compose).
FROM python:3.11-slim

WORKDIR /app

# System build tools + fonts for Thai/CJK in matplotlib charts
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    fonts-thai-tlwg fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for runtime
RUN useradd --create-home --shell /bin/bash --uid 1001 app

# Install Python dependencies (cached layer — only re-runs when requirements.txt changes)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY api/ ./api/

# Models directory — mounted as Fly volume in production, plain dir locally
RUN mkdir -p /app/models && chown -R app:app /app

USER app

EXPOSE 8000

ENV MODELS_DIR=/app/models \
    PYTHONUNBUFFERED=1

# Health check — generous timeouts so a single long LLM/training request
# can't trip a restart while the next probe is still queued.
HEALTHCHECK --interval=30s --timeout=15s --start-period=30s --retries=10 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=10)" || exit 1

# Production: 2 workers so /health (and short routes) keep responding while
# /eda or /train hog one worker for 30-90s on LLM + pandas + plotly. Routes
# are declared sync (`def`) so FastAPI offloads each one to its threadpool
# slot, freeing the worker's event loop to answer health probes.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers", "--forwarded-allow-ips", "*"]
