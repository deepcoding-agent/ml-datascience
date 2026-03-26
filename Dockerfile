FROM python:3.11-slim

WORKDIR /app

# System build tools + fonts for Thai language support in matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl \
    fonts-thai-tlwg fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code (overridden by volume mount in docker-compose for dev)
COPY api/ ./api/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI with --reload for hot reload on code changes
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
