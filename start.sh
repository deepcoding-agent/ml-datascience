#!/usr/bin/env bash
# =============================================================================
# start.sh — Start the DS-Agent FastAPI backend (logs stream to terminal)
# =============================================================================
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Load environment variables from api/.env
if [ -f "api/.env" ]; then
  export $(grep -v '^#' api/.env | xargs)
fi

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
else
  echo "[start] ERROR: no virtual environment found (expected .venv/ or venv/)"
  exit 1
fi

RELOAD="${RELOAD:-true}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"

echo ""
echo "  ┌─────────────────────────────────────────┐"
echo "  │  DS-Agent API                           │"
echo "  │  http://localhost:${PORT}                   │"
echo "  │  LOG_LEVEL=${LOG_LEVEL}   RELOAD=${RELOAD}          │"
echo "  └─────────────────────────────────────────┘"
echo ""

if [ "$RELOAD" = "true" ]; then
  exec uvicorn api.main:app --reload --host 0.0.0.0 --port "$PORT" --log-level "$LOG_LEVEL"
else
  exec uvicorn api.main:app --host 0.0.0.0 --port "$PORT" --log-level "$LOG_LEVEL"
fi
