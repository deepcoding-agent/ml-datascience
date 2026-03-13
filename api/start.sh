#!/usr/bin/env bash
# Start the DS-Agent FastAPI service.
# Run from the ml-datascience/ directory:
#   bash api/start.sh

set -e

cd "$(dirname "$0")/.."  # ensure we're in ml-datascience/

# ── Virtual-env check ──────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate

# ── Install deps (skip if already present) ────────────────────────────────
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# ── Ensure the ai_data_science_team package is installed in editable mode ──
pip install --quiet -e .

# ── Env vars ──────────────────────────────────────────────────────────────
if [ -f "api/.env" ]; then
  export $(grep -v '^#' api/.env | xargs)
fi

if [ -z "$OPENAI_API_KEY" ]; then
  echo "ERROR: OPENAI_API_KEY is not set. Copy api/.env.example to api/.env and fill it in."
  exit 1
fi

# ── Start server ──────────────────────────────────────────────────────────
echo "Starting DS-Agent API on http://localhost:8000 ..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
