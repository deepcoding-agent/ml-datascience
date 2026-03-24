"""
DS-Agent FastAPI service
------------------------
Entry point — creates the app, registers middleware and routers.

Run
---
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Load .env from the api/ directory (works regardless of where uvicorn is launched from)
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.chat import router as chat_router
from api.routes.eda_report import router as eda_router
from api.routes.models import router as models_router
from api.routes.prepare import router as prepare_router
from api.routes.suggest_target import router as suggest_router

app = FastAPI(title="DS-Agent API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(eda_router)
app.include_router(models_router)
app.include_router(prepare_router)
app.include_router(suggest_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
