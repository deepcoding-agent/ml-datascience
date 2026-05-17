"""
DS-Agent FastAPI service
------------------------
Entry point — creates the app, registers middleware and routers.

Run
---
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the api/ directory (works regardless of where uvicorn is launched from)
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.logger import get_logger
from api.routes.auto_clean import router as auto_clean_router
from api.routes.auto_prepare import router as auto_prepare_router
from api.routes.chat import router as chat_router
from api.routes.documents import router as documents_router
from api.routes.eda_report import router as eda_router
from api.routes.insights import router as insights_router
from api.routes.models import router as models_router
from api.routes.parse_file import router as parse_file_router
from api.routes.predict import router as predict_router
from api.routes.prepare import router as prepare_router
from api.routes.suggest_target import router as suggest_router
from api.routes.train import router as train_router

log = get_logger(__name__)

# ── Rate limiter ────────────────────────────────────────────────────────────
# slowapi pulls keys from get_remote_address — when behind Fly's proxy this
# resolves to the client IP because we pass --proxy-headers --forwarded-allow-ips
# in the Dockerfile CMD.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["240/minute"],
    headers_enabled=True,
)

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="DS-Agent API", version="2.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ── CORS ────────────────────────────────────────────────────────────────────
# Set FRONTEND_ORIGIN to a comma-separated allowlist in production.
# Locally falls back to localhost:3000 so dev still works.
_origins_env = os.environ.get("FRONTEND_ORIGIN", "http://localhost:3000")
_allowed_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
log.info("CORS allow_origins=%s", _allowed_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ── Structured error responses ──────────────────────────────────────────────
def _error_body(code: str, message: str, request_id: str) -> dict:
    return {"error": {"code": code, "message": message, "request_id": request_id}}


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request.state.request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    response = await call_next(request)
    response.headers["x-request-id"] = request.state.request_id
    return response


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    rid = getattr(request.state, "request_id", "")
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(code=f"http_{exc.status_code}", message=str(exc.detail), request_id=rid),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    rid = getattr(request.state, "request_id", "")
    return JSONResponse(
        status_code=422,
        content=_error_body(code="validation_error", message="Request body failed validation", request_id=rid)
        | {"details": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    rid = getattr(request.state, "request_id", "")
    log.exception("Unhandled error [%s] on %s: %s", rid, request.url.path, exc)
    # Generic message — do not leak internals
    return JSONResponse(
        status_code=500,
        content=_error_body(code="internal_error", message="Internal server error", request_id=rid),
    )


# ── Routers ─────────────────────────────────────────────────────────────────
app.include_router(auto_clean_router)
app.include_router(auto_prepare_router)
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(eda_router)
app.include_router(insights_router)
app.include_router(models_router)
app.include_router(parse_file_router)
app.include_router(predict_router)
app.include_router(prepare_router)
app.include_router(suggest_router)
app.include_router(train_router)

# Debug routes — only in dev
if os.environ.get("LOG_LEVEL", "info").lower() == "debug":
    from api.routes.handlers_debug import router as debug_router
    app.include_router(debug_router)


# ── Startup hooks ───────────────────────────────────────────────────────────
# Sweep stale training drafts so unsaved models don't accumulate on disk.
_DRAFT_TTL_SECONDS = int(os.environ.get("DRAFT_TTL_SECONDS", "3600"))


@app.on_event("startup")
async def _cleanup_drafts_on_startup() -> None:
    from api.agents.model_storage import cleanup_old_drafts
    from api.routes.train import cleanup_old_job_files
    try:
        deleted = cleanup_old_drafts(_DRAFT_TTL_SECONDS)
        log.info("Startup draft cleanup: removed=%d ttl=%ds", deleted, _DRAFT_TTL_SECONDS)
    except Exception as exc:  # pragma: no cover — never break startup
        log.warning("Startup draft cleanup failed: %s", exc)
    try:
        deleted = cleanup_old_job_files()
        log.info("Startup job-file cleanup: removed=%d", deleted)
    except Exception as exc:  # pragma: no cover
        log.warning("Startup job-file cleanup failed: %s", exc)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
