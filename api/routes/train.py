"""Training routes — POST /train, GET /train/models, GET /train/models/{id}/download."""
from __future__ import annotations

import io
import json
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import joblib
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.agents.model_storage import (
    MODELS_DIR,
    delete_model,
    get_model_path,
    list_library_models,
    list_models,
    promote_draft,
    rename_model,
    set_library_flag,
)
from api.agents.train_agent import run_training
from api.agents.train_setup_agent import analyze_train
from api.llm import get_default_model_id
from api.logger import get_logger
from api.models import TrainAnalyzeRequest, TrainAnalyzeResponse

router = APIRouter()
log = get_logger(__name__)


# ── POST /train/analyze — AI-suggested training setup ──────────────────────

@router.post("/train/analyze", response_model=TrainAnalyzeResponse)
def train_analyze(req: TrainAnalyzeRequest) -> TrainAnalyzeResponse:
    model_id = req.model_id or get_default_model_id()
    log.info(">>> /train/analyze  model=%s  dataset='%s' rows=%d",
             model_id, req.dataset_name, len(req.data))
    if not req.data:
        return TrainAnalyzeResponse(success=False, error="dataset is empty")
    try:
        suggestion = analyze_train(req.data, model_id=req.model_id)
        log.info(
            "<<< /train/analyze ok target=%s task=%s scoring=%s cv=%d trials=%d test=%.2f",
            suggestion.target_column,
            suggestion.task_type,
            suggestion.scoring_metric,
            suggestion.cv_folds,
            suggestion.tun_trials,
            suggestion.test_size,
        )
        return TrainAnalyzeResponse(success=True, suggestion=suggestion)
    except Exception as exc:
        log.exception("/train/analyze error: %s", exc)
        return TrainAnalyzeResponse(success=False, error=str(exc))

# Tight limit on /train — Optuna tuning is expensive (CPU + LLM narrative).
limiter = Limiter(key_func=get_remote_address)


# ── Request / Response models ─────────────────────────────────────────────────

class TrainRequest(BaseModel):
    rows: list[dict]
    columns: list[str]
    target_column: str | None = None
    task_type: str = "auto"
    algorithms: str | list[str] = "auto"
    cv_folds: int = 5
    tune_trials: int = 20
    test_size: float = 0.2
    model_id: str | None = None          # LLM model for AI narrative
    conversation_id: str = ""
    dataset_id: str = ""
    # When the training dataset was produced by /feature, this is the id of
    # the saved feature pipeline /predict will replay on raw input later.
    feature_pipeline_id: str = ""


class TrainResponse(BaseModel):
    success: bool
    model_id: str = ""
    best_algorithm: str = ""
    best_algorithm_display: str = ""
    task_type: str = ""
    target_column: str = ""
    metrics: dict = {}
    comparison_table: list[dict] = []
    charts: list[dict] = []
    feature_importance: list[dict] = []
    classification_report: str = ""
    ai_summary: str = ""
    download_url: str = ""
    dataset_shape: list[int] = []
    training_duration: float = 0.0
    # Surface of the auto feature-engineering step so the UI can show what was applied.
    feature_engineering: dict = {}
    error: str = ""


# ── POST /train ──────────────────────────────────────────────────────────────

@router.post("/train", response_model=TrainResponse)
@limiter.limit("6/minute")
def train(request: Request, req: TrainRequest) -> TrainResponse:
    model_id = req.model_id or get_default_model_id()
    log.info(">>> /train  model=%s  rows=%d  cols=%d  target=%s  task=%s",
             model_id, len(req.rows), len(req.columns), req.target_column, req.task_type)
    try:
        result = run_training(
            data=req.rows,
            columns=req.columns,
            target_column=req.target_column,
            task_type=req.task_type,
            algorithms=req.algorithms,
            cv_folds=req.cv_folds,
            tune_trials=req.tune_trials,
            test_size=req.test_size,
            model_id=req.model_id,
            conversation_id=req.conversation_id,
            dataset_id=req.dataset_id,
            feature_pipeline_id=req.feature_pipeline_id,
        )
        log.info("<<< /train done  best=%s  duration=%.1fs",
                 result.get("best_algorithm_display", "N/A"),
                 result.get("training_duration", 0))
        return TrainResponse(**result)
    except Exception as exc:
        log.exception("/train error: %s", exc)
        return TrainResponse(success=False, error=str(exc))


# ── Async training (start + poll) ───────────────────────────────────────────
# In-flight jobs live in TRAINING_JOBS (memory). Completed jobs are also
# persisted to disk so a server restart (e.g., --reload) doesn't lose results
# that finished just before the restart — the client can still poll for them.

# Static weights used to convert phase progress → percent + ETA. Tune these
# if the pipeline phase split changes significantly.
_PHASE_WEIGHTS: dict[str, float] = {
    "preprocessing": 5.0,
    "cv": 25.0,           # baseline CV across all algos
    "tuning": 55.0,       # tuning top 3 — dominant cost
    "evaluation": 8.0,    # per-algo test-set eval + re-sort
    "charts": 4.0,
    "finalizing": 3.0,    # save + AI summary
}
_PHASE_ORDER = list(_PHASE_WEIGHTS.keys())

TRAINING_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()

# Where completed job snapshots are persisted (sibling of MODELS_DIR).
_JOBS_DISK_DIR = MODELS_DIR.parent / "training_jobs"
_JOBS_DISK_DIR.mkdir(parents=True, exist_ok=True)
# Persisted jobs older than this are swept on next startup poll.
_JOB_FILE_TTL_SECONDS = 24 * 60 * 60


def _persist_job_to_disk(job_id: str, snap: dict[str, Any]) -> None:
    """Write a completed job snapshot to disk so it survives a server restart."""
    try:
        path = _JOBS_DISK_DIR / f"{job_id}.json"
        path.write_text(json.dumps(snap, indent=2, default=str))
    except Exception as exc:
        log.warning("Failed to persist job %s: %s", job_id, exc)


def _load_persisted_job(job_id: str) -> dict[str, Any] | None:
    """Read a job snapshot from disk if it exists, else None."""
    path = _JOBS_DISK_DIR / f"{job_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        log.warning("Failed to load persisted job %s: %s", job_id, exc)
        return None


def cleanup_old_job_files(ttl_seconds: int = _JOB_FILE_TTL_SECONDS) -> int:
    """Delete persisted job snapshots older than ttl_seconds. Returns count."""
    cutoff = time.time() - ttl_seconds
    deleted = 0
    for path in _JOBS_DISK_DIR.glob("*.json"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                deleted += 1
        except OSError:
            continue
    if deleted:
        log.info("Cleaned up %d old training job file(s)", deleted)
    return deleted


def _phase_pct(phase: str, sub_progress: float = 0.0) -> float:
    """Cumulative percent at the START of `phase`, plus its weight × sub_progress."""
    done = sum(_PHASE_WEIGHTS[p] for p in _PHASE_ORDER if _PHASE_ORDER.index(p) < _PHASE_ORDER.index(phase))
    return min(99.0, done + _PHASE_WEIGHTS.get(phase, 0.0) * max(0.0, min(1.0, sub_progress)))


def _make_progress_callback(job_id: str):
    """Returns a callback that mutates TRAINING_JOBS[job_id] on every emit."""
    state = {"current": 0, "total": 0}  # carries sub-phase counters across calls

    def _cb(event: dict[str, Any]) -> None:
        phase = event.get("phase", "")
        elapsed = float(event.get("elapsed", 0.0))

        # Compute sub-phase progress for phases that come with current/total info.
        sub = 0.0
        if phase == "tuning":
            state["current"] = int(event.get("current", 0))
            state["total"] = int(event.get("total", 0))
            if state["total"]:
                sub = (state["current"] - 1) / state["total"]
        elif phase == "cv_done":
            phase = "cv"
            sub = 1.0

        pct = _phase_pct(phase if phase in _PHASE_WEIGHTS else "finalizing", sub)
        eta = (elapsed * (100.0 - pct) / pct) if pct > 1 else None

        with _JOBS_LOCK:
            job = TRAINING_JOBS.get(job_id)
            if job is None:
                return
            job.update({
                "phase": phase,
                "elapsed": elapsed,
                "percent": round(pct, 1),
                "eta_seconds": round(eta, 1) if eta is not None else None,
                "sub_current": state["current"] or None,
                "sub_total": state["total"] or None,
                "extra": {k: v for k, v in event.items() if k not in ("phase", "elapsed")},
            })

    return _cb


def _run_job(job_id: str, req: "TrainRequest") -> None:
    """Background worker. Updates TRAINING_JOBS[job_id] in-place and persists
    the final snapshot to disk so a server restart doesn't lose the result."""
    cb = _make_progress_callback(job_id)
    try:
        result = run_training(
            data=req.rows, columns=req.columns,
            target_column=req.target_column, task_type=req.task_type,
            algorithms=req.algorithms, cv_folds=req.cv_folds,
            tune_trials=req.tune_trials, test_size=req.test_size,
            model_id=req.model_id, conversation_id=req.conversation_id,
            dataset_id=req.dataset_id,
            feature_pipeline_id=req.feature_pipeline_id,
            progress_callback=cb,
        )
        with _JOBS_LOCK:
            job = TRAINING_JOBS.get(job_id)
            if job is not None:
                job.update({
                    "complete": True, "percent": 100.0, "eta_seconds": 0.0,
                    "result": result,
                    "phase": "done" if result.get("success") else "error",
                    "error": result.get("error", "") if not result.get("success") else "",
                })
                snap = dict(job)
        _persist_job_to_disk(job_id, snap)
    except Exception as exc:  # pragma: no cover — defensive
        log.exception("Background train job %s failed: %s", job_id, exc)
        with _JOBS_LOCK:
            job = TRAINING_JOBS.get(job_id)
            if job is not None:
                job.update({
                    "complete": True, "phase": "error",
                    "error": str(exc), "result": None,
                })
                snap = dict(job)
        if "snap" in locals():
            _persist_job_to_disk(job_id, snap)


@router.post("/train/start")
async def train_start(req: TrainRequest, background_tasks: BackgroundTasks) -> dict:
    """Start training in the background. Returns a job_id to poll for progress."""
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        TRAINING_JOBS[job_id] = {
            "job_id": job_id,
            "phase": "queued",
            "elapsed": 0.0,
            "percent": 0.0,
            "eta_seconds": None,
            "sub_current": None,
            "sub_total": None,
            "extra": {},
            "complete": False,
            "result": None,
            "error": "",
            "started_at": time.time(),
        }
    log.info(">>> /train/start  job=%s  rows=%d", job_id, len(req.rows))
    background_tasks.add_task(_run_job, job_id, req)
    return {"job_id": job_id}


@router.get("/train/progress/{job_id}")
async def train_progress(job_id: str) -> dict:
    """Poll a training job. Returns current phase, percent, ETA, and the
    full result (when complete) so the client can swap to the result UI.

    Lookup order: in-memory TRAINING_JOBS → disk-persisted snapshot. The disk
    fallback means a server restart right after training (e.g., uvicorn
    --reload picking up a code change) doesn't strand the client.

    elapsed/ETA are recomputed on every poll so the timer keeps ticking even
    when no emit happened between polls (e.g., during a long Optuna run).
    """
    with _JOBS_LOCK:
        job = TRAINING_JOBS.get(job_id)
        snap = dict(job) if job is not None else None

    if snap is None:
        # Restart-survivor path: a completed job persisted to disk.
        snap = _load_persisted_job(job_id)
        if snap is None:
            raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")

    if not snap["complete"]:
        live_elapsed = time.time() - snap["started_at"]
        snap["elapsed"] = round(live_elapsed, 2)
        pct = snap.get("percent") or 0.0
        snap["eta_seconds"] = round(live_elapsed * (100.0 - pct) / pct, 1) if pct > 1 else None

    # If complete and old, garbage collect the in-memory entry (disk file stays
    # until the periodic cleanup_old_job_files sweep).
    if snap["complete"] and time.time() - snap["started_at"] > 600:
        with _JOBS_LOCK:
            TRAINING_JOBS.pop(job_id, None)
    return snap


# ── GET /train/models/library/all ───────────────────────────────────────────

@router.get("/train/models/library/all")
async def list_library() -> list[dict]:
    """All models marked as saved_to_library (across every conversation)."""
    log.info(">>> /train/models/library/all")
    return list_library_models()


# ── GET /train/models/{conversation_id} ─────────────────────────────────────

@router.get("/train/models/{conversation_id}")
async def get_models(conversation_id: str) -> list[dict]:
    log.info(">>> /train/models  conversation=%s", conversation_id)
    return list_models(conversation_id)


# ── GET /train/models/{model_id}/download ────────────────────────────────────

@router.get("/train/models/{model_id}/download")
async def download_model(model_id: str) -> FileResponse:
    log.info(">>> /train/models/%s/download", model_id)
    path = get_model_path(model_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(
        path=str(path),
        media_type="application/octet-stream",
        filename=f"model_{model_id}.joblib",
    )


# ── PATCH /train/models/{model_id}/rename ───────────────────────────────────

class RenameModelRequest(BaseModel):
    display_name: str


@router.patch("/train/models/{model_id}/rename")
async def rename_model_route(model_id: str, req: RenameModelRequest) -> dict:
    log.info(">>> /train/models/%s/rename -> %s", model_id, req.display_name)
    try:
        return rename_model(model_id, req.display_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ── PATCH /train/models/{model_id}/library ──────────────────────────────────

class LibraryFlagRequest(BaseModel):
    saved: bool


@router.patch("/train/models/{model_id}/library")
async def set_library_route(model_id: str, req: LibraryFlagRequest) -> dict:
    log.info(">>> /train/models/%s/library -> %s", model_id, req.saved)
    try:
        return set_library_flag(model_id, req.saved)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ── DELETE /train/models/{model_id} ─────────────────────────────────────────

@router.delete("/train/models/{model_id}")
async def delete_model_route(model_id: str) -> dict:
    log.info(">>> DELETE /train/models/%s", model_id)
    deleted = delete_model(model_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"deleted": True, "model_id": model_id}


# ── POST /train/models/{model_id}/save ──────────────────────────────────────
# Promote a draft (a model the training pipeline produced but the user has not
# yet kept) so it appears in conversation/library listings.

@router.post("/train/models/{model_id}/save")
async def save_draft_route(model_id: str) -> dict:
    log.info(">>> /train/models/%s/save", model_id)
    try:
        return promote_draft(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ── POST /train/models/upload ───────────────────────────────────────────────
# Strict validation: bundle must contain model + pipeline + feature_columns +
# target_column + task_type. Uploaded models go straight into the library and
# are NOT linked to a conversation.

_REQUIRED_BUNDLE_KEYS = {"model", "pipeline", "feature_columns", "target_column", "task_type"}


@router.post("/train/models/upload")
async def upload_model_route(
    file: UploadFile = File(...),
    conversation_id: str = Form(""),
) -> dict:
    filename = file.filename or "uploaded.joblib"
    if not filename.lower().endswith(".joblib"):
        raise HTTPException(status_code=400, detail="File must be a .joblib bundle")

    raw = await file.read()
    try:
        bundle = joblib.load(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to load joblib: {exc}")

    if not isinstance(bundle, dict):
        raise HTTPException(status_code=422, detail="Bundle must be a dict")
    missing = _REQUIRED_BUNDLE_KEYS - set(bundle.keys())
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Bundle missing required keys: {sorted(missing)}. Required: {sorted(_REQUIRED_BUNDLE_KEYS)}",
        )

    model_id = str(uuid.uuid4())
    model_path = MODELS_DIR / f"{model_id}.joblib"
    meta_path = MODELS_DIR / f"{model_id}.json"

    with open(model_path, "wb") as fh:
        shutil.copyfileobj(io.BytesIO(raw), fh)

    metadata = {
        "model_id": model_id,
        # Link to the user's current chat so the upload appears in their
        # Models tab right away. The library flag also makes it discoverable
        # cross-conversation. Empty string is fine (library-only model).
        "conversation_id": conversation_id or "",
        "dataset_id": "",
        "task_type": str(bundle.get("task_type", "")),
        "algorithm": "uploaded",
        "algorithm_display": filename.rsplit(".", 1)[0],
        "target_column": str(bundle.get("target_column", "")),
        "feature_columns": list(bundle.get("feature_columns", [])),
        "metrics": {},
        "hyperparameters": {},
        "training_duration_seconds": 0.0,
        "dataset_shape": [0, len(bundle.get("feature_columns", []))],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_file": str(model_path),
        "is_draft": False,
        "saved_to_library": True,
        "uploaded": True,
        "display_name": filename.rsplit(".", 1)[0],
    }
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    log.info("Uploaded model %s from %s (%d bytes) → conv=%s",
             model_id, filename, len(raw), conversation_id or "<library>")
    return metadata
