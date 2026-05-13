"""Training routes — POST /train, GET /train/models, GET /train/models/{id}/download."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.agents.model_storage import (
    delete_model,
    get_model_path,
    list_library_models,
    list_models,
    rename_model,
    set_library_flag,
)
from api.agents.train_agent import run_training
from api.logger import get_logger

router = APIRouter()
log = get_logger(__name__)


# ── Request / Response models ─────────────────────────────────────────────────

class TrainRequest(BaseModel):
    rows: list[dict]
    columns: list[str]
    target_column: str | None = None
    task_type: str = "auto"
    algorithms: str | list[str] = "auto"
    cv_folds: int = 5
    tune_trials: int = 50
    test_size: float = 0.2
    model_id: str | None = None          # LLM model for AI narrative
    conversation_id: str = ""
    dataset_id: str = ""


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
    error: str = ""


# ── POST /train ──────────────────────────────────────────────────────────────

@router.post("/train", response_model=TrainResponse)
async def train(req: TrainRequest) -> TrainResponse:
    log.info(">>> /train  rows=%d  cols=%d  target=%s  task=%s",
             len(req.rows), len(req.columns), req.target_column, req.task_type)
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
        )
        log.info("<<< /train done  best=%s  duration=%.1fs",
                 result.get("best_algorithm_display", "N/A"),
                 result.get("training_duration", 0))
        return TrainResponse(**result)
    except Exception as exc:
        log.exception("/train error: %s", exc)
        return TrainResponse(success=False, error=str(exc))


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
