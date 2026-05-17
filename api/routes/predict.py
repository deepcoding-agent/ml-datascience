"""Prediction route — POST /predict reuses a saved model on new rows."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from api.agents.predict_agent import run_prediction
from api.logger import get_logger

router = APIRouter()
log = get_logger(__name__)


class PredictRequest(BaseModel):
    model_id: str
    rows: list[dict]
    columns: list[str]
    dataset_id: str = ""
    conversation_id: str = ""


class PredictResponse(BaseModel):
    success: bool
    model_id: str = ""
    task_type: str = ""
    target_column: str = ""
    prediction_column: str = ""
    rows: list[dict] = []
    columns: list[str] = []
    summary: dict = {}
    error: str = ""


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    log.info(">>> /predict  model=%s  rows=%d", req.model_id, len(req.rows))
    try:
        result = run_prediction(req.model_id, req.rows, req.columns)
        log.info(
            "<<< /predict done  success=%s  predictions=%d",
            result.get("success"),
            len(result.get("rows", [])),
        )
        return PredictResponse(**result)
    except Exception as exc:
        log.exception("/predict error: %s", exc)
        return PredictResponse(success=False, error=str(exc))
