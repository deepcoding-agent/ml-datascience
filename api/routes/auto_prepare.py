"""POST /auto-prepare — AI-driven ML data preparation."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from api.agents.auto_prepare import run_auto_prepare
from api.logger import get_logger

router = APIRouter()
log = get_logger(__name__)


class AutoPrepareRequest(BaseModel):
    dataset_name: str
    data: list[dict]
    target_column: str | None = None
    model_id: str | None = None


@router.post("/auto-prepare")
async def auto_prepare(req: AutoPrepareRequest) -> dict:
    log.info(
        ">>> /auto-prepare  dataset='%s'  rows=%d  target='%s'",
        req.dataset_name, len(req.data), req.target_column or "auto",
    )
    try:
        result = run_auto_prepare(
            data=req.data,
            target_column=req.target_column,
            model_id=req.model_id,
        )
        log.info(
            "<<< /auto-prepare done  success=%s  features=%d  train=%d  test=%d",
            result.get("success"), result.get("n_features", 0),
            result.get("train_rows", 0), result.get("test_rows", 0),
        )
        return result
    except Exception as exc:
        log.exception("/auto-prepare error: %s", exc)
        return {
            "success": False,
            "error": str(exc),
            "report": f"## Auto-Prepare Failed\n\n**Error:** {exc}",
        }
