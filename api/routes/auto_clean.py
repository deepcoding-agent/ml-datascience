"""POST /auto-clean — AI-driven automatic dataset cleaning."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from api.agents.auto_cleaner import run_auto_clean
from api.logger import get_logger

router = APIRouter()
log = get_logger(__name__)


class AutoCleanRequest(BaseModel):
    dataset_name: str
    data: list[dict]
    model_id: str | None = None


class AutoCleanResponse(BaseModel):
    success: bool
    cleaned_data: list[dict] = []
    columns: list[str] = []
    report: str = ""
    operations: list[dict] = []
    original_shape: dict = {}
    cleaned_shape: dict = {}
    error: str = ""


@router.post("/auto-clean", response_model=AutoCleanResponse)
async def auto_clean(req: AutoCleanRequest) -> AutoCleanResponse:
    log.info(">>> /auto-clean  dataset='%s'  rows=%d", req.dataset_name, len(req.data))
    try:
        result = run_auto_clean(data=req.data, model_id=req.model_id)
        log.info(
            "<<< /auto-clean done  %s -> %s  ops=%d",
            result["original_shape"], result["cleaned_shape"],
            len(result["operations"]),
        )
        return AutoCleanResponse(**result)
    except Exception as exc:
        log.exception("/auto-clean error: %s", exc)
        return AutoCleanResponse(
            success=False,
            error=str(exc),
            report=f"## Auto-Clean Failed\n\n**Error:** {exc}",
        )
