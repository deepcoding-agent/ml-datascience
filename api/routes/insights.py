"""POST /insights — AI-powered dataset insights report."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from api.agents.insights_agent import run_insights
from api.logger import get_logger

router = APIRouter()
log = get_logger(__name__)


class InsightsRequest(BaseModel):
    dataset_name: str
    data: list[dict]
    model_id: str | None = None


class InsightsResponse(BaseModel):
    success: bool
    dataset_name: str = ""
    report: str = ""
    ai_insights: dict = {}
    analysis: dict = {}
    duration_seconds: float = 0.0
    error: str = ""


@router.post("/insights", response_model=InsightsResponse)
async def insights(req: InsightsRequest) -> InsightsResponse:
    log.info(">>> /insights  dataset='%s'  rows=%d", req.dataset_name, len(req.data))
    try:
        result = run_insights(
            data=req.data,
            dataset_name=req.dataset_name,
            model_id=req.model_id,
        )
        log.info(
            "<<< /insights done  quality=%s  %.1fs",
            result["ai_insights"].get("data_quality_score"),
            result["duration_seconds"],
        )
        return InsightsResponse(**result)
    except Exception as exc:
        log.exception("/insights error: %s", exc)
        return InsightsResponse(
            success=False,
            error=str(exc),
            report=f"## Insights Failed\n\n**Error:** {exc}",
        )
