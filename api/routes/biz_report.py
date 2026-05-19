"""POST /biz-report — business strategy report (recommendations, KPIs, roadmap)."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from api.agents.biz_report_agent import run_biz_report
from api.llm import get_default_model_id
from api.logger import get_logger

router = APIRouter()
log = get_logger(__name__)


class BizReportRequest(BaseModel):
    dataset_name: str
    data: list[dict]
    model_id: str | None = None


class BizReportResponse(BaseModel):
    success: bool
    dataset_name: str = ""
    document: dict = {}
    column_profiles: list[dict] = []
    analysis: dict = {}
    duration_seconds: float = 0.0
    error: str = ""


@router.post("/biz-report", response_model=BizReportResponse)
def biz_report(req: BizReportRequest) -> BizReportResponse:
    model_id = req.model_id or get_default_model_id()
    log.info(">>> /biz-report  model=%s  dataset='%s'  rows=%d",
             model_id, req.dataset_name, len(req.data))
    try:
        result = run_biz_report(
            data=req.data,
            dataset_name=req.dataset_name,
            model_id=req.model_id,
        )
        log.info(
            "<<< /biz-report done  quality=%s  %.1fs",
            result["document"].get("quality_score"),
            result["duration_seconds"],
        )
        return BizReportResponse(**result)
    except Exception as exc:
        log.exception("/biz-report error: %s", exc)
        return BizReportResponse(success=False, error=str(exc))
