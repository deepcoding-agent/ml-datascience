"""POST /feature/analyze + POST /feature — manual + AI-assisted feature engineering.

`/feature/analyze`  — called by the setup panel on mount; returns the AI-
suggested FeatureConfig + per-column profile so the panel can pre-fill knobs.

`/feature`          — runs the pipeline with the FeatureConfig the user
confirmed in the panel. Returns one engineered dataset plus the report card
artifact; if config.split == "train_test", also returns the split halves.
"""
from __future__ import annotations

from fastapi import APIRouter

from api.agents.feature_agent import analyze, run
from api.logger import get_logger
from api.models import (
    FeatureAnalyzeRequest,
    FeatureAnalyzeResponse,
    FeatureReportArtifact,
    FeatureRequest,
    FeatureResponse,
)

router = APIRouter()
log = get_logger(__name__)


def _slug(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")
    return keep or "engineered"


@router.post("/feature/analyze", response_model=FeatureAnalyzeResponse)
async def feature_analyze(req: FeatureAnalyzeRequest) -> FeatureAnalyzeResponse:
    log.info(">>> /feature/analyze dataset='%s' rows=%d", req.dataset_name, len(req.data))
    if not req.data:
        return FeatureAnalyzeResponse(success=False, error="dataset is empty")
    try:
        suggestion = analyze(req.data, model_id=req.model_id)
        log.info(
            "<<< /feature/analyze ok task=%s target=%s derived=%d",
            suggestion.detected_task,
            suggestion.config.target_column,
            len(suggestion.derived_candidates),
        )
        return FeatureAnalyzeResponse(success=True, suggestion=suggestion)
    except Exception as exc:
        log.exception("/feature/analyze error: %s", exc)
        return FeatureAnalyzeResponse(success=False, error=str(exc))


@router.post("/feature", response_model=FeatureResponse)
async def feature(req: FeatureRequest) -> FeatureResponse:
    log.info(
        ">>> /feature dataset='%s' rows=%d task=%s target=%s split=%s",
        req.dataset_name,
        len(req.data),
        req.config.task,
        req.config.target_column or "(none)",
        req.config.split,
    )
    if not req.data:
        return FeatureResponse(
            success=False,
            error="dataset is empty",
            report=FeatureReportArtifact(success=False, error="dataset is empty"),
        )

    try:
        engineered_df, report, train_df, test_df = run(
            data=req.data,
            config=req.config,
        )

        base = _slug(req.dataset_name)
        out_name = f"{base}_features"
        rows = engineered_df.to_dict(orient="records")

        resp = FeatureResponse(
            success=report.success,
            dataset_name=out_name,
            columns=list(engineered_df.columns),
            rows=rows,
            report=report,
            error=report.error,
        )
        if train_df is not None and test_df is not None:
            resp.has_split = True
            resp.train_rows = train_df.to_dict(orient="records")
            resp.test_rows = test_df.to_dict(orient="records")

        log.info(
            "<<< /feature ok rows %d→%d cols %d→%d steps=%d split=%s",
            report.rows_before,
            report.rows_after,
            report.cols_before,
            report.cols_after,
            len(report.steps),
            resp.has_split,
        )
        return resp
    except Exception as exc:
        log.exception("/feature error: %s", exc)
        return FeatureResponse(
            success=False,
            error=str(exc),
            report=FeatureReportArtifact(success=False, error=str(exc)),
        )
