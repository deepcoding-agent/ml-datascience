"""POST /prepare — run the data-preparation pipeline."""
from __future__ import annotations

from fastapi import APIRouter

from api.data_preparation_agent import run_data_preparation
from api.logger import get_logger
from api.models import PrepareRequest, PrepareResponse

router = APIRouter()
log = get_logger(__name__)


@router.post("/prepare", response_model=PrepareResponse)
async def prepare(req: PrepareRequest) -> PrepareResponse:
    log.info(
        "▶ /prepare  dataset='%s'  rows=%d  target='%s'  mode='%s'",
        req.dataset.name, len(req.dataset.data),
        req.target_column, req.mode,
    )
    try:
        result = run_data_preparation(
            data=req.dataset.data,
            target_column=req.target_column,
            mode=req.mode,
            config=req.config,
        )
        if result.get("success"):
            log.info(
                "✔ /prepare done  mode='%s'  train=%d  test=%d  features=%d",
                result.get("mode"), result.get("train_rows", 0),
                result.get("test_rows", 0), result.get("n_features", 0),
            )
        else:
            log.error("✘ /prepare failed: %s", result.get("error"))
        return PrepareResponse(**result)

    except Exception as exc:
        log.exception("✘ /prepare unhandled error: %s", exc)
        return PrepareResponse(
            success=False,
            report=f"## Preparation Failed\n\n**Error:** {exc}",
            error=str(exc),
        )
