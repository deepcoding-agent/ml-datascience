"""POST /documents — comprehensive EDA document report."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from api.agents.document_agent import run_document
from api.logger import get_logger

router = APIRouter()
log = get_logger(__name__)


class DocumentRequest(BaseModel):
    dataset_name: str
    data: list[dict]
    model_id: str | None = None


class DocumentResponse(BaseModel):
    success: bool
    dataset_name: str = ""
    document: dict = {}
    column_profiles: list[dict] = []
    analysis: dict = {}
    duration_seconds: float = 0.0
    error: str = ""


@router.post("/documents", response_model=DocumentResponse)
async def documents(req: DocumentRequest) -> DocumentResponse:
    log.info(">>> /documents  dataset='%s'  rows=%d", req.dataset_name, len(req.data))
    try:
        result = run_document(
            data=req.data,
            dataset_name=req.dataset_name,
            model_id=req.model_id,
        )
        log.info(
            "<<< /documents done  quality=%s  %.1fs",
            result["document"].get("quality_score"),
            result["duration_seconds"],
        )
        return DocumentResponse(**result)
    except Exception as exc:
        log.exception("/documents error: %s", exc)
        return DocumentResponse(success=False, error=str(exc))
