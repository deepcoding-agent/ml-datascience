"""POST /chat — route the request to the coding or data-science agent."""
from __future__ import annotations

from fastapi import APIRouter

from api.agents.coding import run_coding_agent
from api.agents.datascience import run_datascience_agent
from api.logger import get_logger
from api.models import ChatRequest, ChatResponse

router = APIRouter()
log = get_logger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    mode = "ds-agent" if req.datasets else "coding-agent"
    log.info(
        "▶ /chat  mode=%s  datasets=%s  message='%s'",
        mode,
        [d.name for d in req.datasets],
        req.message[:120],
    )
    try:
        if req.datasets:
            text, artifacts = run_datascience_agent(
                req.message, req.datasets, req.conversation_history
            )
        else:
            text = run_coding_agent(req.message, req.conversation_history)
            artifacts = {}

        log.info("✔ /chat done  artifact_keys=%s", list(artifacts.keys()))
        return ChatResponse(response=text, artifacts=artifacts)

    except Exception as exc:
        log.exception("✘ /chat unhandled error: %s", exc)
        return ChatResponse(
            response=f"Agent error: {exc}",
            artifacts={"error": str(exc)},
        )
