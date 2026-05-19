"""POST /chat — route the request to the coding or data-science agent."""
from __future__ import annotations

from fastapi import APIRouter, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.agents.coding import run_coding_agent
from api.agents.datascience import run_datascience_agent
from api.llm import get_default_model_id
from api.logger import get_logger
from api.models import ChatRequest, ChatResponse

router = APIRouter()
log = get_logger(__name__)

# Rate limit /chat — bounds LLM spend if someone hammers the endpoint.
limiter = Limiter(key_func=get_remote_address)


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("60/minute")
def chat(request: Request, req: ChatRequest) -> ChatResponse:
    model_id = req.model_id or get_default_model_id()
    mode = "ds-agent" if req.datasets else "coding-agent"
    log.info(
        "▶ /chat  mode=%s  model=%s  datasets=%s  message='%s'",
        mode, model_id,
        [d.name for d in req.datasets],
        req.message[:120],
    )
    try:
        if req.datasets:
            text, artifacts = run_datascience_agent(
                req.message, req.datasets, req.conversation_history,
                model_id=model_id,
            )
        else:
            text = run_coding_agent(
                req.message, req.conversation_history,
                model_id=model_id,
            )
            artifacts = {}

        output_type = artifacts.pop("output_type", "text")
        should_activate = artifacts.pop("should_activate", False)
        log.info("✔ /chat done  model=%s  output_type=%s  artifact_keys=%s", model_id, output_type, list(artifacts.keys()))
        return ChatResponse(
            response=text, artifacts=artifacts,
            output_type=output_type, should_activate=should_activate,
            model_used=model_id,
        )

    except Exception as exc:
        log.exception("✘ /chat unhandled error: %s", exc)
        return ChatResponse(
            response=f"Agent error: {exc}",
            artifacts={"error": str(exc)},
            model_used=model_id,
        )
