"""GET /models — list available LLM models with provider availability."""
from __future__ import annotations

import os

from fastapi import APIRouter

from api.llm import get_default_model_id
from api.models import ModelInfo

router = APIRouter()

ALL_MODELS = [
    ModelInfo(id="gpt-4o-mini",       label="GPT-4o mini",    provider="openai",    badge="Fast"),
    ModelInfo(id="gpt-4o",            label="GPT-4o",         provider="openai",    badge="Smart"),
    ModelInfo(id="claude-haiku-4-5",  label="Claude Haiku",   provider="anthropic", badge="Fast"),
    ModelInfo(id="claude-sonnet-4-5", label="Claude Sonnet",  provider="anthropic", badge="Smart"),
    ModelInfo(id="claude-opus-4-5",   label="Claude Opus",    provider="anthropic", badge="Powerful"),
]


@router.get("/models")
def get_available_models() -> dict:
    """Returns all models with availability based on configured API keys."""
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))

    result = []
    for model in ALL_MODELS:
        m = model.model_copy()
        m.available = (
            has_openai if model.provider == "openai"
            else has_anthropic
        )
        result.append(m)

    return {"models": result, "default": get_default_model_id()}
