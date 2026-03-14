"""LLM factory (cached) and conversation history helpers."""
from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from api.models import ChatMessage


@lru_cache(maxsize=8)
def _make_llm(api_key: str, model_name: str, temperature: float) -> ChatOpenAI:
    """Internal cached constructor — keyed on (api_key, model, temperature)."""
    return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """
    Return a cached ChatOpenAI instance for the given temperature.
    Re-reads env vars every call so hot-reloads / config changes are respected,
    but the underlying object is only constructed once per unique (key, model, temp).
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return _make_llm(api_key, model_name, temperature)


def build_lc_history(history: list[ChatMessage]) -> list:
    """Convert a list of ChatMessage dicts into LangChain message objects."""
    mapping = {
        "user":      HumanMessage,
        "assistant": AIMessage,
        "system":    SystemMessage,
    }
    return [
        mapping[m.role](content=m.content)
        for m in history
        if m.role in mapping
    ]
