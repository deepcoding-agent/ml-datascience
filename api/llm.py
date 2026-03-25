"""LLM factory (cached, multi-provider) and conversation history helpers."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from api.models import ChatMessage

OPENAI_MODELS = {"gpt-4o-mini", "gpt-4o"}
ANTHROPIC_MODELS = {"claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5",
                    "claude-sonnet-4-6", "claude-opus-4-6"}


@lru_cache(maxsize=16)
def _make_llm(
    provider: str, api_key: str, model_name: str, temperature: float, max_tokens: int,
) -> Union["ChatOpenAI", "ChatAnthropic"]:  # type: ignore[name-defined]
    """Internal cached constructor — keyed on (provider, key, model, temp, max_tokens)."""
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name, temperature=temperature,
            api_key=api_key, max_tokens=max_tokens,
        )
    else:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name, temperature=temperature,
            api_key=api_key, max_tokens=max_tokens,
        )


def get_llm(
    temperature: float = 0.0,
    max_tokens: int = 1024,
    model_id: str | None = None,
) -> Union["ChatOpenAI", "ChatAnthropic"]:  # type: ignore[name-defined]
    """
    Return a cached LLM instance. Auto-detects provider from model_id.
    Falls back to env var defaults if model_id is not provided.
    """
    active_model = model_id or get_default_model_id()

    if active_model in OPENAI_MODELS:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return _make_llm("openai", api_key, active_model, temperature, max_tokens)
    elif active_model in ANTHROPIC_MODELS:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        return _make_llm("anthropic", api_key, active_model, temperature, max_tokens)
    else:
        raise ValueError(
            f"Unknown model_id: {active_model}. "
            f"Supported: {OPENAI_MODELS | ANTHROPIC_MODELS}"
        )


def get_default_model_id() -> str:
    """Returns the default model ID from environment."""
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def build_lc_history(
    history: list[ChatMessage],
    max_chars_per_msg: int = 3000,
) -> list:
    """Convert ChatMessage list into LangChain message objects.

    Long assistant messages are truncated to *max_chars_per_msg* so they
    don't blow up the context window on follow-up turns.
    """
    mapping = {
        "user":      HumanMessage,
        "assistant": AIMessage,
        "system":    SystemMessage,
    }
    result = []
    for m in history:
        if m.role not in mapping:
            continue
        content = m.content
        if m.role == "assistant" and len(content) > max_chars_per_msg:
            content = content[:max_chars_per_msg] + "\n... (truncated)"
        result.append(mapping[m.role](content=content))
    return result
