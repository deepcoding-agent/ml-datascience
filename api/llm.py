"""LLM factory (cached) and conversation history helpers."""
from __future__ import annotations

import os
from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from api.models import ChatMessage


@lru_cache(maxsize=16)
def _make_llm(api_key: str, model_name: str, temperature: float, max_tokens: int) -> ChatAnthropic:
    """Internal cached constructor — keyed on (api_key, model, temperature, max_tokens)."""
    return ChatAnthropic(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        max_tokens=max_tokens,
    )


def get_llm(temperature: float = 0.0, max_tokens: int = 1024) -> ChatAnthropic:
    """
    Return a cached ChatAnthropic instance for the given temperature and max_tokens.
    Re-reads env vars every call so hot-reloads / config changes are respected,
    but the underlying object is only constructed once per unique (key, model, temp, max_tokens).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    model_name = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    return _make_llm(api_key, model_name, temperature, max_tokens)


def build_lc_history(
    history: list[ChatMessage],
    max_chars_per_msg: int = 1500,
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
