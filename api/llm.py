"""LLM factory (cached, multi-provider) and conversation history helpers."""
from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from api.logger import get_logger
from api.models import ChatMessage

log = get_logger(__name__)

OPENAI_MODELS = {"gpt-5", "gpt-5-mini", "gpt-5.4-nano"}
ANTHROPIC_MODELS = {"claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5",
                    "claude-sonnet-4-6", "claude-opus-4-6", "claude-opus-4-7"}

# Hard fallback when an unknown / retired model_id is requested. Matches the
# project default (Claude Haiku 4.5) — fastest model in the picker with strong
# instruction following and ANTHROPIC_API_KEY is always provisioned on Render.
FALLBACK_MODEL_ID = "claude-haiku-4-5"

# Cross-provider failover chain for long-running narrators (/eda, /biz-report).
# When the primary model returns a transient error (Anthropic 529 overloaded,
# 503, timeout, network reset), we walk this chain until one succeeds. Mixing
# providers means a single-provider outage doesn't take the agent down.
MODEL_FAILOVER_CHAIN = [
    "claude-haiku-4-5",   # fastest Anthropic, lowest cost
    "gpt-5.4-nano",       # fastest OpenAI, cross-provider safety
    "claude-sonnet-4-6",  # same provider but bigger / more headroom
    "gpt-5-mini",         # final cross-provider attempt
]

# Models where the `temperature` parameter has been deprecated by the provider.
# Sending it returns 400 invalid_request_error. Newer reasoning-style models
# (Anthropic 4-6+/Opus 4-7, OpenAI GPT-5 family) use fixed internal sampling.
NO_TEMPERATURE_MODELS = {
    # Anthropic (extended-thinking generation)
    "claude-opus-4-6", "claude-opus-4-7", "claude-sonnet-4-6",
    # OpenAI GPT-5 family
    "gpt-5", "gpt-5-mini", "gpt-5.4-nano",
}


@lru_cache(maxsize=16)
def _make_llm(
    provider: str, api_key: str, model_name: str, temperature: float, max_tokens: int,
) -> Union["ChatOpenAI", "ChatAnthropic"]:  # type: ignore[name-defined]
    """Internal cached constructor — keyed on (provider, key, model, temp, max_tokens)."""
    accepts_temp = model_name not in NO_TEMPERATURE_MODELS
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs: dict = {"model": model_name, "api_key": api_key, "max_tokens": max_tokens}
        if accepts_temp:
            kwargs["temperature"] = temperature
        return ChatOpenAI(**kwargs)
    else:
        from langchain_anthropic import ChatAnthropic
        kwargs = {"model": model_name, "api_key": api_key, "max_tokens": max_tokens}
        if accepts_temp:
            kwargs["temperature"] = temperature
        return ChatAnthropic(**kwargs)


def _get_raw_llm(
    temperature: float,
    max_tokens: int,
    model_id: str | None,
) -> Union["ChatOpenAI", "ChatAnthropic"]:  # type: ignore[name-defined]
    """Return the underlying cached ChatAnthropic / ChatOpenAI instance.

    Used internally by invoke_with_failover. Most callers should use get_llm()
    instead, which wraps this in a _FailoverLLM that retries cross-provider
    when the primary model errors out.
    """
    requested = model_id or get_default_model_id()
    active_model = requested
    if active_model not in OPENAI_MODELS and active_model not in ANTHROPIC_MODELS:
        log.warning(
            "Unknown model_id '%s' — falling back to '%s'. Supported: %s",
            active_model, FALLBACK_MODEL_ID, OPENAI_MODELS | ANTHROPIC_MODELS,
        )
        active_model = FALLBACK_MODEL_ID

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
    raise RuntimeError(
        f"Fallback model '{FALLBACK_MODEL_ID}' is itself not in any whitelist — "
        "check OPENAI_MODELS / ANTHROPIC_MODELS in api/llm.py."
    )


class _FailoverLLM:
    """Drop-in stand-in for a langchain ChatModel whose .invoke() walks the
    MODEL_FAILOVER_CHAIN on transient errors (Anthropic 529 overloaded,
    OpenAI 503, timeouts, network drops). Every existing call site that
    does `llm = get_llm(...); llm.invoke(x)` automatically gains
    cross-provider failover with no per-site refactor.

    Only .invoke() is implemented — callers that need .stream(), .batch(),
    or other Runnable methods should fetch the raw model via _get_raw_llm().
    """
    __slots__ = ("model_id", "temperature", "max_tokens")

    def __init__(self, model_id: str | None, temperature: float, max_tokens: int):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, input):  # noqa: A002 — matches langchain Runnable signature
        return invoke_with_failover(
            input,
            model_id=self.model_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


def get_llm(
    temperature: float = 0.0,
    max_tokens: int = 1024,
    model_id: str | None = None,
) -> _FailoverLLM:
    """Return a failover-aware LLM wrapper.

    Behaves like a langchain ChatModel for .invoke() callers but transparently
    retries against MODEL_FAILOVER_CHAIN if the primary errors out (provider
    overload, rate-limit, transient network). Construction is cheap — the
    actual ChatAnthropic / ChatOpenAI instances are still cached inside
    _make_llm() by (provider, key, model, temp, max_tokens).
    """
    return _FailoverLLM(model_id, temperature, max_tokens)


def get_default_model_id() -> str:
    """Returns the default model ID from environment.

    The env var is historically named OPENAI_MODEL but the value can be any
    supported model id (OpenAI or Anthropic). Default fallback is Claude
    Haiku 4.5 — fastest model in the picker with strong instruction
    following.
    """
    return os.environ.get("OPENAI_MODEL", "claude-haiku-4-5")


def invoke_with_failover(
    messages: list,
    model_id: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    max_attempts: int = 4,
    base_delay: float = 1.2,
):
    """Invoke an LLM, failing over to other models on transient errors.

    Walks MODEL_FAILOVER_CHAIN until one model succeeds or all are exhausted.
    The user-selected primary is tried first; deduplicated, the chain visits
    a cross-provider candidate next so a single provider outage doesn't kill
    the request. Small exponential backoff between attempts.

    Returns the LangChain response object. Raises the last exception if all
    chain members fail.
    """
    primary = model_id or get_default_model_id()
    chain = [primary] + [m for m in MODEL_FAILOVER_CHAIN if m != primary]
    chain = chain[:max_attempts]
    last_exc: Exception | None = None
    for i, mid in enumerate(chain):
        try:
            llm = _get_raw_llm(temperature=temperature, max_tokens=max_tokens, model_id=mid)
            result = llm.invoke(messages)
            if i > 0:
                log.info("LLM failover succeeded on '%s' after %d failed attempts", mid, i)
            return result
        except Exception as exc:
            last_exc = exc
            if i < len(chain) - 1:
                delay = base_delay * (1.5 ** i)
                log.warning(
                    "LLM '%s' failed (%s) — failing over to '%s' in %.1fs",
                    mid, exc, chain[i + 1], delay,
                )
                time.sleep(delay)
    assert last_exc is not None
    log.warning("All %d models in failover chain failed; last: %s", len(chain), last_exc)
    raise last_exc


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


def format_history_for_prompt(
    history: list[ChatMessage] | None,
    max_messages: int = 6,
    max_chars_per_msg: int = 500,
) -> str:
    """Format recent conversation history as a compact text block for prompts.

    Returns a plain-text block like:
        user: ...
        assistant: ...

    Always returns a non-empty string. When *history* is empty, returns a
    sentinel "(no previous conversation)" so downstream prompts can include
    a single uniform "CONVERSATION HISTORY" section unconditionally.
    """
    if not history:
        return "(no previous conversation)"
    recent = history[-max_messages:]
    lines: list[str] = []
    for m in recent:
        role = (m.role or "").strip() or "user"
        content = (m.content or "").strip()
        if not content:
            continue
        if len(content) > max_chars_per_msg:
            content = content[:max_chars_per_msg] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(no previous conversation)"
