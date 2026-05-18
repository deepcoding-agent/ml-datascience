"""Persistent conversation memory — extract durable facts from the full
chat history so multi-turn conversations don't lose context past the
6-message truncation window.

Each turn we run one cheap LLM call (default gpt-5.4-nano) that distills
the history into a small dict:

    {
      "target_column":    "churn",
      "active_dataset":   "telecom_features",
      "last_task":        "classification",
      "last_model":       "XGBoost",
      "last_metrics":     "f1=0.87",
      "preferences":      "user prefers concise summaries in Thai",
      "open_questions":   "still wants ROI segmentation analysis"
    }

The dict is then formatted as a SESSION MEMORY block and injected into
the planner / interpreter / critique prompts so they see condensed
long-term state alongside the recent verbatim turns.

Design notes:
  * One-shot extraction per turn — no DB writes, no cache invalidation.
    Stateless on the server, matches how the rest of the agent works.
  * Conservative output — extractor only returns a field when it's
    explicitly present in the history. Missing → omitted, never invented.
  * Cheap fallback — if the LLM call errors out, returns an empty dict
    and downstream prompts simply skip the memory block.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from api.logger import get_logger
from api.models import ChatMessage

log = get_logger(__name__)


MEMORY_SYSTEM = """\
You are a precise memory extractor for a data-science chat assistant.
Read the conversation history and output a single JSON object that
captures DURABLE facts the assistant will need to remember on later
turns — even after older messages are truncated from its context.

Only emit a field when the history explicitly establishes it. Do NOT
guess, infer beyond evidence, or include speculative values.

Schema (every field optional — omit when unknown):
  {
    "target_column":   "<column name the user has picked as the target>",
    "active_dataset":  "<latest dataset name the user is working with>",
    "last_task":       "classification | regression | clustering | unsupervised",
    "last_model":      "<algorithm display name from the latest /train>",
    "last_metrics":    "<key metric=value pairs from latest training>",
    "preferences":     "<user-stated preferences: language, brevity, format>",
    "open_questions":  "<unresolved follow-up the user asked for but hasn't gotten>"
  }

Rules:
  * Output ONLY the JSON object. No markdown fences, no commentary.
  * Values are strings; missing keys are simply omitted (not null, not "").
  * Trust most-recent messages over older ones when they conflict.
"""


def _safe_json(text: str) -> dict[str, Any]:
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        parsed = json.loads(m.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _condense_history(history: list[ChatMessage], max_messages: int = 40,
                      max_chars_per_msg: int = 600) -> str:
    """Produce a compact text block of the FULL conversation (not truncated to
    6 like the planner sees) so the extractor has long-term context."""
    if not history:
        return "(empty)"
    recent = history[-max_messages:]
    lines: list[str] = []
    for m in recent:
        role = (m.role or "user").strip()
        content = (m.content or "").strip()
        if not content:
            continue
        if len(content) > max_chars_per_msg:
            content = content[:max_chars_per_msg] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(empty)"


def extract_session_facts(history: list[ChatMessage], llm) -> dict[str, str]:
    """Run the memory extractor once. Returns a possibly-empty dict.

    `llm` should be a cheap fast model (gpt-5.4-nano by default). Errors
    swallow to an empty dict — memory is best-effort, never blocks the
    turn.
    """
    if not history:
        return {}

    convo = _condense_history(history)
    try:
        reply = llm.invoke([
            SystemMessage(content=MEMORY_SYSTEM),
            HumanMessage(content=f"## Conversation\n{convo}\n\nOutput JSON now."),
        ]).content
    except Exception as exc:
        log.info("memory extractor LLM error: %s — using empty facts", exc)
        return {}

    facts = _safe_json(reply or "")
    # Coerce all values to short strings; drop empties.
    cleaned: dict[str, str] = {}
    for k, v in facts.items():
        if not isinstance(k, str):
            continue
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() in {"null", "none", "n/a", "tbd"}:
            continue
        cleaned[k] = s[:300]
    return cleaned


def format_facts_for_prompt(facts: dict[str, str]) -> str:
    """Render facts as a SESSION MEMORY block for inclusion in agent prompts."""
    if not facts:
        return "(no session memory — first turn or nothing durable yet)"
    label_map = {
        "target_column":  "Target column",
        "active_dataset": "Active dataset",
        "last_task":      "Last task type",
        "last_model":     "Last trained model",
        "last_metrics":   "Last metrics",
        "preferences":    "User preferences",
        "open_questions": "Open follow-up",
    }
    lines = []
    for key, label in label_map.items():
        if key in facts:
            lines.append(f"  - {label}: {facts[key]}")
    # Any extra keys the extractor invented — show them too, gracefully.
    for k, v in facts.items():
        if k not in label_map:
            lines.append(f"  - {k}: {v}")
    return "\n".join(lines) if lines else "(no session memory yet)"
