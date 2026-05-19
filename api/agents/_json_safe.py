"""Lenient parser for LLM-emitted JSON.

LLMs occasionally produce JSON with small syntax errors — a missing comma,
an unterminated string at max_tokens, prose lead-in before the opening
brace. This helper tries strict json.loads first (the common, fast path),
then walks two cheap fallbacks before reaching for json_repair as a last
resort. Returns None when even repair can't recover, so the caller can
fall through to its own heuristic stub.
"""
from __future__ import annotations

import json

from api.logger import get_logger

log = get_logger(__name__)


def safe_parse_json(raw: str) -> dict | None:
    """Parse LLM JSON output with markdown stripping + json_repair salvage."""
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        pass

    # Slice to {...} region — strips prose like "Here is your report: {...}"
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        candidate = text[first : last + 1]
        try:
            result = json.loads(candidate)
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            text = candidate

    try:
        from json_repair import repair_json
        result = repair_json(text, return_objects=True)
        if isinstance(result, dict) and result:
            log.info("LLM JSON salvaged via json_repair (%d chars)", len(text))
            return result
        log.warning("json_repair produced %s — not a usable dict", type(result).__name__)
    except Exception as exc:
        log.warning("json_repair failed: %s", exc)
    return None
