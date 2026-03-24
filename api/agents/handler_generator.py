"""Dynamic handler generator — LLM writes, validates, and caches reusable handler functions."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Optional

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)

# Runtime cache: cache_key → compiled handler function
_DYNAMIC_HANDLER_CACHE: dict[str, Callable] = {}

# Disk cache directory
GENERATED_HANDLERS_DIR = Path(__file__).resolve().parent.parent / "handlers" / "generated"
GENERATED_HANDLERS_DIR.mkdir(parents=True, exist_ok=True)

HANDLER_GENERATION_PROMPT = """\
Write a Python handler function for a data science agent.

TASK: {operation_description}

DATAFRAME CONTEXT:
{df_context}

REQUIRED SIGNATURE:
def handle_{handler_name}(df: pd.DataFrame, params: dict) -> HandlerResult:

RULES:
1. Function name must be exactly: handle_{handler_name}
2. Must return HandlerResult object — never raw values
3. If df is None: return HandlerResult(success=False, error="No dataset")
4. Check required params → return error if missing
5. Use BaseHandler.smart_column_match(df, keyword) for column lookup
6. New DataFrame → result_df=result, output_type="generate"
7. Chart → charts_plotly=[fig.to_json()], output_type="query"
8. Stats → stdout=text, output_type="query"
9. Always set summary= with human-readable description
10. Wrap in try/except → HandlerResult(success=False, error=str(e))
11. Use df is None, NEVER if not df
12. Never hardcode column names — use params or smart_column_match
13. The function must be self-contained — imports inside function body

Expected params: {expected_params}

Return ONLY the Python function code. No markdown fences, no explanation.
Start directly with: def handle_{handler_name}(df, params):
"""

VALIDATION_PROMPT = """\
Review this handler function. Respond with JSON only.

FUNCTION:
{generated_code}

USER INTENT: {operation_description}

Check: returns HandlerResult in all paths? handles df is None? no hardcoded columns?
correct output_type? handles edge cases (empty df, missing column)?

Respond ONLY with:
{{"is_valid": true/false, "issues": [...], "fixed_code": "...", "confidence": 0.0-1.0}}
"""


def generate_handler(
    operation_description: str,
    handler_name: str,
    expected_params: dict,
    df_context: str,
    llm: Any,
) -> Optional[Callable]:
    """Generate, validate, and compile a new handler function. Returns callable or None."""
    cache_key = hashlib.md5(f"{handler_name}:{operation_description}".encode()).hexdigest()[:12]

    # Check runtime cache
    if cache_key in _DYNAMIC_HANDLER_CACHE:
        log.info("  handler_gen: cache hit for '%s'", handler_name)
        return _DYNAMIC_HANDLER_CACHE[cache_key]

    # Check disk cache
    disk_path = GENERATED_HANDLERS_DIR / f"{handler_name}_{cache_key}.py"
    if disk_path.exists():
        log.info("  handler_gen: loading from disk '%s'", disk_path.name)
        handler = _load_from_disk(disk_path, handler_name)
        if handler:
            _DYNAMIC_HANDLER_CACHE[cache_key] = handler
            return handler

    log.info("  handler_gen: generating new handler '%s'", handler_name)

    # Step 1: Generate code
    gen_prompt = HANDLER_GENERATION_PROMPT.format(
        operation_description=operation_description,
        handler_name=handler_name,
        expected_params=json.dumps(expected_params, indent=2),
        df_context=df_context[:2000],
    )
    try:
        raw_code = llm.invoke(gen_prompt).content.strip()
    except Exception as e:
        log.error("  handler_gen: LLM generation failed: %s", e)
        return None

    raw_code = _strip_fences(raw_code)

    # Step 2: Syntax check
    if not _check_syntax(raw_code):
        return None

    # Step 3: Validate via LLM
    try:
        val_prompt = VALIDATION_PROMPT.format(
            generated_code=raw_code,
            operation_description=operation_description,
        )
        val_response = llm.invoke(val_prompt).content.strip()
        validation = json.loads(val_response)
        final_code = validation.get("fixed_code", raw_code)
        confidence = validation.get("confidence", 0.5)
    except (json.JSONDecodeError, Exception):
        log.info("  handler_gen: validation parse failed, using raw code")
        final_code = raw_code
        confidence = 0.6

    if not _check_syntax(final_code):
        final_code = raw_code  # fall back to original if fix broke syntax

    # Step 4: Compile
    handler = _compile(final_code, handler_name)
    if handler is None:
        return None

    # Step 5: Cache
    _save_to_disk(disk_path, final_code, operation_description, confidence)
    _DYNAMIC_HANDLER_CACHE[cache_key] = handler
    log.info("  handler_gen: cached '%s' (confidence=%.2f)", handler_name, confidence)
    return handler


def _strip_fences(code: str) -> str:
    lines = code.split("\n")
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _check_syntax(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        log.error("  handler_gen: syntax error: %s", e)
        return False


def _compile(code: str, handler_name: str) -> Optional[Callable]:
    try:
        import numpy as np
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go

        ns: dict[str, Any] = {
            "__builtins__": __builtins__,
            "pd": pd, "np": np, "px": px, "go": go,
            "json": json, "re": __import__("re"),
            "HandlerResult": HandlerResult,
            "BaseHandler": BaseHandler,
        }
        exec(compile(code, f"<dynamic:{handler_name}>", "exec"), ns)
        func_name = f"handle_{handler_name}"
        if func_name not in ns:
            log.error("  handler_gen: function '%s' not found after exec", func_name)
            return None
        return ns[func_name]
    except Exception as e:
        log.error("  handler_gen: compile failed: %s", e)
        return None


def _save_to_disk(path: Path, code: str, desc: str, confidence: float) -> None:
    header = f"# AUTO-GENERATED HANDLER\n# Description: {desc}\n# Confidence: {confidence:.2f}\n\n"
    try:
        path.write_text(header + code, encoding="utf-8")
    except Exception as e:
        log.warning("  handler_gen: disk save failed: %s", e)


def _load_from_disk(path: Path, handler_name: str) -> Optional[Callable]:
    try:
        code = path.read_text(encoding="utf-8")
        code_lines = [l for l in code.split("\n") if not l.startswith("#")]
        return _compile("\n".join(code_lines).strip(), handler_name)
    except Exception as e:
        log.warning("  handler_gen: disk load failed: %s", e)
        return None


def get_cached_handlers() -> list[str]:
    return list(_DYNAMIC_HANDLER_CACHE.keys())


def clear_handler_cache() -> None:
    _DYNAMIC_HANDLER_CACHE.clear()
    log.info("  handler_gen: cache cleared")
