"""Pre-return critique pass for the DS chat agent.

After `execute_plan` runs but before we build the final response, a lightweight
LLM critique checks: does the produced answer (text + table) actually address
the user's question?

If the critique decides the result is misaligned (e.g. the inline_table reflects
the wrong dataset, or the answer is missing a comparison the user asked for),
it returns a `fix_task` — a short codegen description the caller can execute
ONCE as a follow-up step to overwrite the final_df.

Design notes:
  * Conservative bias — when in doubt, return "ok" so we don't trigger spurious
    rewrites that cost tokens and add latency.
  * Returns plain dict, not a dataclass, so this stays cheap to call and serialize.
  * The LLM is asked for short, strict JSON.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from api.llm import format_history_for_prompt
from api.logger import get_logger

log = get_logger(__name__)


CRITIQUE_SYSTEM = """\
You are an extremely strict reviewer of a data-science assistant's output.
Decide whether the produced answer actually addresses the user's question.

You will be given:
  * The user's original question
  * A short summary of what the agent planned and ran
  * Available dataframes and their column shapes
  * A small preview of the final table the user will see (if any)
  * Any stdout/text the steps produced

Return JSON ONLY in this exact shape:
  {"verdict": "ok" | "fix_needed", "reason": "...", "fix_task": null | "..."}

Rules (BE CONSERVATIVE — false positives are worse than false negatives):
  * Return "ok" unless you can name a SPECIFIC, CONCRETE, OBJECTIVE mismatch.
  * If the user asked to COMPARE multiple datasets but the final table only
    shows one of them, flag "fix_needed" and describe a codegen task that
    produces a SINGLE table containing rows/columns from each dataset side by
    side, referencing the original dataframes by their variable names.
  * If the final table looks like it came from a DIFFERENT dataset than the
    one the question is about (e.g. all-zero null counts when the user asked
    about a dataset that clearly has nulls), flag "fix_needed".
  * If the final table is empty / NaN-only and the user asked for a non-trivial
    answer, flag "fix_needed".
  * Do NOT flag a table just for being "too broad" or "containing extra
    columns" — extra information is rarely harmful and the planner may be
    surfacing useful context. Only narrow when the EXTRA content is clearly
    wrong, not merely unrequested.
  * Do NOT flag stylistic issues, missing charts the user didn't ask for, or
    anything subjective. Only flag substantive mismatches.
  * If you're unsure whether something is a mismatch, return "ok".
  * The `fix_task` (when non-null) must be a one-paragraph Python coding task
    that, when run as a single codegen step with `df` and named extra
    dataframes available, produces BOTH the correct `result` DataFrame AND
    the correct Plotly `fig` (when a chart is appropriate for the question).
    If the original chart visualised the wrong columns/rows, the fix_task
    MUST instruct that the new chart be rebuilt from the corrected data —
    do not leave the user with a chart that contradicts the table.
"""


def _final_df_preview(df: pd.DataFrame | None) -> str:
    if df is None:
        return "(no result table produced)"
    head = df.head(5).to_string(index=False)
    return f"shape={df.shape}, columns={list(df.columns)}\n{head}"


def _plan_summary(plan: dict) -> str:
    steps = plan.get("steps", []) or []
    lines = []
    for s in steps:
        desc = (s.get("description") or "").strip()
        target = (s.get("dataset") or "df").strip()
        route = "handler" if s.get("handler") else "codegen"
        lines.append(f"  - [{route} on {target}] {desc[:120]}")
    return "\n".join(lines) if lines else "  (no steps)"


def _safe_json_extract(text: str) -> dict[str, Any] | None:
    """Pull the first {...} object out of an LLM response."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def critique_result(
    user_message: str,
    plan: dict,
    exec_result: dict,
    df_context: str,
    llm,
    history: list | None = None,
) -> dict:
    """Return {"verdict": "ok" | "fix_needed", "reason": str, "fix_task": str | None}.

    On any failure the verdict is "ok" — we never want the critique itself to
    break a working request.
    """
    final_df = exec_result.get("final_df")
    stdout = (exec_result.get("stdout") or "").strip()

    # Cheap heuristic skips — avoid an LLM call when there's clearly nothing
    # to critique (transformations, greeting/direct_answer paths, etc.).
    if exec_result.get("output_type") == "generate":
        return {"verdict": "ok", "reason": "transformation step", "fix_task": None}
    if final_df is None and not stdout:
        return {"verdict": "ok", "reason": "no result to critique", "fix_task": None}

    preview = _final_df_preview(final_df)
    plan_summary = _plan_summary(plan)
    stdout_snippet = stdout[:600] if stdout else "(no stdout)"

    user_block = (
        f"## User question (current turn)\n{user_message}\n\n"
        f"## Recent conversation (use only if the current question is a follow-up)\n"
        f"{format_history_for_prompt(history)}\n\n"
        f"## Available data context\n{df_context[:1500]}\n\n"
        f"## What the agent planned\n{plan_summary}\n\n"
        f"## Final table preview\n{preview}\n\n"
        f"## Step stdout (truncated)\n{stdout_snippet}\n\n"
        f"Reminder: if the current question is a follow-up, judge alignment "
        f"against the FOLLOW-UP intent (e.g. 'drop area' should result in no "
        f"`area` column), not the original question. If the question stands "
        f"alone, ignore the history block entirely.\n"
    )

    try:
        reply = llm.invoke([
            SystemMessage(content=CRITIQUE_SYSTEM),
            HumanMessage(content=user_block),
        ]).content
    except Exception as e:
        log.info("critique LLM error: %s — defaulting to ok", e)
        return {"verdict": "ok", "reason": f"llm error: {e}", "fix_task": None}

    parsed = _safe_json_extract(reply or "")
    if not parsed:
        log.info("critique reply not parseable: %s", (reply or "")[:120])
        return {"verdict": "ok", "reason": "unparseable critique", "fix_task": None}

    verdict = (parsed.get("verdict") or "").strip()
    if verdict not in ("ok", "fix_needed"):
        return {"verdict": "ok", "reason": "unknown verdict", "fix_task": None}

    fix_task = parsed.get("fix_task")
    if verdict == "fix_needed" and not fix_task:
        # If the model said fix-needed but gave no task, fall back to ok rather
        # than guessing — we'd rather under-correct than spin on nothing.
        return {
            "verdict": "ok",
            "reason": "fix_needed but no fix_task provided",
            "fix_task": None,
        }

    return {
        "verdict": verdict,
        "reason": (parsed.get("reason") or "").strip(),
        "fix_task": fix_task.strip() if isinstance(fix_task, str) else None,
    }
