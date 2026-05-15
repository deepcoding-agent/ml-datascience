"""Adaptive replanner — look at actual results, propose follow-up steps.

Where the initial planner produces a static plan from `df_context` ALONE,
this replanner sees the REAL output of that plan and decides whether more
work is needed to fully serve the user. Examples:

  * User asked "drop area, compare min vs max price" — initial plan picked
    `analysis.compare_extremes` which surfaced ALL columns including `area`.
    The replanner sees the table still has `area` and proposes one extra
    codegen step: drop `area` from the final result and rebuild the chart.

  * User asked an open-ended question and the initial plan answered narrowly.
    The replanner can propose follow-up steps (a second view, a chart, a
    correlation) to deepen the answer.

This is the difference between a tool that follows rules and an agent that
THINKS. The replanner is allowed to add steps, but its prompt strongly
discourages it from adding noise — "ok, no follow-up" is the default verdict.

Cost / safety:
  * One extra LLM call per chat turn (max_tokens ~512).
  * Capped at ONE replanning round per turn — no recursive loops.
  * Returns plain dict; failures degrade gracefully to "no follow-up".
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


REPLANNER_SYSTEM = """\
You are the ADAPTIVE REPLANNER for a data-science agent. The initial plan has
ALREADY executed and produced output. Your job is to read the user's question,
the plan summary, and the ACTUAL result, then decide if any short follow-up
steps would meaningfully improve the answer.

You will be given:
  * User question
  * Loaded dataframes (variable names + column shapes)
  * Initial plan summary (descriptions + handler/codegen routing)
  * The actual final table preview (head, shape, columns)
  * Step stdout

Return JSON ONLY in this exact shape:
  {"follow_up_needed": true | false,
   "reason": "...",
   "steps": [ ...new step objects (same schema as planner steps)... ]}

Default to "false". Only propose steps when ALL of these are true:
  * The current output OBJECTIVELY misses something the user asked for
    (e.g. user said "drop column X" but X is still in the output;
     user said "compare A and B" but only A is shown;
     user asked for a chart and none was produced).
  * The fix can be expressed as 1-2 short steps using handlers OR codegen.
  * The follow-up will RESHAPE or AUGMENT the result, not redo it from
    scratch — assume the current `result_df` is available in the sandbox
    as `df` for codegen follow-ups (in addition to the original primary
    and extras by name).

Follow-up step schema (mirrors the planner schema):
  {"step_num": 1,
   "description": "Drop the `area` column from result",
   "codegen": {"task": "...", "produces": "table"},
   "dataset": "df"}

OR with a handler:
  {"step_num": 1,
   "description": "Top-5 correlated features",
   "handler": {"id": "stats.correlation", "params": {"top_k": 5}},
   "dataset": "df"}

Reasons to RETURN "false":
  * The current answer fully addresses the question — even if it has extra
    information, leave it alone (more context is not wrong).
  * The user's request is ambiguous and the current answer is one valid
    interpretation — do not second-guess.
  * Any proposed step would just rephrase existing output without new value.

Be a thoughtful colleague, not a perfectionist. Most turns need NO follow-up.
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
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def propose_followup_steps(
    user_message: str,
    plan: dict,
    exec_result: dict,
    df_context: str,
    llm,
    history: list | None = None,
) -> list[dict]:
    """Return a list of follow-up step dicts, or [] when no follow-up is needed.

    Steps follow the same schema as the planner's primary steps and will be
    executed by the same `execute_plan` machinery.
    """
    if exec_result.get("output_type") == "generate":
        return []  # transformations are already concrete — don't second-guess

    final_df = exec_result.get("final_df")
    stdout = (exec_result.get("stdout") or "").strip()
    if final_df is None and not stdout:
        return []

    user_block = (
        f"## User question (current turn)\n{user_message}\n\n"
        f"## Recent conversation (use only when the current turn is a follow-up)\n"
        f"{format_history_for_prompt(history)}\n\n"
        f"## Loaded dataframes & schema\n{df_context[:2000]}\n\n"
        f"## Initial plan that ran\n{_plan_summary(plan)}\n\n"
        f"## Final table preview\n{_final_df_preview(final_df)}\n\n"
        f"## Step stdout (truncated)\n{stdout[:600] or '(none)'}\n\n"
        f"Reminder: do NOT propose follow-ups for content the user already saw "
        f"in earlier turns of this conversation — only for things the CURRENT "
        f"turn objectively missed. If the current question stands alone, ignore "
        f"the history block.\n"
    )

    try:
        reply = llm.invoke([
            SystemMessage(content=REPLANNER_SYSTEM),
            HumanMessage(content=user_block),
        ]).content
    except Exception as e:
        log.info("replanner LLM error: %s — no follow-up", e)
        return []

    parsed = _safe_json_extract(reply or "")
    if not parsed:
        log.info("replanner reply not parseable: %s", (reply or "")[:120])
        return []

    if not parsed.get("follow_up_needed"):
        return []

    steps = parsed.get("steps")
    if not isinstance(steps, list) or not steps:
        return []

    # Filter to well-formed steps only — never pass garbage to the executor.
    clean_steps = [
        s for s in steps
        if isinstance(s, dict) and (s.get("handler") or s.get("codegen"))
    ][:2]  # hard cap at 2 follow-up steps to keep latency bounded

    if clean_steps:
        log.info(
            "replanner: %d follow-up step(s) — %s",
            len(clean_steps),
            (parsed.get("reason") or "")[:100],
        )
    return clean_steps
