"""
DS-Agent Orchestrator — AI-first routing.

The AI planner is the SOLE decision-maker for handler vs codegen routing.
No hardcoded keywords, no regex patterns. The planner sees the full handler
catalog and decides what to use for each step.

Flow:
  1. Greeting shortcut (trivial, no AI)
  2. AI Planner → structured plan with handler.id or codegen per step
  3. Step Executor → follows planner decisions, codegen fallback on failure
  4. Response → handler summary (fast) or LLM interpreter (codegen results)
"""
from __future__ import annotations

import re
import time
from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage

from api.agents.context_analyzer import analyze_context
from api.agents.planner import plan_steps
from api.agents.result_interpreter import interpret_final_result
from api.agents.step_executor import execute_plan
from api.context import data_context, sanitize_var_name
from api.llm import build_lc_history, get_llm
from api.logger import get_logger
from api.models import ChatMessage, DatasetPayload

log = get_logger(__name__)

# ── Greetings (very short, no-task messages) ─────────────────────────────────

_GREETINGS = frozenset({
    "hi", "hello", "hey", "สวัสดี", "หวัดดี", "ดีครับ", "ดีค่ะ",
    "yo", "sup", "hola", "good morning", "good afternoon",
})


def _is_pure_greeting(message: str) -> bool:
    """Only trigger greeting for very short messages with no task words."""
    words = message.lower().strip().split()
    return len(words) <= 3 and any(w in _GREETINGS for w in words)


# ── Dataset name generator ───────────────────────────────────────────────────

def _generate_dataset_name(message: str, model_id: str | None = None) -> str:
    """Generate a short snake_case dataset name via LLM."""
    try:
        llm = get_llm(temperature=0.0, max_tokens=50, model_id=model_id)
        reply = llm.invoke([HumanMessage(
            content=f'Generate a short snake_case dataset name (max 5 words) for: "{message}". Reply ONLY the name.'
        )]).content.strip().replace(" ", "_").lower()
        return re.sub(r"[^a-z0-9_]", "", reply)[:60] or "result_dataset"
    except Exception:
        return "result_dataset"


# ── Response building ────────────────────────────────────────────────────────

def _build_response_text(
    exec_result: dict,
    plan: dict,
    message: str,
    model_id: str | None,
) -> str:
    """Build response text.

    - Handler-only results → use handler summaries directly (no extra LLM call)
    - Codegen or mixed results → LLM interpreter for richer explanation
    """
    stdout = exec_result.get("stdout", "")
    step_results = exec_result.get("step_results", [])

    # If all steps used handlers and there's stdout, use it directly
    all_handler = step_results and all(s.get("used_handler") for s in step_results)
    if all_handler and stdout:
        return stdout

    # For codegen or mixed results, use LLM interpreter
    try:
        interp_llm = get_llm(temperature=0.0, max_tokens=4096, model_id=model_id)
        return interpret_final_result(message, plan, exec_result, interp_llm)
    except Exception as e:
        log.error("Interpretation failed: %s", e)
        return stdout or "Operation completed."


# ── Main orchestrator ────────────────────────────────────────────────────────

def run_datascience_agent(
    message: str,
    datasets: list[DatasetPayload],
    history: list[ChatMessage],
    model_id: str | None = None,
) -> tuple[str, dict]:
    """AI-first agent: planner decides everything, no hardcoded routing."""
    t0 = time.perf_counter()
    primary = datasets[0]

    # Step 1: Load DataFrames
    log.info("━━ DS-Agent start ━━  datasets=%s", [d.name for d in datasets])
    df = pd.DataFrame(primary.data)
    extra_dfs: dict[str, pd.DataFrame] = {}
    for ds in datasets[1:]:
        extra_dfs[sanitize_var_name(ds.name)] = pd.DataFrame(ds.data)

    # Step 2: Analyze context
    ctx = analyze_context(df)
    log.info("  context: %s, nulls=%d cols, dupes=%d", ctx.shape, len(ctx.null_cols), ctx.duplicate_count)

    # Step 3: Greeting shortcut (trivial — no AI needed)
    if _is_pure_greeting(message):
        elapsed = time.perf_counter() - t0
        log.info("━━ DS-Agent done (greeting) elapsed=%.1fs ━━", elapsed)
        return _handle_greeting(df, ctx, primary.name)

    # Step 4: AI planner — the sole decision-maker
    df_ctx = data_context(df, primary.name)
    planner_llm = get_llm(temperature=0.0, max_tokens=2048, model_id=model_id)
    plan = plan_steps(
        user_message=message,
        df_context=df_ctx,
        llm=planner_llm,
        model_id=model_id,
    )

    # Step 4b: Direct answer — planner decided this is NOT about the dataset
    if plan.get("direct_answer"):
        log.info("  planner → direct_answer (not about dataset)")
        from langchain_core.messages import SystemMessage
        answer_llm = get_llm(temperature=0.3, max_tokens=4096, model_id=model_id)
        msgs = (
            [SystemMessage(content=(
                "You are a helpful AI assistant called PrepPilot. "
                "A dataset is loaded but the user's question is NOT about the dataset. "
                "Answer the question directly and naturally. Be concise."
            ))]
            + build_lc_history(history[-6:])
            + [HumanMessage(content=message)]
        )
        direct_reply = answer_llm.invoke(msgs).content
        elapsed = time.perf_counter() - t0
        log.info("━━ DS-Agent done (direct_answer) elapsed=%.1fs ━━", elapsed)
        return direct_reply, {"output_type": "text", "should_activate": False}

    # Step 6: Execute each step (follows planner's handler/codegen decisions)
    executor_llm = get_llm(temperature=0.0, max_tokens=4096, model_id=model_id)
    exec_result = execute_plan(
        plan=plan,
        df=df,
        df_context=df_ctx,
        llm=executor_llm,
    )

    # Step 7: Build response text
    final_text = _build_response_text(exec_result, plan, message, model_id)

    # Step 8: Build artifacts
    output_type = exec_result.get("output_type", "query")
    artifacts: dict[str, Any] = {}

    if exec_result.get("code"):
        artifacts["code"] = exec_result["code"]

    if exec_result.get("charts_plotly"):
        artifacts["chart_json"] = exec_result["charts_plotly"][0]

    final_df = exec_result.get("final_df")

    if output_type == "generate" and final_df is not None:
        rows_data = final_df.to_dict(orient="records")
        dataset_name = _generate_dataset_name(message, model_id=model_id)
        artifacts["data_wrangled"] = rows_data
        artifacts["dataset_name"] = dataset_name
        artifacts["dataset_shape"] = {
            "rows": len(final_df),
            "cols": len(final_df.columns),
        }
    elif final_df is not None:
        artifacts["inline_table"] = final_df.to_dict(orient="records")

    artifacts["output_type"] = output_type
    artifacts["should_activate"] = False

    elapsed = time.perf_counter() - t0
    log.info(
        "━━ DS-Agent done  output=%s  steps=%d  elapsed=%.1fs ━━",
        output_type,
        len(plan.get("steps", [])),
        elapsed,
    )

    return final_text, artifacts


def _handle_greeting(
    df: pd.DataFrame,
    ctx,
    dataset_name: str,
) -> tuple[str, dict]:
    """Handle pure greetings with a dataset summary."""
    summary = (
        f"Hello! You have the **{dataset_name}** dataset loaded "
        f"({ctx.shape[0]:,} rows × {ctx.shape[1]} columns). "
    )
    if ctx.null_cols:
        null_pct = sum(ctx.null_cols.values()) / len(ctx.null_cols)
        summary += f"There are {len(ctx.null_cols)} columns with missing values (avg {null_pct:.1f}%). "
    if ctx.duplicate_count > 0:
        summary += f"Found {ctx.duplicate_count:,} duplicate rows. "
    summary += "What would you like to do with this data?"

    return summary, {"output_type": "text", "should_activate": False}
