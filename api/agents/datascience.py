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
from api.agents.critique import critique_result
from api.agents.insights import surface_insights
from api.agents.planner import plan_steps
from api.agents.replanner import propose_followup_steps
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
    history: list[ChatMessage] | None = None,
) -> str:
    """Build response text.

    - Handler-only results → use handler summaries directly (no extra LLM call)
    - Codegen or mixed results → LLM interpreter for richer explanation
    - Non-Latin user messages → route through interpreter for language matching
      (handlers always emit English; the interpreter respects the user's locale)
    """
    stdout = exec_result.get("stdout", "")
    step_results = exec_result.get("step_results", [])

    # If all steps used handlers and the user wrote in plain ASCII (English),
    # we can return raw handler stdout for speed. Otherwise we MUST go through
    # the interpreter so the response matches the user's language.
    all_handler = step_results and all(s.get("used_handler") for s in step_results)
    user_is_ascii = all(ord(c) < 128 for c in message)
    if all_handler and stdout and user_is_ascii:
        return stdout

    # For codegen or mixed results, use LLM interpreter
    try:
        interp_llm = get_llm(temperature=0.0, max_tokens=4096, model_id=model_id)
        return interpret_final_result(
            message, plan, exec_result, interp_llm, history=history,
        )
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
    # Also expose the primary by its sanitized variable name so codegen can
    # reference the ORIGINAL primary df (not the pipelined `df` that may have
    # been mutated by earlier steps). `df` itself remains the chained current df.
    primary_var = sanitize_var_name(primary.name)
    if primary_var and primary_var != "df":
        extra_dfs[primary_var] = df.copy()
    for ds in datasets[1:]:
        var_name = sanitize_var_name(ds.name)
        if var_name and var_name != "df":
            extra_dfs[var_name] = pd.DataFrame(ds.data)

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
    if extra_dfs:
        lines = ["", "## All loaded datasets (available as Python variables)"]
        for var_name, edf in extra_dfs.items():
            tag = " (primary; same as `df`)" if var_name == primary_var else ""
            lines.append(
                f"- `{var_name}`{tag}: shape={edf.shape}, columns={list(edf.columns)[:20]}"
            )
        lines.append(
            "These variables always point to the ORIGINAL dataframes, never "
            "to intermediate pipeline output. Prefer referencing them by name "
            "in codegen for cross-dataset operations so results don't depend "
            "on the order of earlier steps."
        )
        df_ctx = df_ctx + "\n" + "\n".join(lines)

    # Step 4a: Persistent session memory — distill durable facts (target column,
    # active dataset, last model, user preferences) from the FULL history, not
    # just the last 6 turns the planner sees verbatim. Cheap memory model.
    from api.agents.conversation_memory import (
        extract_session_facts,
        format_facts_for_prompt,
    )
    memory_llm = get_llm(temperature=0.0, max_tokens=300, model_id=model_id)
    session_facts = extract_session_facts(history, memory_llm)
    if session_facts:
        log.info("  session memory: %s", list(session_facts.keys()))
        df_ctx = (
            df_ctx
            + "\n\n## SESSION MEMORY (durable facts across the whole chat)\n"
            + format_facts_for_prompt(session_facts)
        )
    planner_llm = get_llm(temperature=0.0, max_tokens=2048, model_id=model_id)
    plan = plan_steps(
        user_message=message,
        df_context=df_ctx,
        llm=planner_llm,
        model_id=model_id,
        history=history,
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
        extra_dfs=extra_dfs or None,
        history=history,
    )

    # Step 6b: Critic-retry loop — run critique → if fix_needed → codegen
    # fix → re-critique. Up to MAX_CRITIQUE_ITERATIONS rounds, stopping on
    # verdict=ok. Each subsequent fix task carries forward the previous
    # critique reasons as "PRIOR ATTEMPTS" feedback so the codegen doesn't
    # repeat earlier mistakes.
    MAX_CRITIQUE_ITERATIONS = 3
    critique_llm = get_llm(temperature=0.0, max_tokens=400, model_id=model_id)
    prior_reasons: list[str] = []

    for iteration in range(MAX_CRITIQUE_ITERATIONS):
        critique = critique_result(
            user_message=message,
            plan=plan,
            exec_result=exec_result,
            df_context=df_ctx,
            llm=critique_llm,
            history=history,
        )
        log.info("  critique[%d]: verdict=%s reason=%s",
                 iteration + 1,
                 critique.get("verdict"),
                 (critique.get("reason") or "")[:80])

        if critique.get("verdict") != "fix_needed" or not critique.get("fix_task"):
            break

        prior_reasons.append((critique.get("reason") or "").strip())
        fix_task = critique["fix_task"]
        if prior_reasons[:-1]:
            # Tell the codegen what previous attempts got wrong so it doesn't
            # re-make the same mistakes. The most recent critique is the
            # most relevant — list it last.
            feedback_block = "\n".join(
                f"  - attempt {i + 1}: {r}"
                for i, r in enumerate(prior_reasons[:-1])
                if r
            )
            if feedback_block:
                fix_task = (
                    f"{fix_task}\n\n"
                    f"PRIOR ATTEMPTS (already tried — do NOT repeat these mistakes):\n"
                    f"{feedback_block}"
                )

        log.info("  critique[%d] → running fix codegen step", iteration + 1)
        fix_plan = {
            "output_type": exec_result.get("output_type", "query"),
            "steps": [{
                "step_num": f"fix-{iteration + 1}",
                "description": f"Critique-driven fix (iter {iteration + 1})",
                "codegen": {"task": fix_task, "produces": "table"},
            }],
        }
        fix_result = execute_plan(
            plan=fix_plan,
            df=df,
            df_context=df_ctx,
            llm=executor_llm,
            extra_dfs=extra_dfs or None,
            history=history,
        )
        # Only adopt the fix if it actually produced something — otherwise keep
        # the original (a failed critique fix should not erase a working result).
        fix_final_df = fix_result.get("final_df")
        fix_charts = fix_result.get("charts_plotly") or []
        fix_stdout = (fix_result.get("stdout") or "").strip()
        if fix_final_df is None and not fix_stdout and not fix_charts:
            log.info("  critique[%d] fix produced nothing — keeping original", iteration + 1)
            break
        # The critique deemed the output wrong, so the fix REPLACES the
        # user-facing artifacts. Only keep the original table/chart when the
        # fix itself produced none — that way a fix that only added clarifying
        # text doesn't accidentally wipe the answer.
        new_final_df = (
            fix_final_df if fix_final_df is not None
            else exec_result.get("final_df")
        )
        new_charts = fix_charts if fix_charts else exec_result.get("charts_plotly", [])
        exec_result = {
            **exec_result,
            "final_df": new_final_df,
            "stdout": (
                (exec_result.get("stdout") or "")
                + "\n"
                + fix_stdout
            ).strip(),
            "code": (
                (exec_result.get("code") or "")
                + f"\n\n# Critique fix (iter {iteration + 1})\n"
                + (fix_result.get("code") or "")
            ).strip(),
            "charts_plotly": new_charts,
        }

    # Step 6c: Adaptive replanning — the planner gets to see the actual output
    # and propose follow-up steps if the answer is objectively incomplete.
    # This is the "think for yourself" pass — it can add, augment, or refine
    # but is biased toward leaving good answers alone.
    replanner_llm = get_llm(temperature=0.0, max_tokens=600, model_id=model_id)
    followup_steps = propose_followup_steps(
        user_message=message,
        plan=plan,
        exec_result=exec_result,
        df_context=df_ctx,
        llm=replanner_llm,
        history=history,
    )
    if followup_steps:
        # Run follow-ups against the CURRENT result so they refine instead of
        # restarting. The original primary and extras are still reachable by
        # variable name (we injected them into extra_dfs earlier).
        followup_df = exec_result.get("final_df")
        if followup_df is None:
            followup_df = df  # fall back to the original primary
        followup_result = execute_plan(
            plan={
                "output_type": exec_result.get("output_type", "query"),
                "steps": followup_steps,
            },
            df=followup_df,
            df_context=df_ctx,
            llm=executor_llm,
            extra_dfs=extra_dfs or None,
            history=history,
        )
        fu_final_df = followup_result.get("final_df")
        fu_charts = followup_result.get("charts_plotly") or []
        fu_stdout = (followup_result.get("stdout") or "").strip()
        if fu_final_df is not None or fu_stdout or fu_charts:
            exec_result = {
                **exec_result,
                "final_df": fu_final_df if fu_final_df is not None else exec_result.get("final_df"),
                "charts_plotly": fu_charts if fu_charts else exec_result.get("charts_plotly", []),
                "stdout": (
                    (exec_result.get("stdout") or "")
                    + "\n"
                    + fu_stdout
                ).strip(),
                "code": (
                    (exec_result.get("code") or "")
                    + "\n\n# Adaptive follow-up\n"
                    + (followup_result.get("code") or "")
                ).strip(),
            }

    # Step 7: Build response text
    final_text = _build_response_text(
        exec_result, plan, message, model_id, history=history,
    )

    # Step 7b: Proactive insights — append a short "💡 Notes" section when a
    # senior data scientist would naturally point out something useful.
    insights_llm = get_llm(temperature=0.3, max_tokens=350, model_id=model_id)
    insight_block = surface_insights(
        user_message=message,
        exec_result=exec_result,
        df_context=df_ctx,
        final_text=final_text,
        llm=insights_llm,
    )
    if insight_block:
        final_text = final_text + insight_block

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
