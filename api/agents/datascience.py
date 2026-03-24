"""
DS-Agent Orchestrator — 3-tier multi-step reasoning agent.

Tier 1: Pre-built handlers      → instant, zero LLM call
Tier 2: Dynamic handler gen     → LLM writes reusable handler, validates, caches
Tier 3: One-shot sandbox exec   → fallback for unique tasks
"""
from __future__ import annotations

import re
import time
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from api.agents.context_analyzer import DataContext, analyze_context
from api.agents.handler_generator import generate_handler
from api.agents.intent_classifier import IntentResult, classify_intent, generate_handler_name
from api.agents.result_validator import validate_result
from api.context import data_context, extract_code_blocks, sanitize_var_name
from api.handlers import get_handler
from api.handlers.base import BaseHandler, HandlerResult
from api.llm import build_lc_history, get_llm
from api.logger import get_logger
from api.models import ChatMessage, DatasetPayload
from api.sandbox import run_code

log = get_logger(__name__)

# ── System prompts ────────────────────────────────────────────────────────────

DS_SYSTEM_CODEGEN = """\
You are PrepPilot's Data Science Agent — expert Python analyst.
The dataset is ALREADY LOADED as `df`. Use it directly.

RULES:
- Be CONCISE. Direct answer first, brief explanation.
- COLUMN MATCHING: Match user keywords to actual column names from context.
  Never hallucinate column names.
- LANGUAGE: Respond in same language as user.
- Write ONE Python code block. `df` is pre-loaded.
- Always print() for output. Assign to `result`.
- For modify: result = df.copy() → modify → print(result.shape)
- For charts: fig = px.chart_type(...) — captured automatically. Never fig.show().
  Available: px, go, make_subplots, ff, sns, msno.
- NULL INJECTION: "inject null"/"add null"/"แทรก null" → df_new = df.copy(),
  randomly set X% to NaN. result = df_new. output_type = "generate".

{data_context}
"""

DS_SYSTEM_INTERPRET = """\
You are a data science assistant interpreting execution output.
Be concise — lead with key finding. 2-3 sentences max. Bullets for multiple findings.
Respond in the same language as the user's question.

Dataset: {dataset_info}
Question: {question}
Code: ```python
{code}
```
Output: {output}
"""

_COMPLEX_KEYWORDS = frozenset({
    "train", "model", "predict", "eda", "profile", "explore",
    "pipeline", "feature engineering", "hypothesis", "test",
    "compare models", "cross validation", "confusion matrix",
})


# ── Tier 3: One-shot sandbox exec ─────────────────────────────────────────────

def _run_llm_codegen(
    message: str,
    df: pd.DataFrame,
    datasets: list[DatasetPayload],
    extra_dfs: dict[str, pd.DataFrame],
    history: list[ChatMessage],
    model_id: str | None,
    ctx: DataContext,
    previous_error: str | None = None,
) -> HandlerResult:
    """Tier 3 fallback: LLM generates code, executed in sandbox."""
    ctx_parts = [data_context(df, datasets[0].name)]
    for ds, (_, edf) in zip(datasets[1:], extra_dfs.items()):
        ctx_parts.append(data_context(edf, ds.name))
    full_ctx = "\n\n---\n\n".join(ctx_parts)
    system = DS_SYSTEM_CODEGEN.format(data_context=full_ctx)

    is_complex = any(kw in message.lower() for kw in _COMPLEX_KEYWORDS)
    tokens = 2048 if is_complex else 1024
    llm = get_llm(temperature=0.0, max_tokens=tokens, model_id=model_id)
    hist = build_lc_history(history[-20:]) if history else []

    # Include previous error if retrying
    user_content = message
    if previous_error:
        user_content = f"{message}\n\n(Previous attempt failed: {previous_error}. Try a different approach.)"

    msgs = [SystemMessage(content=system)] + hist + [HumanMessage(content=user_content)]
    step1_reply = llm.invoke(msgs).content

    code_blocks = extract_code_blocks(step1_reply)
    if not code_blocks:
        return HandlerResult(success=True, stdout=step1_reply, summary=step1_reply,
                             metadata={"tier": 3})

    all_code = "\n".join(code_blocks)
    stdout, result_df, chart_b64, sandbox_df, chart_json = run_code(all_code, df, extra_dfs)

    # Auto-retry on error
    if stdout.startswith("Code execution error"):
        log.info("  tier3: sandbox error — retrying")
        fix_msgs = msgs + [AIMessage(content=step1_reply),
                           HumanMessage(content=f"Error:\n{stdout}\n\nFix the code. ONE corrected block.")]
        retry_reply = get_llm(temperature=0.0, max_tokens=1024, model_id=model_id).invoke(fix_msgs).content
        retry_blocks = extract_code_blocks(retry_reply)
        if retry_blocks:
            stdout, result_df, chart_b64, sandbox_df, chart_json = run_code(retry_blocks[0], df, extra_dfs)
            all_code = retry_blocks[0]

    charts = [chart_json] if chart_json else []
    has_big_df = result_df is not None and len(result_df) > 10 and len(result_df.columns) >= 3
    return HandlerResult(
        success=not stdout.startswith("Code execution error"),
        result_df=result_df, charts_plotly=charts, stdout=stdout,
        output_type="generate" if has_big_df else "query",
        error=stdout if stdout.startswith("Code execution error") else None,
        metadata={"code": all_code, "chart_image": chart_b64, "tier": 3},
    )


# ── Dataset name generator ───────────────────────────────────────────────────

def _generate_dataset_name(message: str, model_id: str | None = None) -> str:
    try:
        llm = get_llm(temperature=0.0, max_tokens=50, model_id=model_id)
        reply = llm.invoke([HumanMessage(
            content=f"Generate a short snake_case dataset name (max 5 words) for: \"{message}\". Reply ONLY the name."
        )]).content.strip().replace(" ", "_").lower()
        return re.sub(r"[^a-z0-9_]", "", reply)[:60] or "result_dataset"
    except Exception:
        return "result_dataset"


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_datascience_agent(
    message: str,
    datasets: list[DatasetPayload],
    history: list[ChatMessage],
    model_id: str | None = None,
) -> tuple[str, dict]:
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

    # Step 3: Classify intent
    intent = classify_intent(message, ctx, history)
    log.info("  intent: %s/%s  conf=%.2f  output=%s  fallback=%s",
             intent.category, intent.sub_intent, intent.confidence,
             intent.output_type, intent.fallback_to_custom)

    result: HandlerResult | None = None

    # ── TIER 1: Pre-built handler ─────────────────────────────────────────
    if intent.confidence >= 0.3 and not intent.fallback_to_custom:
        handler_fn = get_handler(intent.category, intent.sub_intent)
        if handler_fn:
            log.info("  TIER 1: pre-built handler %s/%s", intent.category, intent.sub_intent)
            try:
                result = handler_fn(df, intent.params)
                if result:
                    result.metadata["tier"] = 1
            except Exception as e:
                log.error("  TIER 1 error: %s", e)
                result = None

    # ── TIER 2: Dynamic handler generation ────────────────────────────────
    if result is None or not result.success:
        log.info("  TIER 2: generating dynamic handler")
        handler_name = generate_handler_name(message)
        ctx_str = data_context(df, primary.name)

        # Describe operation
        try:
            llm = get_llm(temperature=0.0, max_tokens=200, model_id=model_id)
            desc_reply = llm.invoke([HumanMessage(
                content=(
                    f"Given request: \"{message}\"\n"
                    f"Dataset columns: {ctx.column_list[:20]}\n"
                    f"Describe in ONE sentence what pandas/numpy/plotly operation to perform."
                )
            )]).content.strip()
        except Exception:
            desc_reply = message

        # Extract params
        params = _extract_operation_params(message, df)

        dynamic_fn = generate_handler(
            operation_description=desc_reply,
            handler_name=handler_name,
            expected_params=params,
            df_context=ctx_str,
            llm=get_llm(temperature=0.0, max_tokens=1024, model_id=model_id),
        )

        if dynamic_fn is not None:
            log.info("  TIER 2: executing dynamic handler '%s'", handler_name)
            try:
                result = dynamic_fn(df, params)
                if result:
                    result.metadata["tier"] = 2
                    result.metadata["handler_name"] = handler_name
            except Exception as e:
                log.error("  TIER 2 exec error: %s", e)
                result = None

    # ── TIER 3: One-shot sandbox fallback ─────────────────────────────────
    if result is None or not result.success:
        log.info("  TIER 3: one-shot sandbox fallback")
        prev_err = result.error if result else None
        result = _run_llm_codegen(message, df, datasets, extra_dfs, history, model_id, ctx, prev_err)

    # Step 5: Validate
    validation = validate_result(result, intent, df)

    # Step 6: Retry if validation failed
    if not validation.success and validation.retry_strategy == "fallback_to_custom":
        log.info("  validation failed — retrying via tier 3")
        result = _run_llm_codegen(message, df, datasets, extra_dfs, history, model_id, ctx, validation.errors[0] if validation.errors else None)

    # Step 7: LLM interpretation
    final_text = result.summary or result.stdout or ""
    if result.stdout and not result.stdout.startswith("Code execution error") and result.metadata.get("tier") == 3:
        try:
            ds_info = f"{primary.name}: {ctx.shape[0]:,} rows, {ctx.shape[1]} cols"
            interp = DS_SYSTEM_INTERPRET.format(
                dataset_info=ds_info, question=message,
                code=result.metadata.get("code", "(handler)"),
                output=result.stdout[:2000],
            )
            hist_msgs = build_lc_history(history[-20:]) if history else []
            interp_msgs = [SystemMessage(content=interp)] + hist_msgs + [
                HumanMessage(content=f"Question: {message}\nInterpret the result.")
            ]
            final_text = get_llm(temperature=0.0, max_tokens=512, model_id=model_id).invoke(interp_msgs).content
        except Exception as e:
            log.error("  interpretation error: %s", e)

    if validation.warnings:
        final_text += "\n\n**Warnings:** " + ", ".join(validation.warnings)

    # Step 8: Build artifacts
    artifacts: dict[str, Any] = {}
    if result.metadata.get("code"):
        artifacts["code"] = result.metadata["code"]
    if result.metadata.get("chart_image"):
        artifacts["chart_image"] = result.metadata["chart_image"]
    if result.charts_plotly:
        artifacts["chart_json"] = result.charts_plotly[0]

    output_type = result.output_type
    if output_type == "generate" and result.result_df is not None:
        rows_data = result.result_df.to_dict(orient="records")
        dataset_name = _generate_dataset_name(message, model_id=model_id)
        artifacts["data_wrangled"] = rows_data
        artifacts["dataset_name"] = dataset_name
        artifacts["dataset_shape"] = {"rows": len(result.result_df), "cols": len(result.result_df.columns)}
    elif result.result_df is not None:
        artifacts["inline_table"] = result.result_df.to_dict(orient="records")

    artifacts["output_type"] = output_type
    artifacts["should_activate"] = False
    artifacts["_debug_tier"] = result.metadata.get("tier", 0)

    log.info(
        "━━ DS-Agent done  tier=%s  intent=%s/%s  output=%s  elapsed=%.1fs ━━",
        result.metadata.get("tier"), intent.category, intent.sub_intent,
        output_type, time.perf_counter() - t0,
    )
    return final_text, artifacts


def _extract_operation_params(message: str, df: pd.DataFrame) -> dict:
    """Extract concrete parameters from the user message."""
    params: dict = {}
    msg = message.lower()

    # Percentage
    import re as _re
    pct = _re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent|pct|เปอร์เซ็นต์)", msg)
    if pct:
        params["fraction"] = float(pct.group(1)) / 100
        params["value"] = float(pct.group(1))

    # Integer N
    n_match = _re.search(r"(?:top|first|last|window|n=)\s*(\d+)", msg)
    if n_match:
        params["n"] = int(n_match.group(1))

    # Column match
    for col in df.columns:
        if col.lower() in msg or col in message:
            params["column"] = col
            break
    if "column" not in params:
        for word in msg.split():
            match = BaseHandler.smart_column_match(df, word)
            if match:
                params["column"] = match
                break

    return params
