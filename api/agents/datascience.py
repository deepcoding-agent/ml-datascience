"""
DS-Agent Orchestrator — multi-step reasoning agent for data science tasks.

Flow
----
1. Load DataFrames from dataset payloads
2. Analyze context (nulls, skewness, shape, etc.)
3. Classify intent (category + sub_intent + params)
4. Route to pre-built handler OR fallback to LLM code generation
5. Validate result
6. Retry if failed (up to 1 retry with different strategy)
7. LLM interprets result
8. Build final response with output routing
"""
from __future__ import annotations

import re
import time
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from api.agents.context_analyzer import DataContext, analyze_context
from api.agents.intent_classifier import IntentResult, classify_intent
from api.agents.result_validator import validate_result
from api.context import data_context, extract_code_blocks, sanitize_var_name
from api.handlers import get_handler
from api.handlers.base import HandlerResult
from api.handlers.code_handler import CodeHandler
from api.llm import build_lc_history, get_llm
from api.logger import get_logger
from api.models import ChatMessage, DatasetPayload
from api.sandbox import run_code

log = get_logger(__name__)

# ── System prompts ────────────────────────────────────────────────────────────

DS_SYSTEM_CODEGEN = """\
You are PrepPilot's Data Science Agent — an expert Python data analyst.
The dataset is ALREADY LOADED as `df`. Use it directly.

CORE RULES:
- Be CONCISE. Direct answer first, brief explanation.
- COLUMN MATCHING: Match user keywords to actual column names from context below.
  Never hallucinate column names. If ambiguous, mention which column you picked.
- LANGUAGE: Respond in the same language the user used.
- Write ONE clean Python code block. `df` is pre-loaded.
- Always use print() for output. Assign results to `result` variable.
- For generate/modify: result = df.copy() → modify result → print(result.shape)
- For viewing: result = df.head(N); print(result)
- For charts: fig = px.chart_type(...) — system captures `fig` automatically.
  Never call fig.show(). Available: px, go, make_subplots, ff.

{data_context}
"""

DS_SYSTEM_INTERPRET = """\
You are a data science assistant interpreting execution output.
Be concise — lead with the key finding. 2-3 sentences max.
Use bullet points for multiple findings (max 5).
Respond in the same language as the user's question.

Dataset info: {dataset_info}

Question: {question}
Code: ```python
{code}
```
Output: {output}
"""

# ── Complexity detection ──────────────────────────────────────────────────────

_COMPLEX_KEYWORDS = frozenset({
    "train", "model", "predict", "eda", "profile", "explore",
    "pipeline", "feature engineering", "hypothesis", "test",
    "compare models", "cross validation", "confusion matrix",
})


def _is_complex(message: str) -> bool:
    msg = message.lower()
    return any(kw in msg for kw in _COMPLEX_KEYWORDS)


# ── Dataset name generator ───────────────────────────────────────────────────

def _generate_dataset_name(message: str, model_id: str | None = None) -> str:
    try:
        llm = get_llm(temperature=0.0, max_tokens=50, model_id=model_id)
        reply = llm.invoke([HumanMessage(
            content=(
                f"Generate a short snake_case dataset name (max 5 words, no spaces) "
                f"for: \"{message}\". Reply with ONLY the name."
            )
        )]).content.strip().replace(" ", "_").lower()
        return re.sub(r"[^a-z0-9_]", "", reply)[:60] or "result_dataset"
    except Exception:
        return "result_dataset"


# ── LLM code generation fallback ─────────────────────────────────────────────

def _run_llm_codegen(
    message: str,
    df: pd.DataFrame,
    datasets: list[DatasetPayload],
    extra_dfs: dict[str, pd.DataFrame],
    history: list[ChatMessage],
    model_id: str | None,
    ctx: DataContext,
) -> HandlerResult:
    """Fallback: use LLM to generate code, execute in sandbox."""
    # Build context
    ctx_parts = [data_context(df, datasets[0].name)]
    for ds, (_, edf) in zip(datasets[1:], extra_dfs.items()):
        ctx_parts.append(data_context(edf, ds.name))
    full_context = "\n\n---\n\n".join(ctx_parts)
    system_prompt = DS_SYSTEM_CODEGEN.format(data_context=full_context)

    tokens = 2048 if _is_complex(message) else 1024
    llm = get_llm(temperature=0.0, max_tokens=tokens, model_id=model_id)
    history_msgs = build_lc_history(history[-20:]) if history else []

    msgs = [SystemMessage(content=system_prompt)] + history_msgs + [HumanMessage(content=message)]
    step1_reply = llm.invoke(msgs).content

    code_blocks = extract_code_blocks(step1_reply)
    if not code_blocks:
        return HandlerResult(success=True, stdout=step1_reply, summary=step1_reply)

    # Execute code
    all_code = "\n".join(code_blocks)
    stdout, result_df, chart_b64, sandbox_df, chart_json = run_code(all_code, df, extra_dfs)

    # Auto-retry on error
    if stdout.startswith("Code execution error"):
        log.info("  codegen: sandbox error — retrying")
        fix_prompt = f"Error:\n{stdout}\n\nFix the code. Write ONE corrected Python code block."
        retry_msgs = msgs + [AIMessage(content=step1_reply), HumanMessage(content=fix_prompt)]
        retry_reply = get_llm(temperature=0.0, max_tokens=1024, model_id=model_id).invoke(retry_msgs).content
        retry_blocks = extract_code_blocks(retry_reply)
        if retry_blocks:
            stdout, result_df, chart_b64, sandbox_df, chart_json = run_code(retry_blocks[0], df, extra_dfs)
            all_code = retry_blocks[0]

    charts: list[str] = []
    if chart_json:
        charts.append(chart_json)

    has_substantial_df = (result_df is not None and len(result_df) > 10 and len(result_df.columns) >= 3)
    output_type = "generate" if has_substantial_df else "query"

    return HandlerResult(
        success=not stdout.startswith("Code execution error"),
        result_df=result_df,
        charts_plotly=charts,
        stdout=stdout,
        output_type=output_type,
        error=stdout if stdout.startswith("Code execution error") else None,
        metadata={"code": all_code, "chart_image": chart_b64},
    )


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
        var = sanitize_var_name(ds.name)
        extra_dfs[var] = pd.DataFrame(ds.data)

    # Step 2: Analyze context
    ctx = analyze_context(df)
    log.info("  context: %s, nulls=%d cols, dupes=%d", ctx.shape, len(ctx.null_cols), ctx.duplicate_count)

    # Step 3: Classify intent
    intent = classify_intent(message, ctx, history)
    log.info(
        "  intent: %s/%s  confidence=%.2f  output=%s  fallback=%s",
        intent.category, intent.sub_intent, intent.confidence,
        intent.output_type, intent.fallback_to_custom,
    )

    # Step 4: Route to handler or LLM codegen
    if intent.confidence >= 0.3 and not intent.fallback_to_custom:
        handler_fn = get_handler(intent.category, intent.sub_intent)
        if handler_fn:
            log.info("  routing to handler: %s/%s", intent.category, intent.sub_intent)
            try:
                result = handler_fn(df, intent.params)
            except Exception as e:
                log.error("  handler error: %s — falling back to codegen", e)
                result = _run_llm_codegen(message, df, datasets, extra_dfs, history, model_id, ctx)
        else:
            result = _run_llm_codegen(message, df, datasets, extra_dfs, history, model_id, ctx)
    else:
        log.info("  routing to LLM codegen (low confidence or fallback)")
        result = _run_llm_codegen(message, df, datasets, extra_dfs, history, model_id, ctx)

    # Step 5: Validate
    validation = validate_result(result, intent, df)

    # Step 6: Retry if failed
    if not validation.success and validation.retry_strategy == "fallback_to_custom":
        log.info("  retrying with LLM codegen after handler failure")
        result = _run_llm_codegen(message, df, datasets, extra_dfs, history, model_id, ctx)
        validation = validate_result(result, intent, df)

    # Step 7: LLM interprets result (if we have stdout or data to explain)
    if result.stdout and not result.stdout.startswith("Code execution error"):
        try:
            ds_info = f"{primary.name}: {ctx.shape[0]:,} rows, {ctx.shape[1]} cols → {ctx.column_list[:15]}"
            interp_prompt = DS_SYSTEM_INTERPRET.format(
                dataset_info=ds_info,
                question=message,
                code=result.metadata.get("code", "(handler execution)"),
                output=result.stdout[:2000],
            )
            history_msgs = build_lc_history(history[-20:]) if history else []
            interp_msgs = [SystemMessage(content=interp_prompt)] + history_msgs + [
                HumanMessage(content=f"Original question: {message}\nInterpret the result.")
            ]
            llm_interp = get_llm(temperature=0.0, max_tokens=512, model_id=model_id)
            final_text = llm_interp.invoke(interp_msgs).content
        except Exception as e:
            log.error("  interpretation error: %s", e)
            final_text = result.summary or result.stdout
    else:
        final_text = result.summary or result.stdout or result.error or "No output produced."

    # Add warnings to response
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

    # Output routing
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

    log.info(
        "━━ DS-Agent done  intent=%s/%s  output=%s  elapsed=%.1fs ━━",
        intent.category, intent.sub_intent, output_type, time.perf_counter() - t0,
    )
    return final_text, artifacts
