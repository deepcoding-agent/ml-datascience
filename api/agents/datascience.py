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
from api.handlers.base import (
    BaseHandler, HandlerResult,
    get_useless_columns, has_visualization_intent, select_chart_type,
    translate_thai_keywords,
)
from api.llm import build_lc_history, get_llm
from api.logger import get_logger
from api.models import ChatMessage, DatasetPayload
from api.sandbox import run_code

log = get_logger(__name__)

# ── Explicit chart type detection ─────────────────────────────────────────────

EXPLICIT_CHART_MAP: dict[str, str] = {
    "pie": "pie", "piechart": "pie", "piechart": "pie",
    "ส่วนแบ่ง": "pie", "สัดส่วน": "pie", "วงกลม": "pie",
    "bar": "bar", "barchart": "bar", "แท่ง": "bar",
    "histogram": "histogram", "hist": "histogram",
    "การกระจาย": "histogram", "distribution": "histogram",
    "line": "line", "linechart": "line", "เส้น": "line",
    "trend": "line", "แนวโน้ม": "line",
    "scatter": "scatter", "scatterplot": "scatter", "จุด": "scatter",
    "box": "box", "boxplot": "box", "กล่อง": "box",
    "heatmap": "heatmap", "ความร้อน": "heatmap",
    "violin": "violin",
}

_PERCENT_KEYWORDS = frozenset({
    "percent", "percentage", "%", "proportion", "ratio",
    "เปอร์เซ็นต์", "สัดส่วน", "ร้อยละ",
})


# ── Auto-chart triggers for stats responses ───────────────────────────────────

AUTO_CHART_TRIGGERS = frozenset({
    "percent", "percentage", "เปอร์เซ็น", "ร้อยละ", "สัดส่วน",
    "กี่ percent", "กี่เปอร์เซ็น", "คิดเป็น",
    "how many", "count", "how much", "จำนวน", "กี่", "มีกี่",
    "each", "แต่ละ", "แต่ละแบบ", "แต่ละประเภท",
    "distribution", "breakdown", "proportion", "การกระจาย", "แจกแจง",
})


def should_auto_chart(message: str) -> bool:
    """Return True if the response should include a chart alongside text."""
    msg = message.lower()
    return any(t in msg for t in AUTO_CHART_TRIGGERS)


def _auto_chart_from_result(df: pd.DataFrame, col: str) -> str | None:
    """Generate a chart for a categorical column breakdown. Returns plotly JSON or None."""
    import plotly.express as px
    import plotly.io as pio

    if col not in df.columns:
        return None

    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(2)
    n_cats = len(counts)

    if n_cats <= 8:
        fig = px.pie(counts, names=col, values="count",
                     title=f"{col} Distribution (%)", hole=0.3,
                     hover_data={"pct": ":.2f"})
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(uniformtext_minsize=11, uniformtext_mode="hide")
    else:
        counts_sorted = counts.sort_values("pct", ascending=True)
        fig = px.bar(counts_sorted, x="pct", y=col, orientation="h",
                     title=f"{col} Distribution (%)", text="pct",
                     labels={"pct": "Percentage (%)"})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")

    fig.update_layout(template="plotly_white")
    return pio.to_json(fig)


def detect_explicit_chart_type(message: str) -> str | None:
    """Check if the user explicitly named a chart type. Returns type or None."""
    msg = message.lower().replace(" ", "")
    for keyword, chart_type in EXPLICIT_CHART_MAP.items():
        if keyword.replace(" ", "") in msg:
            return chart_type
    return None


def wants_percentage(message: str) -> bool:
    """Check if the user wants percentage/proportion values."""
    msg = message.lower()
    return any(kw in msg for kw in _PERCENT_KEYWORDS)


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
- PLOTLY TYPE SAFETY: Always convert Interval/Category/mixed types to str before
  passing to px.bar/px.pie x or names. Use [str(x) for x in values] or .astype(str).
  Never mix float and str in the same axis — numpy will raise DType error.
- NULL INJECTION: "inject null"/"add null"/"แทรก null" → df_new = df.copy(),
  randomly set X% to NaN. result = df_new. output_type = "generate".
- BIN / SPLIT / LEVEL: When user says "split into N levels/bins/groups/range/tier":
  Use pd.cut(df['col'], bins=N) NOT value_counts().
  ALWAYS convert bin labels to str before passing to Plotly (avoids numpy dtype errors).
  Example:
  bins = pd.cut(df['SalePrice'], bins=5)
  counts = bins.value_counts().sort_index()
  labels = [str(x) for x in counts.index]  # MUST convert to str
  fig = px.bar(x=labels, y=counts.values.tolist(), title='Price by Level')
- MULTI-STEP: If request has "and visualization", "then plot", etc., do ALL steps:
  first transform/compute, then visualize the result. Never skip the viz step.

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

# ── Complex / multi-step request detection ────────────────────────────────────

_COMPLEX_PATTERNS = [
    # Multi-step conjunctions
    "and visualization", "and plot", "and show", "and chart",
    "then", "after that", "แล้ว", "จากนั้น",
    # Binning / splitting
    "split", "bin", "level", "range", "group into", "tier",
    "แบ่ง", "ระดับ", "ช่วง", "จัดกลุ่ม",
    # Combined operations
    "per", "by each", "for each", "แต่ละ",
    "compare", "vs", "cumulative",
    # Math expressions
    "multiply", "divide", "calculate", "คำนวณ",
]


def _is_complex_request(message: str) -> bool:
    """Detect multi-step or advanced requests that need LLM codegen."""
    msg = message.lower()
    matches = [p for p in _COMPLEX_PATTERNS if p in msg]
    return len(matches) >= 1


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

    # Step 2: Translate Thai keywords → English before any matching
    translated_msg = translate_thai_keywords(message)
    if translated_msg != message:
        log.info("  translated: '%s' → '%s'", message[:60], translated_msg[:60])

    # Step 3: Analyze context + detect useless columns
    ctx = analyze_context(df)
    useless_cols = get_useless_columns(df)
    log.info("  context: %s, nulls=%d cols, dupes=%d, useless=%s",
             ctx.shape, len(ctx.null_cols), ctx.duplicate_count, useless_cols)

    result: HandlerResult | None = None
    intent = IntentResult()  # default, overwritten below

    # Step 4: Complex request → skip fast-path, go straight to codegen
    is_complex = _is_complex_request(translated_msg)
    if is_complex:
        log.info("  complex request detected — routing to LLM codegen")

    # Step 4b: Visualization fast-path (only for SIMPLE viz requests)
    if has_visualization_intent(translated_msg) and not is_complex:
        log.info("  viz fast-path triggered")
        # Find target column via fuzzy match on translated message
        target_col = None
        for word in re.split(r"[\s,;]+", translated_msg.lower()):
            if len(word) < 3:
                continue
            from api.agents.intent_classifier import STRUCTURAL_KEYWORDS
            if word in STRUCTURAL_KEYWORDS:
                continue
            match = BaseHandler.smart_column_match(df, word, exclude=useless_cols)
            if match:
                target_col = match
                break
        if target_col:
            # Explicit chart type from user message wins over auto-detection
            explicit = detect_explicit_chart_type(translated_msg)
            chart_type = explicit or select_chart_type(df, target_col)
            show_pct = wants_percentage(translated_msg)
            log.info("  viz: col='%s' chart='%s' explicit=%s pct=%s", target_col, chart_type, explicit, show_pct)
            from api.handlers.viz_handler import VizHandler
            viz_params = {"column": target_col, "percentage": show_pct}
            viz_map = {
                "bar": VizHandler.handle_bar_chart,
                "histogram": VizHandler.handle_histogram,
                "line": VizHandler.handle_line_chart,
                "pie": VizHandler.handle_pie_chart,
                "scatter": VizHandler.handle_scatter,
                "box": VizHandler.handle_box_plot,
                "heatmap": VizHandler.handle_heatmap,
                "violin": VizHandler.handle_violin_plot,
            }
            handler_fn = viz_map.get(chart_type, VizHandler.handle_bar_chart)
            result = handler_fn(df, viz_params)
            if result:
                result.metadata["tier"] = 1
            # Skip to interpretation
            intent = IntentResult(category="viz", sub_intent=chart_type, params=viz_params,
                                  output_type="query", confidence=1.0, fallback_to_custom=False)
        else:
            log.info("  viz: no target column found, falling through")

    # Step 5: Classify intent (if not already resolved by viz fast-path)
    if not (has_visualization_intent(translated_msg) and not is_complex and result is not None and result.success):
        intent = classify_intent(translated_msg, ctx, history)
        log.info("  intent: %s/%s  conf=%.2f  output=%s  fallback=%s",
                 intent.category, intent.sub_intent, intent.confidence,
                 intent.output_type, intent.fallback_to_custom)

    result_resolved = result is not None and result.success

    # ── Complex requests → skip Tier 1, go to codegen ─────────────────────
    if is_complex and not result_resolved:
        log.info("  complex → skipping Tier 1, using LLM codegen")
        result = _run_llm_codegen(message, df, datasets, extra_dfs, history, model_id, ctx)
        result_resolved = result is not None and result.success

    # ── TIER 1: Pre-built handler (simple requests only) ──────────────────
    if not result_resolved and not is_complex and intent.confidence >= 0.3 and not intent.fallback_to_custom:
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

    # Auto-chart: add chart to stats responses with categorical breakdown
    if (not result.charts_plotly
            and "chart_json" not in artifacts
            and should_auto_chart(translated_msg)
            and result.result_df is not None
            and result.output_type == "query"):
        # Find the categorical column from params or result
        auto_col = intent.params.get("column")
        if auto_col and auto_col in df.columns:
            chart_json = _auto_chart_from_result(df, auto_col)
            if chart_json:
                artifacts["chart_json"] = chart_json
                log.info("  auto-chart: added chart for '%s'", auto_col)

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
    # Translate Thai keywords first
    translated = translate_thai_keywords(message)
    msg = translated.lower()

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

    # Column match — skip structural keywords and useless columns
    from api.agents.intent_classifier import STRUCTURAL_KEYWORDS
    useless = get_useless_columns(df)
    for col in df.columns:
        if col in useless:
            continue
        if col.lower() in msg or col in translated:
            params["column"] = col
            break
    if "column" not in params:
        for word in msg.split():
            if word in STRUCTURAL_KEYWORDS:
                continue
            match = BaseHandler.smart_column_match(df, word, exclude=useless)
            if match:
                params["column"] = match
                break

    return params
