"""Step Executor — runs each step from the AI planner's plan.

Routing strategy (smart handler scoring):
  - If step description matches a known handler keyword → use handler (instant, 0 LLM calls)
  - If step matches ALWAYS_CODEGEN patterns → use AI codegen (custom logic)
  - If handler fails → silent fallback to codegen
  - Viz steps always try viz handler first
"""
from __future__ import annotations

import re

import pandas as pd
import plotly.express as px
import plotly.io as pio

from api.agents.code_generator import generate_step_code
from api.handlers import HANDLER_REGISTRY
from api.handlers.base import BaseHandler
from api.logger import get_logger
from api.sandbox import run_code

log = get_logger(__name__)


# ── Handler keyword map — step description → (category, sub_intent) ──────────

HANDLER_KEYWORD_MAP: dict[str, tuple[str, str]] = {
    # Stats — instant, always correct
    "shape": ("stats", "shape"),
    "how many rows": ("stats", "shape"),
    "how many columns": ("stats", "shape"),
    "rows and columns": ("stats", "shape"),
    "number of rows": ("stats", "shape"),
    "describe": ("stats", "describe"),
    "summary": ("stats", "describe"),
    "statistics": ("stats", "describe"),
    "null report": ("stats", "null_report"),
    "missing report": ("stats", "null_report"),
    "null": ("stats", "null_report"),
    "missing": ("stats", "null_report"),
    "nan": ("stats", "null_report"),
    "dtypes": ("stats", "dtypes"),
    "data types": ("stats", "dtypes"),
    "column types": ("stats", "dtypes"),
    "unique values": ("stats", "unique_values"),
    "cardinality": ("stats", "unique_values"),
    "value counts": ("stats", "value_counts"),
    "correlation": ("stats", "correlation"),
    "corr": ("stats", "correlation"),
    "skewness": ("stats", "skewness"),
    "skew": ("stats", "skewness"),
    "outlier": ("stats", "outlier_report"),
    "duplicate report": ("stats", "duplicate_report"),

    # Clean — safe, well-tested
    "fill null": ("clean", "fill_nulls"),
    "fill missing": ("clean", "fill_nulls"),
    "impute": ("clean", "fill_nulls"),
    "remove duplicate": ("clean", "remove_duplicates"),
    "drop duplicate": ("clean", "remove_duplicates"),
    "fix dtype": ("clean", "fix_dtypes"),
    "convert type": ("clean", "fix_dtypes"),
    "drop column": ("clean", "drop_column"),
    "remove column": ("clean", "drop_column"),
    "rename": ("clean", "rename_column"),
    "strip whitespace": ("clean", "strip_whitespace"),

    # Transform — correct and fast
    "filter": ("transform", "filter"),
    "sort": ("transform", "sort"),
    "order by": ("transform", "sort"),
    "group by": ("transform", "groupby_agg"),
    "groupby": ("transform", "groupby_agg"),
    "aggregate": ("transform", "groupby_agg"),
    "assign": ("transform", "assign_value"),
    "set all": ("transform", "assign_value"),
    "encode": ("transform", "encode_label"),
    "label encode": ("transform", "encode_label"),
    "one hot": ("transform", "encode_onehot"),
    "scale": ("transform", "scale_standard"),
    "normalize": ("transform", "scale_minmax"),
    "sample": ("transform", "sample_rows"),
    "head": ("transform", "head"),
    "tail": ("transform", "tail"),
    "inject null": ("transform", "inject_null"),
    "add null": ("transform", "inject_null"),
    "generate null": ("transform", "inject_null"),
    "random null": ("transform", "inject_null"),
    "null value": ("transform", "inject_null"),

    # Viz — always use handler (fast + consistent styling)
    "bar chart": ("viz", "bar_chart"),
    "bar graph": ("viz", "bar_chart"),
    "histogram": ("viz", "histogram"),
    "scatter": ("viz", "scatter"),
    "scatter plot": ("viz", "scatter"),
    "pie chart": ("viz", "pie_chart"),
    "pie": ("viz", "pie_chart"),
    "line chart": ("viz", "line_chart"),
    "line plot": ("viz", "line_chart"),
    "box plot": ("viz", "box_plot"),
    "boxplot": ("viz", "box_plot"),
    "heatmap": ("viz", "heatmap"),
    "heat map": ("viz", "heatmap"),
    "heatmap of correlations": ("viz", "heatmap"),
    "correlation heatmap": ("viz", "heatmap"),
    "pairplot": ("viz", "pairplot"),
    "pair plot": ("viz", "pairplot"),
    "distribution": ("viz", "histogram"),
    "count plot": ("viz", "count_plot"),
    "violin": ("viz", "violin_plot"),

    # Feature engineering — use handler
    "feature importance": ("feature", "feature_importance"),
    "important feature": ("feature", "feature_importance"),
    "pca": ("feature", "pca"),
    "principal component": ("feature", "pca"),
    "log transform": ("feature", "log_transform"),
    "correlation filter": ("feature", "correlation_filter"),
}

# Steps that should ALWAYS use codegen — never handlers
ALWAYS_CODEGEN = frozenset({
    "bin", "cut", "split into", "level", "range",
    "moving average", "rolling", "cumulative",
    "z-score", "zscore", "standardize",
    "synthetic", "bootstrap", "resample",
    "polynomial", "interaction feature",
    "custom", "calculate", "compute",
    "merge", "join", "concat",
    "pivot", "reshape", "melt",
    "regex", "extract", "parse",
    "percent of", "proportion of",
})


# ── Handler scoring ──────────────────────────────────────────────────────────

def should_use_handler(step_description: str) -> tuple[str, str] | None:
    """Check if a step should use a pre-built handler.

    Returns (category, sub_intent) if handler found, None if should use codegen.
    Priority:
      1. ALWAYS_CODEGEN patterns → None
      2. HANDLER_KEYWORD_MAP match → handler key
      3. Otherwise → None (codegen)
    """
    desc_lower = step_description.lower()

    # Check always-codegen patterns first
    for pattern in ALWAYS_CODEGEN:
        if pattern in desc_lower:
            return None

    # Check handler keywords — longer match = more specific = higher priority
    best_match: tuple[str, str] | None = None
    best_score = 0

    for keyword, handler_key in HANDLER_KEYWORD_MAP.items():
        if keyword in desc_lower:
            score = len(keyword)
            if score > best_score:
                best_score = score
                best_match = handler_key

    return best_match


def _extract_handler_params(
    step: dict,
    df: pd.DataFrame,
    handler_key: tuple[str, str],
) -> dict:
    """Extract params for a handler from the step description and handler_params."""
    desc = step.get("custom_code_description", "") or step.get("description", "")
    # Start with any params the planner provided
    params = dict(step.get("handler_params") or {})

    # Extract column name via fuzzy match if not already set
    if "column" not in params:
        for word in desc.lower().split():
            if len(word) < 3:
                continue
            col = BaseHandler.smart_column_match(df, word)
            if col:
                params["column"] = col
                break

    # Extract numeric N
    n_match = re.search(r"\b(\d+)\b", desc)
    if n_match and "n" not in params:
        params["n"] = int(n_match.group(1))

    # Extract percentage
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", desc)
    if pct_match:
        params["fraction"] = float(pct_match.group(1)) / 100
        params["value"] = float(pct_match.group(1))

    # Handler-specific params
    cat, sub = handler_key

    if sub == "groupby_agg" and "agg" not in params:
        params["agg"] = "count"
        for agg_kw, agg_fn in [("sum", "sum"), ("mean", "mean"), ("max", "max"), ("min", "min")]:
            if agg_kw in desc.lower():
                params["agg"] = agg_fn
                break

    if sub == "fill_nulls" and "strategy" not in params:
        params["strategy"] = "median"
        if "mean" in desc.lower():
            params["strategy"] = "mean"
        elif "mode" in desc.lower():
            params["strategy"] = "mode"

    if sub == "filter":
        op_match = re.search(r"(>=|<=|!=|==|>|<)\s*(\d+(?:\.\d+)?)", desc)
        if op_match:
            params["operator"] = op_match.group(1)
            params["value"] = float(op_match.group(2))

    if sub == "sort":
        params["ascending"] = "desc" not in desc.lower()

    return params


# ── Main executor ────────────────────────────────────────────────────────────

def execute_plan(
    plan: dict,
    df: pd.DataFrame,
    df_context: str,
    llm,
) -> dict:
    """Execute each step. Uses handler scoring to pick instant handlers vs codegen."""
    current_df = df.copy() if df is not None else None
    all_charts: list[str] = []
    all_stdout: list[str] = []
    final_df: pd.DataFrame | None = None
    output_type = plan.get("output_type", "query")
    step_results: list[dict] = []
    all_code: list[str] = []

    for step in plan.get("steps", []):
        step_num = step.get("step_num", "?")
        description = step.get("description", "")
        custom_desc = step.get("custom_code_description") or description
        produces = step.get("produces", "text")
        add_viz = step.get("add_visualization", False)
        viz_type = step.get("visualization_type") or "bar"

        log.info("  Step %s: %s", step_num, description[:80])

        result_charts: list[str] = []
        result_df: pd.DataFrame | None = None
        result_stdout = ""
        step_success = True

        # ── Decide: handler or codegen? ──────────────────────────────────
        handler_key = should_use_handler(custom_desc)

        if handler_key and current_df is not None:
            cat, sub = handler_key
            handler_fn = HANDLER_REGISTRY.get((cat, sub))

            if handler_fn:
                log.info("    handler: %s/%s (instant)", cat, sub)
                try:
                    params = _extract_handler_params(step, current_df, handler_key)
                    handler_result = handler_fn(current_df, params)

                    if handler_result.success:
                        result_df = handler_result.result_df
                        result_charts = handler_result.charts_plotly or []
                        result_stdout = handler_result.stdout or handler_result.summary or ""
                        # Skip codegen — handler succeeded
                        handler_key = handler_key  # keep truthy
                    else:
                        log.info("    handler %s failed: %s → codegen fallback", sub, handler_result.error)
                        handler_key = None  # trigger codegen
                except Exception as e:
                    log.info("    handler %s raised %s → codegen fallback", sub, e)
                    handler_key = None  # trigger codegen
            else:
                handler_key = None  # not in registry

        if not handler_key:
            # ── Codegen path ─────────────────────────────────────────────
            # For pure viz steps, try viz handler first
            is_viz_step = produces == "chart" or (
                add_viz and produces not in ("dataframe", "text")
            )
            if is_viz_step:
                result_charts = _run_viz_step(step, current_df, viz_type, df_context, llm, all_code)
            else:
                log.info("    codegen: %s", custom_desc[:60])
                code = generate_step_code(
                    step_description=custom_desc,
                    df_context=df_context,
                    current_df=current_df,
                    produces=produces,
                    llm=llm,
                )
                all_code.append(code)

                stdout, sandbox_result_df, chart_b64, sandbox_df, chart_json = run_code(
                    code, current_df
                )

                # Auto-retry on error
                if stdout.startswith("Code execution error"):
                    log.info("    retry: %s", stdout[:100])
                    retry_code = generate_step_code(
                        step_description=custom_desc,
                        df_context=df_context,
                        current_df=current_df,
                        produces=produces,
                        llm=llm,
                        previous_error=stdout,
                    )
                    all_code.append(retry_code)
                    stdout, sandbox_result_df, chart_b64, sandbox_df, chart_json = run_code(
                        retry_code, current_df
                    )

                result_df = sandbox_result_df
                result_stdout = stdout
                if chart_json:
                    result_charts.append(chart_json)
                step_success = not stdout.startswith("Code execution error")

        # ── Collect outputs ──────────────────────────────────────────────
        if result_df is not None:
            current_df = result_df
            final_df = result_df
        if result_charts:
            all_charts.extend(result_charts)
        if result_stdout:
            all_stdout.append(result_stdout)

        # ── Auto viz if step needs chart but none produced yet ───────────
        if add_viz and not result_charts and current_df is not None:
            chart = _auto_visualize(current_df, viz_type, description)
            if chart:
                all_charts.append(chart)

        step_results.append(
            {
                "step": step_num,
                "description": description,
                "success": step_success,
                "produced": produces,
            }
        )

    return {
        "final_df": final_df,
        "charts_plotly": all_charts,
        "stdout": "\n".join(all_stdout),
        "output_type": output_type,
        "step_results": step_results,
        "code": "\n\n".join(all_code) if all_code else "",
        "success": all(s["success"] for s in step_results) if step_results else True,
    }


# ── Viz step execution ───────────────────────────────────────────────────────

def _run_viz_step(
    step: dict,
    current_df: pd.DataFrame,
    viz_type: str,
    df_context: str,
    llm,
    all_code: list[str],
) -> list[str]:
    """Execute a visualization step. Try handler first, codegen fallback."""
    handler_fn = _find_viz_handler(viz_type)

    if handler_fn and current_df is not None:
        log.info("    viz handler: %s (instant)", viz_type)
        try:
            handler_params = {
                "column": (step.get("handler_params") or {}).get("column")
                          or step.get("column"),
                "columns": (step.get("handler_params") or {}).get("columns", []),
                "percentage": (step.get("handler_params") or {}).get("percentage", False),
            }
            result = handler_fn(current_df, handler_params)
            if result.success and result.charts_plotly:
                return result.charts_plotly
            log.info("    viz handler returned no charts, falling through to codegen")
        except Exception as e:
            log.info("    viz handler raised %s, falling through to codegen", e)

    return _codegen_viz(step, current_df, df_context, llm, all_code)


def _find_viz_handler(viz_type: str):
    """Look up a viz handler in the registry by viz_type."""
    fn = HANDLER_REGISTRY.get(("viz", viz_type))
    if fn:
        return fn

    suffixed = f"{viz_type}_chart"
    fn = HANDLER_REGISTRY.get(("viz", suffixed))
    if fn:
        return fn

    suffixed_plot = f"{viz_type}_plot"
    fn = HANDLER_REGISTRY.get(("viz", suffixed_plot))
    if fn:
        return fn

    for (cat, sub), fn in HANDLER_REGISTRY.items():
        if cat == "viz" and (sub == viz_type or viz_type in sub):
            return fn

    return None


def _codegen_viz(
    step: dict,
    current_df: pd.DataFrame,
    df_context: str,
    llm,
    all_code: list[str],
) -> list[str]:
    """Generate and execute code for a viz step. Returns list of plotly JSON."""
    code_desc = step.get("custom_code_description") or step.get("description", "")
    log.info("    viz codegen: %s", code_desc[:60])

    code = generate_step_code(
        step_description=code_desc,
        df_context=df_context,
        current_df=current_df,
        produces="chart",
        llm=llm,
    )
    all_code.append(code)

    stdout, result_df, chart_b64, sandbox_df, chart_json = run_code(code, current_df)
    if chart_json:
        return [chart_json]

    if stdout.startswith("Code execution error"):
        log.info("    viz codegen retry: %s", stdout[:100])
        retry_code = generate_step_code(
            step_description=code_desc,
            df_context=df_context,
            current_df=current_df,
            produces="chart",
            llm=llm,
            previous_error=stdout,
        )
        all_code.append(retry_code)
        stdout, result_df, chart_b64, sandbox_df, chart_json = run_code(
            retry_code, current_df
        )
        if chart_json:
            return [chart_json]

    return []


# ── Bin/range formatting helpers ─────────────────────────────────────────────

def _format_bin_label(interval) -> str:
    """Convert a pandas Interval to a human-readable string like '34K – 154K'."""
    def _fmt(n: float) -> str:
        if abs(n) >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if abs(n) >= 1_000:
            return f"{int(n / 1_000)}K"
        return f"{n:.0f}"

    return f"{_fmt(interval.left)} – {_fmt(interval.right)}"


def _looks_like_range(series: pd.Series) -> bool:
    """Return True if a Series contains interval/range-style labels."""
    if series.empty:
        return False
    sample = str(series.iloc[0])
    return ("(" in sample or "[" in sample) and "," in sample


def _clean_chart_title(description: str) -> str:
    """Derive a clean chart title from a step description."""
    for prefix in ("Use pd.cut to ", "Calculate ", "Compute ", "Generate ", "Create "):
        if description.startswith(prefix):
            description = description[len(prefix):]
            break
    title = description[:60].strip().rstrip(".")
    if title:
        title = title[0].upper() + title[1:]
    return title or "Distribution"


# ── Auto-visualize fallback ──────────────────────────────────────────────────

def _auto_visualize(
    df: pd.DataFrame,
    viz_type: str,
    title: str,
) -> str | None:
    """Auto-generate a Plotly chart for a DataFrame result."""
    try:
        if df is None or df.empty or len(df.columns) < 1:
            return None

        plot_df = df.copy()
        cols = list(plot_df.columns)
        title = _clean_chart_title(title)

        if len(cols) == 2:
            x_col, y_col = cols[0], cols[1]
        elif len(cols) >= 2:
            numeric = plot_df.select_dtypes(include="number").columns.tolist()
            non_numeric = [c for c in cols if c not in numeric]
            x_col = non_numeric[0] if non_numeric else cols[0]
            y_col = numeric[0] if numeric else cols[1]
        else:
            x_col = cols[0]
            y_col = cols[0]
            viz_type = "histogram"

        is_range = _looks_like_range(plot_df[x_col])
        if is_range:
            try:
                plot_df[x_col] = plot_df[x_col].apply(
                    lambda v: _format_bin_label(v) if hasattr(v, "left") else str(v)
                )
            except Exception:
                plot_df[x_col] = plot_df[x_col].astype(str)

        n_unique = plot_df[x_col].nunique() if x_col in plot_df.columns else 10

        if viz_type == "pie" and n_unique > 10:
            viz_type = "bar"

        if viz_type == "pie":
            fig = px.pie(plot_df, names=x_col, values=y_col, title=title)
            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Value: %{value}<br>%{percent}",
            )
        elif viz_type == "histogram":
            fig = px.histogram(plot_df, x=x_col, title=title)
        elif viz_type == "scatter":
            fig = px.scatter(plot_df, x=x_col, y=y_col, title=title)
        elif viz_type == "line":
            fig = px.line(plot_df, x=x_col, y=y_col, title=title)
        elif viz_type == "box":
            fig = px.box(plot_df, y=y_col, title=title)
        elif viz_type == "heatmap":
            num_cols = plot_df.select_dtypes(include="number").columns.tolist()
            if len(num_cols) >= 2:
                fig = px.imshow(
                    plot_df[num_cols].corr().round(2),
                    text_auto=".2f",
                    title=title,
                )
            else:
                return None
        else:
            fig = px.bar(plot_df, x=x_col, y=y_col, title=title, text=y_col)
            fig.update_traces(
                texttemplate="%{text:,}",
                textposition="outside",
                marker_color="#FB8C3C",
            )

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, Noto Sans Thai, Tahoma, sans-serif", size=13),
            xaxis=dict(tickangle=0, tickfont=dict(size=12)),
            yaxis=dict(tickformat=","),
            bargap=0.3,
            margin=dict(l=40, r=20, t=50, b=80),
        )

        return pio.to_json(fig)

    except Exception as e:
        log.warning("auto_visualize failed: %s", e)
        return None
