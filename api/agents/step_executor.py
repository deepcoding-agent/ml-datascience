"""Step Executor — runs each step from the AI planner's plan.

Routing rule:
  - Viz steps → try viz handler from HANDLER_REGISTRY, codegen fallback
  - All other steps → always AI codegen (never use handlers)
Chains step results: each step receives the DataFrame from the previous step.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.io as pio

from api.agents.code_generator import generate_step_code
from api.handlers import HANDLER_REGISTRY
from api.handlers.base import HandlerResult
from api.logger import get_logger
from api.sandbox import run_code

log = get_logger(__name__)


def execute_plan(
    plan: dict,
    df: pd.DataFrame,
    df_context: str,
    llm,
) -> dict:
    """Execute each step in the plan sequentially.

    Viz steps use handlers (with codegen fallback).
    All other steps always use AI codegen.
    """
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
        produces = step.get("produces", "text")
        add_viz = step.get("add_visualization", False)
        viz_type = step.get("visualization_type") or "bar"
        is_viz_step = produces == "chart" or (add_viz and produces != "dataframe" and produces != "text")

        log.info("  Step %s: %s", step_num, description[:80])

        result_charts: list[str] = []
        result_df: pd.DataFrame | None = None
        result_stdout = ""
        step_success = True

        if is_viz_step:
            # ── Viz step: try handler first, codegen fallback ────────────
            result_charts = _run_viz_step(step, current_df, viz_type, df_context, llm, all_code)
        else:
            # ── Non-viz step: always AI codegen ──────────────────────────
            code_desc = step.get("custom_code_description") or description
            log.info("    codegen: %s", code_desc[:60])

            code = generate_step_code(
                step_description=code_desc,
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
                    step_description=code_desc,
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
    # Find matching viz handler
    handler_fn = _find_viz_handler(viz_type)

    if handler_fn and current_df is not None:
        log.info("    viz handler: %s", viz_type)
        try:
            handler_params = {
                "column": step.get("handler_params", {}).get("column")
                          or step.get("column"),
                "columns": step.get("handler_params", {}).get("columns", []),
                "percentage": step.get("handler_params", {}).get("percentage", False),
            }
            result = handler_fn(current_df, handler_params)
            if result.success and result.charts_plotly:
                return result.charts_plotly
            log.info("    viz handler returned no charts, falling through to codegen")
        except Exception as e:
            log.info("    viz handler raised %s, falling through to codegen", e)

    # Codegen fallback for viz
    return _codegen_viz(step, current_df, df_context, llm, all_code)


def _find_viz_handler(viz_type: str):
    """Look up a viz handler in the registry by viz_type."""
    # Direct match
    fn = HANDLER_REGISTRY.get(("viz", viz_type))
    if fn:
        return fn

    # Fuzzy: "bar" → "bar_chart", "line" → "line_chart"
    suffixed = f"{viz_type}_chart"
    fn = HANDLER_REGISTRY.get(("viz", suffixed))
    if fn:
        return fn

    # Fuzzy: "box" → "box_plot"
    suffixed_plot = f"{viz_type}_plot"
    fn = HANDLER_REGISTRY.get(("viz", suffixed_plot))
    if fn:
        return fn

    # Scan all viz entries
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

    # Retry once
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

        # Detect x and y columns
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

        # Format interval/range labels to human-readable strings
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
