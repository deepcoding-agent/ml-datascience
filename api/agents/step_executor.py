"""Step Executor — runs each step from the AI planner's plan.

Routing is entirely AI-driven:
  1. Planner specifies handler.id → resolve from registry → execute
  2. Handler fails or not found → silent codegen fallback
  3. Planner specifies codegen → LLM-generated Python → sandbox execution

No hardcoded keyword matching. The AI planner is the sole decision-maker.
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


# ── Handler resolution ──────────────────────────────────────────────────────

def _resolve_handler(handler_id: str):
    """Resolve 'category.sub' string → handler function from registry."""
    if not handler_id or "." not in handler_id:
        return None
    cat, sub = handler_id.split(".", 1)
    return HANDLER_REGISTRY.get((cat, sub))


def _extract_params(step: dict, df: pd.DataFrame) -> dict:
    """Extract handler params from step. Uses planner params + smart column matching."""
    handler_spec = step.get("handler") or {}
    params = dict(handler_spec.get("params") or {})

    # Also check legacy handler_params field
    if not params and step.get("handler_params"):
        params = dict(step["handler_params"])

    # Fuzzy-match column name if specified but not found in df
    if "column" in params and params["column"] not in df.columns:
        matched = BaseHandler.smart_column_match(df, params["column"])
        if matched:
            params["column"] = matched
        else:
            del params["column"]  # remove bad column, let handler pick default

    # Fuzzy-match columns list (for scatter, etc.)
    if "columns" in params:
        resolved = []
        for c in params["columns"]:
            if c in df.columns:
                resolved.append(c)
            else:
                m = BaseHandler.smart_column_match(df, c)
                if m:
                    resolved.append(m)
        params["columns"] = resolved

    return params


# ── Main executor ────────────────────────────────────────────────────────────

def execute_plan(
    plan: dict,
    df: pd.DataFrame,
    df_context: str,
    llm,
) -> dict:
    """Execute each step. Follows planner's routing — handler or codegen."""
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
        log.info("  Step %s: %s", step_num, description[:80])

        result_df: pd.DataFrame | None = None
        result_charts: list[str] = []
        result_stdout = ""
        step_success = True
        used_handler = False

        # ── Route 1: Planner specified handler.id ─────────────────────────
        handler_spec = step.get("handler")
        if handler_spec and current_df is not None:
            handler_id = handler_spec.get("id", "")
            handler_fn = _resolve_handler(handler_id)
            if handler_fn:
                log.info("    route=handler id=%s", handler_id)
                try:
                    params = _extract_params(step, current_df)
                    hr = handler_fn(current_df, params)
                    if hr.success:
                        result_df = hr.result_df
                        result_charts = hr.charts_plotly or []
                        result_stdout = hr.stdout or hr.summary or ""
                        used_handler = True
                    else:
                        log.info(
                            "    handler %s returned error: %s → codegen fallback",
                            handler_id, hr.error,
                        )
                except Exception as e:
                    log.info(
                        "    handler %s raised %s → codegen fallback",
                        handler_id, e,
                    )
            else:
                log.info("    handler %s not in registry → codegen fallback", handler_id)

        # ── Route 2: Codegen (planner specified, or handler fallback) ─────
        if not used_handler:
            codegen_spec = step.get("codegen") or {}
            task = (
                codegen_spec.get("task")
                or step.get("custom_code_description")
                or description
            )
            produces = codegen_spec.get("produces") or step.get("produces", "text")

            log.info("    route=codegen task=%s", task[:60])
            code = generate_step_code(
                step_description=task,
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
                log.info("    codegen retry: %s", stdout[:100])
                retry_code = generate_step_code(
                    step_description=task,
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

        # ── Auto-viz if old-format step requested chart but none produced ──
        add_viz = step.get("add_visualization", False)
        viz_type = step.get("visualization_type") or "bar"
        if add_viz and not result_charts and current_df is not None:
            chart = _auto_visualize(current_df, viz_type, description)
            if chart:
                all_charts.append(chart)

        step_results.append({
            "step": step_num,
            "description": description,
            "success": step_success,
            "used_handler": used_handler,
        })

    return {
        "final_df": final_df,
        "charts_plotly": all_charts,
        "stdout": "\n".join(all_stdout),
        "output_type": output_type,
        "step_results": step_results,
        "code": "\n\n".join(all_code) if all_code else "",
        "success": all(s["success"] for s in step_results) if step_results else True,
    }


# ── Auto-visualize fallback ────────────────────────────────────────────────

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

        # Format interval-style labels
        if _looks_like_range(plot_df[x_col]):
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
            fig.update_traces(textposition="inside", textinfo="percent+label")
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
                    plot_df[num_cols].corr().round(2), text_auto=".2f", title=title
                )
            else:
                return None
        else:
            fig = px.bar(plot_df, x=x_col, y=y_col, title=title, text=y_col)
            fig.update_traces(
                texttemplate="%{text:,}", textposition="outside",
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


# ── Formatting helpers ──────────────────────────────────────────────────────

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
