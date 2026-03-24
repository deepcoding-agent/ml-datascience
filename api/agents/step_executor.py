"""Step Executor — runs each step from the AI planner's plan.

Uses pre-built handlers when available, generates code for the rest.
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

    Returns combined result dict with final_df, charts, stdout, etc.
    """
    current_df = df.copy()
    all_charts: list[str] = []
    all_stdout: list[str] = []
    final_df: pd.DataFrame | None = None
    output_type = plan.get("output_type", "query")
    step_results: list[dict] = []
    all_code: list[str] = []

    for step in plan.get("steps", []):
        step_num = step.get("step_num", "?")
        description = step.get("description", "")
        use_handler = step.get("use_handler")
        handler_category = step.get("handler_category")
        handler_params = step.get("handler_params") or {}
        needs_custom_code = step.get("needs_custom_code", False)
        produces = step.get("produces", "text")
        add_viz = step.get("add_visualization", False)
        viz_type = step.get("visualization_type") or "bar"

        log.info("  Step %s: %s", step_num, description[:80])

        step_result: HandlerResult | None = None

        # ── Try pre-built handler first ──────────────────────────────────
        if use_handler and not needs_custom_code:
            handler_fn = _find_handler(use_handler, handler_category)

            if handler_fn:
                log.info("    handler: %s/%s", handler_category, use_handler)
                try:
                    step_result = handler_fn(current_df, handler_params)
                except Exception as e:
                    log.error("    handler error: %s", e)
                    step_result = None
            else:
                log.info("    handler '%s' not found, generating code", use_handler)
                needs_custom_code = True

        # ── Generate and execute custom code ─────────────────────────────
        if needs_custom_code or step_result is None:
            log.info("    generating custom code")
            code_desc = step.get("custom_code_description") or description

            code = generate_step_code(
                step_description=code_desc,
                df_context=df_context,
                current_df=current_df,
                produces=produces,
                llm=llm,
            )
            all_code.append(code)

            stdout, result_df, chart_b64, sandbox_df, chart_json = run_code(
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
                stdout, result_df, chart_b64, sandbox_df, chart_json = run_code(
                    retry_code, current_df
                )

            charts_from_sandbox = [chart_json] if chart_json else []
            step_result = HandlerResult(
                success=not stdout.startswith("Code execution error"),
                result_df=result_df,
                charts_plotly=charts_from_sandbox,
                stdout=stdout,
                output_type=output_type,
                error=stdout if stdout.startswith("Code execution error") else None,
                metadata={"code": code, "chart_image": chart_b64},
            )

        if step_result is None:
            continue

        # ── Collect step outputs ─────────────────────────────────────────
        if step_result.result_df is not None:
            current_df = step_result.result_df
            final_df = step_result.result_df

        if step_result.charts_plotly:
            all_charts.extend(step_result.charts_plotly)

        if step_result.stdout:
            all_stdout.append(step_result.stdout)

        # ── Auto-add visualization if planned ────────────────────────────
        if add_viz and not step_result.charts_plotly:
            target_df = step_result.result_df if step_result.result_df is not None else current_df
            chart = _auto_visualize(target_df, viz_type, description)
            if chart:
                all_charts.append(chart)

        step_results.append(
            {
                "step": step_num,
                "description": description,
                "success": step_result.success,
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


def _find_handler(handler_name: str, category: str | None):
    """Look up a handler in the registry by name and optional category."""
    # Direct match by (category, sub_intent)
    if category:
        fn = HANDLER_REGISTRY.get((category, handler_name))
        if fn:
            return fn

    # Fuzzy match: try all entries
    for (cat, sub), fn in HANDLER_REGISTRY.items():
        if sub == handler_name:
            return fn
        if f"handle_{sub}" == handler_name:
            return fn
    return None


def _auto_visualize(
    df: pd.DataFrame,
    viz_type: str,
    title: str,
) -> str | None:
    """Auto-generate a Plotly chart for a DataFrame result."""
    try:
        if df is None or df.empty or len(df.columns) < 1:
            return None

        cols = list(df.columns)

        # Detect x and y columns
        if len(cols) == 2:
            x_col, y_col = cols[0], cols[1]
        elif len(cols) >= 2:
            numeric = df.select_dtypes(include="number").columns.tolist()
            non_numeric = [c for c in cols if c not in numeric]
            x_col = non_numeric[0] if non_numeric else cols[0]
            y_col = numeric[0] if numeric else cols[1]
        else:
            # Single column
            x_col = cols[0]
            y_col = cols[0]
            viz_type = "histogram"

        n_unique = df[x_col].nunique() if x_col in df.columns else 10

        # Auto-select best viz type if pie requested but too many categories
        if viz_type == "pie" and n_unique > 10:
            viz_type = "bar"

        if viz_type == "pie":
            fig = px.pie(df, names=x_col, values=y_col, title=title)
            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Value: %{value}<br>%{percent}",
            )
        elif viz_type == "histogram":
            fig = px.histogram(df, x=x_col, title=title)
        elif viz_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, title=title)
        elif viz_type == "line":
            fig = px.line(df, x=x_col, y=y_col, title=title)
        elif viz_type == "box":
            fig = px.box(df, y=y_col, title=title)
        elif viz_type == "heatmap":
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if len(num_cols) >= 2:
                fig = px.imshow(
                    df[num_cols].corr().round(2),
                    text_auto=".2f",
                    title=title,
                )
            else:
                return None
        else:
            # Default bar
            df_sorted = df.sort_values(y_col, ascending=False) if y_col != x_col else df
            fig = px.bar(df_sorted, x=x_col, y=y_col, title=title, text=y_col)
            fig.update_traces(texttemplate="%{text}", textposition="outside")

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, Noto Sans Thai, Tahoma, sans-serif", size=13),
            margin=dict(l=40, r=20, t=50, b=40),
        )

        return pio.to_json(fig)

    except Exception as e:
        log.warning("auto_visualize failed: %s", e)
        return None
