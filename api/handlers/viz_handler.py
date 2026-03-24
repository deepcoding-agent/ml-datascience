"""Visualization handler — all Plotly chart types with clean minimal theme."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)

# ── Shared minimal theme ─────────────────────────────────────────────────────

_THEME = dict(
    template="plotly_white",
    font=dict(family="Inter, Noto Sans Thai, Tahoma, sans-serif", size=13),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=50, r=30, t=50, b=50),
    hoverlabel=dict(bgcolor="white", font_size=12),
    colorway=["#FB8C3C", "#2EC4B6", "#457B9D", "#E71D36",
              "#FF9F1C", "#A8DADC", "#1D3557", "#6B4226"],
)


def _style(fig: go.Figure, **overrides) -> go.Figure:
    """Apply PrepPilot minimal theme to any figure."""
    layout = {**_THEME, **overrides}
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=False, tickfont=dict(size=11))
    fig.update_yaxes(showgrid=True, gridcolor="#F0F0F0", tickfont=dict(size=11))
    return fig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _group_pie(df: pd.DataFrame, col: str, top_n: int = 6) -> pd.DataFrame:
    """Group values into top N + Other for pie charts."""
    counts = df[col].value_counts()
    if len(counts) > top_n:
        top = counts.head(top_n)
        other = pd.Series({"Other": counts.iloc[top_n:].sum()})
        counts = pd.concat([top, other])
    pie_df = counts.reset_index()
    pie_df.columns = ["category", "count"]
    return pie_df


# ── Handler class ─────────────────────────────────────────────────────────────

class VizHandler(BaseHandler):

    @staticmethod
    def handle_bar_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
        cats = df.select_dtypes(include=["object", "category"]).columns
        col = params.get("column") or (cats[0] if len(cats) > 0 else df.columns[0])
        show_pct = params.get("percentage", False)
        vc = df[col].value_counts().head(15).reset_index()
        vc.columns = [col, "count"]
        if show_pct:
            vc["pct"] = (vc["count"] / len(df) * 100).round(1)
            fig = px.bar(vc, x=col, y="pct", text="pct",
                         labels={"pct": "Percentage (%)"})
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                              marker_color="#FB8C3C")
        else:
            fig = px.bar(vc, x=col, y="count", text="count")
            fig.update_traces(texttemplate="%{text:,}", textposition="outside",
                              marker_color="#FB8C3C")
        _style(fig, title=f"{col}", bargap=0.3)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Bar chart of '{col}'")

    @staticmethod
    def handle_histogram(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        if not col or col not in df.columns:
            num = df.select_dtypes(include="number").columns
            col = num[0] if len(num) > 0 else df.columns[0]
        fig = px.histogram(df, x=col, marginal="box")
        fig.update_traces(marker_color="#FB8C3C", opacity=0.85)
        _style(fig, title=f"Distribution of {col}")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Histogram of '{col}'")

    @staticmethod
    def handle_scatter(df: pd.DataFrame, params: dict) -> HandlerResult:
        cols = params.get("columns", [])
        num_cols = df.select_dtypes(include="number").columns.tolist()
        x = cols[0] if len(cols) > 0 and cols[0] in df.columns else (num_cols[0] if len(num_cols) > 0 else df.columns[0])
        y = cols[1] if len(cols) > 1 and cols[1] in df.columns else (num_cols[1] if len(num_cols) > 1 else df.columns[-1])
        color = cols[2] if len(cols) > 2 and cols[2] in df.columns else None
        fig = px.scatter(df, x=x, y=y, color=color, opacity=0.7)
        _style(fig, title=f"{x} vs {y}")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Scatter plot: {x} vs {y}")

    @staticmethod
    def handle_line_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        y_col = col if col and col in df.columns else (num_cols[0] if num_cols else df.columns[0])
        fig = px.line(df.reset_index(), x="index", y=y_col)
        fig.update_traces(line_color="#FB8C3C")
        _style(fig, title=f"{y_col}")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Line chart of '{y_col}'")

    @staticmethod
    def handle_box_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        if col and col in df.columns:
            fig = px.box(df, y=col)
            title = f"Box Plot: {col}"
        else:
            num_cols = df.select_dtypes(include="number").columns[:6].tolist()
            melted = df[num_cols].melt(var_name="column", value_name="value")
            fig = px.box(melted, x="column", y="value")
            title = "Box Plots"
        _style(fig, title=title)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=title)

    @staticmethod
    def handle_violin_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        num_cols = df.select_dtypes(include="number").columns
        col = col if col and col in df.columns else (num_cols[0] if len(num_cols) > 0 else df.columns[0])
        fig = px.violin(df, y=col, box=True)
        _style(fig, title=f"Violin Plot: {col}")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Violin plot of '{col}'")

    @staticmethod
    def handle_heatmap(df: pd.DataFrame, params: dict) -> HandlerResult:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return HandlerResult(success=False, error="Need at least 2 numeric columns for heatmap")
        corr = df[num_cols].corr().round(2)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlBu_r",
                        aspect="auto")
        _style(fig, title="Correlation Heatmap")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary="Correlation heatmap")

    @staticmethod
    def handle_pie_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        cats = df.select_dtypes(include=["object", "category"]).columns
        col = col if col and col in df.columns else (cats[0] if len(cats) > 0 else df.columns[0])
        pie_df = _group_pie(df, col)
        fig = px.pie(pie_df, names="category", values="count", hole=0.4)
        fig.update_traces(textposition="inside", textinfo="percent+label",
                          marker=dict(line=dict(color="white", width=2)))
        _style(fig, title=f"{col}", showlegend=False)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Pie chart of '{col}'")

    @staticmethod
    def handle_pairplot(df: pd.DataFrame, params: dict) -> HandlerResult:
        num_cols = df.select_dtypes(include="number").columns[:5].tolist()
        if len(num_cols) < 2:
            return HandlerResult(success=False, error="Need at least 2 numeric columns for pairplot")
        fig = px.scatter_matrix(df[num_cols].dropna(), dimensions=num_cols)
        fig.update_traces(diagonal_visible=True, marker=dict(size=3, opacity=0.5))
        _style(fig, title="Pair Plot")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Pair plot of {len(num_cols)} columns")

    @staticmethod
    def handle_missing_heatmap(df: pd.DataFrame, params: dict) -> HandlerResult:
        null_matrix = df.isnull().astype(int)
        fig = px.imshow(null_matrix, color_continuous_scale=["#F8F8F8", "#FB8C3C"],
                        aspect="auto")
        _style(fig, title="Missing Values Pattern")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary="Missing values heatmap")

    @staticmethod
    def handle_count_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        cats = df.select_dtypes(include=["object", "category"]).columns
        col = col if col and col in df.columns else (cats[0] if len(cats) > 0 else df.columns[0])
        vc = df[col].value_counts().head(20).reset_index()
        vc.columns = [col, "count"]
        fig = px.bar(vc, x=col, y="count", text="count")
        fig.update_traces(texttemplate="%{text:,}", textposition="outside",
                          marker_color="#FB8C3C")
        _style(fig, title=f"{col}", bargap=0.3)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Count plot of '{col}'")

    @staticmethod
    def handle_time_series(df: pd.DataFrame, params: dict) -> HandlerResult:
        dt_cols = df.select_dtypes(include="datetime").columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not dt_cols:
            return HandlerResult(success=False, error="No datetime columns found")
        x = dt_cols[0]
        y = params.get("column") or (num_cols[0] if num_cols else df.columns[1])
        fig = px.line(df.sort_values(x), x=x, y=y)
        fig.update_traces(line_color="#FB8C3C")
        _style(fig, title=f"{y} over time")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Time series: {y} over {x}")

    @staticmethod
    def handle_bubble_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 3:
            return HandlerResult(success=False, error="Need at least 3 numeric columns for bubble chart")
        fig = px.scatter(df, x=num_cols[0], y=num_cols[1], size=num_cols[2],
                         opacity=0.7)
        _style(fig, title=f"{num_cols[0]} vs {num_cols[1]} (size: {num_cols[2]})")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary="Bubble chart")

    @staticmethod
    def handle_treemap(df: pd.DataFrame, params: dict) -> HandlerResult:
        cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
        col = params.get("column") or (cats[0] if cats else None)
        if not col or col not in df.columns:
            return HandlerResult(success=False, error="No categorical column for treemap")
        vc = df[col].value_counts().reset_index()
        vc.columns = [col, "count"]
        fig = px.treemap(vc, path=[col], values="count")
        _style(fig, title=f"{col}")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Treemap of '{col}'")

    @staticmethod
    def handle_sunburst(df: pd.DataFrame, params: dict) -> HandlerResult:
        cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if len(cats) < 2:
            return HandlerResult(success=False, error="Need at least 2 categorical columns for sunburst")
        fig = px.sunburst(df, path=cats[:2])
        _style(fig, title=f"{cats[0]} → {cats[1]}")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Sunburst: {cats[0]} → {cats[1]}")

    @staticmethod
    def handle_parallel_coords(df: pd.DataFrame, params: dict) -> HandlerResult:
        num_cols = df.select_dtypes(include="number").columns[:6].tolist()
        if len(num_cols) < 2:
            return HandlerResult(success=False, error="Need at least 2 numeric columns")
        fig = px.parallel_coordinates(df[num_cols].dropna(), dimensions=num_cols)
        _style(fig, title="Parallel Coordinates")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary="Parallel coordinates plot")

    @staticmethod
    def handle_distribution(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        num_cols = df.select_dtypes(include="number").columns
        col = col if col and col in df.columns else (num_cols[0] if len(num_cols) > 0 else df.columns[0])
        fig = px.histogram(df, x=col, marginal="box")
        fig.update_traces(marker_color="#FB8C3C", opacity=0.85)
        _style(fig, title=f"Distribution of {col}")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Distribution of '{col}'")
