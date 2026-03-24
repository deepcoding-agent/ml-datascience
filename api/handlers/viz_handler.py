"""Visualization handler — all Plotly chart types."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def _group_pie(df: pd.DataFrame, col: str, top_n: int = 5) -> pd.DataFrame:
    """Group values into top N + Other for pie charts."""
    counts = df[col].value_counts()
    if len(counts) > top_n:
        top = counts.head(top_n)
        other = pd.Series({"Other": counts.iloc[top_n:].sum()})
        counts = pd.concat([top, other])
    pie_df = counts.reset_index()
    pie_df.columns = ["category", "count"]
    return pie_df


class VizHandler(BaseHandler):

    @staticmethod
    def handle_bar_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column") or df.select_dtypes(include=["object", "category"]).columns[0] if len(df.select_dtypes(include=["object", "category"]).columns) > 0 else df.columns[0]
        vc = df[col].value_counts().head(15).reset_index()
        vc.columns = [col, "count"]
        fig = px.bar(vc, x=col, y="count", title=f"Bar Chart: {col}")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()], summary=f"Bar chart of '{col}'")

    @staticmethod
    def handle_histogram(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        if not col or col not in df.columns:
            num = df.select_dtypes(include="number").columns
            col = num[0] if len(num) > 0 else df.columns[0]
        fig = px.histogram(df, x=col, title=f"Histogram: {col}", marginal="box")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()], summary=f"Histogram of '{col}'")

    @staticmethod
    def handle_scatter(df: pd.DataFrame, params: dict) -> HandlerResult:
        cols = params.get("columns", [])
        num_cols = df.select_dtypes(include="number").columns.tolist()
        x = cols[0] if len(cols) > 0 and cols[0] in df.columns else (num_cols[0] if len(num_cols) > 0 else df.columns[0])
        y = cols[1] if len(cols) > 1 and cols[1] in df.columns else (num_cols[1] if len(num_cols) > 1 else df.columns[-1])
        color = cols[2] if len(cols) > 2 and cols[2] in df.columns else None
        fig = px.scatter(df, x=x, y=y, color=color, title=f"Scatter: {x} vs {y}")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()], summary=f"Scatter plot: {x} vs {y}")

    @staticmethod
    def handle_line_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        y_col = col if col and col in df.columns else (num_cols[0] if num_cols else df.columns[0])
        fig = px.line(df.reset_index(), x="index", y=y_col, title=f"Line Chart: {y_col}")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()], summary=f"Line chart of '{y_col}'")

    @staticmethod
    def handle_box_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        if col and col in df.columns:
            fig = px.box(df, y=col, title=f"Box Plot: {col}")
        else:
            num_cols = df.select_dtypes(include="number").columns[:6].tolist()
            melted = df[num_cols].melt(var_name="column", value_name="value")
            fig = px.box(melted, x="column", y="value", title="Box Plots")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()], summary=f"Box plot")

    @staticmethod
    def handle_violin_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        num_cols = df.select_dtypes(include="number").columns
        col = col if col and col in df.columns else (num_cols[0] if len(num_cols) > 0 else df.columns[0])
        fig = px.violin(df, y=col, title=f"Violin Plot: {col}", box=True)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()], summary=f"Violin plot of '{col}'")

    @staticmethod
    def handle_heatmap(df: pd.DataFrame, params: dict) -> HandlerResult:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return HandlerResult(success=False, error="Need at least 2 numeric columns for heatmap")
        corr = df[num_cols].corr().round(2)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlBu_r", title="Correlation Heatmap")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()], summary="Correlation heatmap")

    @staticmethod
    def handle_pie_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        cats = df.select_dtypes(include=["object", "category"]).columns
        col = col if col and col in df.columns else (cats[0] if len(cats) > 0 else df.columns[0])
        pie_df = _group_pie(df, col)
        fig = px.pie(pie_df, names="category", values="count", title=f"Distribution: {col}", hole=0.3)
        fig.update_traces(textposition="inside", textinfo="label+percent")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()], summary=f"Pie chart of '{col}'")

    @staticmethod
    def handle_pairplot(df: pd.DataFrame, params: dict) -> HandlerResult:
        num_cols = df.select_dtypes(include="number").columns[:5].tolist()
        if len(num_cols) < 2:
            return HandlerResult(success=False, error="Need at least 2 numeric columns for pairplot")
        fig = px.scatter_matrix(df[num_cols].dropna(), dimensions=num_cols, title="Pair Plot")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()], summary=f"Pair plot of {len(num_cols)} columns")

    @staticmethod
    def handle_missing_heatmap(df: pd.DataFrame, params: dict) -> HandlerResult:
        null_matrix = df.isnull().astype(int)
        fig = px.imshow(null_matrix, title="Missing Values Pattern", color_continuous_scale=["white", "#FF6B35"])
        return HandlerResult(success=True, charts_plotly=[fig.to_json()], summary="Missing values heatmap")

    @staticmethod
    def handle_count_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        cats = df.select_dtypes(include=["object", "category"]).columns
        col = col if col and col in df.columns else (cats[0] if len(cats) > 0 else df.columns[0])
        vc = df[col].value_counts().head(20).reset_index()
        vc.columns = [col, "count"]
        fig = px.bar(vc, x=col, y="count", title=f"Count Plot: {col}")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()], summary=f"Count plot of '{col}'")

    @staticmethod
    def handle_time_series(df: pd.DataFrame, params: dict) -> HandlerResult:
        dt_cols = df.select_dtypes(include="datetime").columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not dt_cols:
            return HandlerResult(success=False, error="No datetime columns found")
        x = dt_cols[0]
        y = params.get("column") or (num_cols[0] if num_cols else df.columns[1])
        fig = px.line(df.sort_values(x), x=x, y=y, title=f"Time Series: {y}")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()], summary=f"Time series: {y} over {x}")

    @staticmethod
    def handle_bubble_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 3:
            return HandlerResult(success=False, error="Need at least 3 numeric columns for bubble chart")
        fig = px.scatter(df, x=num_cols[0], y=num_cols[1], size=num_cols[2],
                         title=f"Bubble: {num_cols[0]} vs {num_cols[1]} (size={num_cols[2]})")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()], summary="Bubble chart")

    @staticmethod
    def handle_treemap(df: pd.DataFrame, params: dict) -> HandlerResult:
        cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if not cats:
            return HandlerResult(success=False, error="No categorical columns for treemap")
        col = cats[0]
        vc = df[col].value_counts().reset_index()
        vc.columns = [col, "count"]
        fig = px.treemap(vc, path=[col], values="count", title=f"Treemap: {col}")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()], summary=f"Treemap of '{col}'")

    @staticmethod
    def handle_sunburst(df: pd.DataFrame, params: dict) -> HandlerResult:
        cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if len(cats) < 2:
            return HandlerResult(success=False, error="Need at least 2 categorical columns for sunburst")
        fig = px.sunburst(df, path=cats[:2], title=f"Sunburst: {cats[0]} → {cats[1]}")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()], summary=f"Sunburst: {cats[0]} → {cats[1]}")

    @staticmethod
    def handle_parallel_coords(df: pd.DataFrame, params: dict) -> HandlerResult:
        num_cols = df.select_dtypes(include="number").columns[:6].tolist()
        if len(num_cols) < 2:
            return HandlerResult(success=False, error="Need at least 2 numeric columns")
        fig = px.parallel_coordinates(df[num_cols].dropna(), dimensions=num_cols, title="Parallel Coordinates")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()], summary="Parallel coordinates plot")

    @staticmethod
    def handle_distribution(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        num_cols = df.select_dtypes(include="number").columns
        col = col if col and col in df.columns else (num_cols[0] if len(num_cols) > 0 else df.columns[0])
        fig = px.histogram(df, x=col, marginal="box", title=f"Distribution: {col}")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()], summary=f"Distribution of '{col}'")
