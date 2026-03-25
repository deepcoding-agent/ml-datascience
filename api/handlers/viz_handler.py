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
    fig.update_layout(title_font=dict(size=14, color="#1D1D1F"))
    fig.update_xaxes(showgrid=False, tickfont=dict(size=11),
                     title_font=dict(size=12, color="#86868B"))
    fig.update_yaxes(showgrid=True, gridcolor="#F0F0F0", tickfont=dict(size=11),
                     title_font=dict(size=12, color="#86868B"))
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
        total = len(df)
        vc["pct"] = (vc["count"] / total * 100).round(1)
        if show_pct:
            fig = px.bar(vc, x=col, y="pct", text="pct")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                              marker_color="#FB8C3C")
            _style(fig, title=f"{col} — Percentage Distribution",
                   xaxis_title=col, yaxis_title="Percentage (%)", bargap=0.3)
        else:
            fig = px.bar(vc, x=col, y="count", text="count",
                         hover_data={"pct": ":.1f"})
            fig.update_traces(texttemplate="%{text:,}", textposition="outside",
                              marker_color="#FB8C3C",
                              hovertemplate=f"<b>%{{x}}</b><br>Count: %{{y:,}}<br>Percent: %{{customdata[0]:.1f}}%<extra></extra>")
            _style(fig, title=f"{col} — Value Counts (n={total:,})",
                   xaxis_title=col, yaxis_title="Count", bargap=0.3)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Bar chart of '{col}'")

    @staticmethod
    def handle_histogram(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        if not col or col not in df.columns:
            num = df.select_dtypes(include="number").columns
            col = num[0] if len(num) > 0 else df.columns[0]
        data = df[col].dropna()
        fig = px.histogram(df, x=col, marginal="box")
        fig.update_traces(marker_color="#FB8C3C", opacity=0.85)
        _style(fig, title=f"Distribution of {col} (n={len(data):,}, mean={data.mean():.2f})",
               xaxis_title=col, yaxis_title="Frequency")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Histogram of '{col}'")

    @staticmethod
    def handle_scatter(df: pd.DataFrame, params: dict) -> HandlerResult:
        cols = params.get("columns", [])
        num_cols = df.select_dtypes(include="number").columns.tolist()
        x = cols[0] if len(cols) > 0 and cols[0] in df.columns else (num_cols[0] if len(num_cols) > 0 else df.columns[0])
        y = cols[1] if len(cols) > 1 and cols[1] in df.columns else (num_cols[1] if len(num_cols) > 1 else df.columns[-1])
        color = cols[2] if len(cols) > 2 and cols[2] in df.columns else None
        fig = px.scatter(df, x=x, y=y, color=color, opacity=0.7,
                         trendline="ols" if color is None else None)
        _style(fig, title=f"{y} vs {x} (n={len(df):,})",
               xaxis_title=x, yaxis_title=y)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Scatter plot: {x} vs {y}")

    @staticmethod
    def handle_line_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        y_col = col if col and col in df.columns else (num_cols[0] if num_cols else df.columns[0])
        fig = px.line(df.reset_index(), x="index", y=y_col)
        fig.update_traces(line_color="#FB8C3C")
        _style(fig, title=f"{y_col} — Trend", xaxis_title="Index", yaxis_title=y_col)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Line chart of '{y_col}'")

    @staticmethod
    def handle_box_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        if col and col in df.columns:
            data = df[col].dropna()
            fig = px.box(df, y=col, points="outliers")
            title = f"Box Plot: {col} (median={data.median():.2f})"
        else:
            num_cols = df.select_dtypes(include="number").columns[:6].tolist()
            melted = df[num_cols].melt(var_name="column", value_name="value")
            fig = px.box(melted, x="column", y="value", points="outliers")
            title = f"Box Plots — {len(num_cols)} Numeric Columns"
        _style(fig, title=title, yaxis_title="Value")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=title)

    @staticmethod
    def handle_violin_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        num_cols = df.select_dtypes(include="number").columns
        col = col if col and col in df.columns else (num_cols[0] if len(num_cols) > 0 else df.columns[0])
        data = df[col].dropna()
        fig = px.violin(df, y=col, box=True, points="outliers")
        _style(fig, title=f"Violin: {col} (median={data.median():.2f})",
               yaxis_title=col)
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
                          marker=dict(line=dict(color="white", width=2)),
                          hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percent: %{percent}<extra></extra>")
        _style(fig, title=f"{col} — Proportion (n={len(df):,})", showlegend=True)
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
        total = len(df)
        vc["pct"] = (vc["count"] / total * 100).round(1)
        fig = px.bar(vc, x=col, y="count", text="count", hover_data={"pct": ":.1f"})
        fig.update_traces(texttemplate="%{text:,}", textposition="outside",
                          marker_color="#FB8C3C",
                          hovertemplate=f"<b>%{{x}}</b><br>Count: %{{y:,}}<br>Percent: %{{customdata[0]:.1f}}%<extra></extra>")
        _style(fig, title=f"{col} — Frequency (n={total:,})",
               xaxis_title=col, yaxis_title="Count", bargap=0.3)
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
        data = df[col].dropna()
        fig = px.histogram(df, x=col, marginal="box")
        fig.update_traces(marker_color="#FB8C3C", opacity=0.85)
        _style(fig, title=f"Distribution of {col} (n={len(data):,}, mean={data.mean():.2f}, std={data.std():.2f})",
               xaxis_title=col, yaxis_title="Frequency")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Distribution of '{col}'")

    # ── Stacked bar ──────────────────────────────────────────────────────────

    @staticmethod
    def handle_stacked_bar(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Stacked bar chart — group by one column, color by another."""
        cols = params.get("columns", [])
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        x_col = cols[0] if len(cols) > 0 and cols[0] in df.columns else (cat_cols[0] if len(cat_cols) > 0 else df.columns[0])
        color_col = cols[1] if len(cols) > 1 and cols[1] in df.columns else (cat_cols[1] if len(cat_cols) > 1 else None)

        if color_col:
            ct = pd.crosstab(df[x_col], df[color_col])
            fig = px.bar(ct, barmode="stack")
        else:
            vc = df[x_col].value_counts().reset_index()
            vc.columns = [x_col, "count"]
            fig = px.bar(vc, x=x_col, y="count")
            fig.update_traces(marker_color="#FB8C3C")

        _style(fig, title=f"{x_col}" + (f" by {color_col}" if color_col else ""), bargap=0.3)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Stacked bar: {x_col}" + (f" by {color_col}" if color_col else ""))

    # ── Area chart ───────────────────────────────────────────────────────────

    @staticmethod
    def handle_area_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Area chart for trends or compositions."""
        col = params.get("column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        y_col = col if col and col in df.columns else (num_cols[0] if num_cols else df.columns[0])
        fig = px.area(df.reset_index(), x="index", y=y_col)
        _style(fig, title=f"{y_col}")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Area chart of '{y_col}'")

    # ── QQ plot ──────────────────────────────────────────────────────────────

    @staticmethod
    def handle_qq_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
        """QQ plot — check if data follows normal distribution."""
        from scipy import stats as sp_stats

        col = params.get("column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col = col if col and col in df.columns else (num_cols[0] if num_cols else None)
        if not col:
            return HandlerResult(success=False, error="No numeric column for QQ plot")

        data = df[col].dropna().values
        (theoretical, sample), (slope, intercept, _) = sp_stats.probplot(data, dist="norm")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=theoretical, y=sample, mode="markers",
                                 marker=dict(color="#FB8C3C", size=5, opacity=0.7), name="Data"))
        line_x = [min(theoretical), max(theoretical)]
        line_y = [slope * x + intercept for x in line_x]
        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines",
                                 line=dict(color="#457B9D", dash="dash"), name="Normal"))
        _style(fig, title=f"QQ Plot: {col}",
               xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"QQ plot of '{col}'")

    # ── Density plot (KDE) ───────────────────────────────────────────────────

    @staticmethod
    def handle_density_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
        """KDE density plot — smoother alternative to histogram."""
        col = params.get("column")
        group_col = params.get("group")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col = col if col and col in df.columns else (num_cols[0] if num_cols else None)
        if not col:
            return HandlerResult(success=False, error="No numeric column for density plot")

        if group_col and group_col in df.columns:
            fig = px.histogram(df, x=col, color=group_col, histnorm="density",
                               marginal="rug", barmode="overlay", opacity=0.5)
        else:
            fig = px.histogram(df, x=col, histnorm="density", marginal="rug")
            fig.update_traces(marker_color="#FB8C3C", opacity=0.7)

        _style(fig, title=f"Density: {col}")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Density plot of '{col}'")

    # ── Strip / jitter plot ──────────────────────────────────────────────────

    @staticmethod
    def handle_strip_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Strip/jitter plot — shows individual data points."""
        col = params.get("column")
        group_col = params.get("group")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        col = col if col and col in df.columns else (num_cols[0] if num_cols else df.columns[0])
        group_col = group_col if group_col and group_col in df.columns else (cat_cols[0] if cat_cols else None)

        if group_col:
            fig = px.strip(df, x=group_col, y=col, color=group_col)
        else:
            fig = px.strip(df, y=col)

        fig.update_traces(marker=dict(size=5, opacity=0.6))
        _style(fig, title=f"{col}" + (f" by {group_col}" if group_col else ""), showlegend=False)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Strip plot of '{col}'")

    # ── Donut chart ───────────────────────────────────────────────────────────

    @staticmethod
    def handle_donut_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Donut chart — pie chart with a hole."""
        col = params.get("column")
        cats = df.select_dtypes(include=["object", "category"]).columns
        col = col if col and col in df.columns else (cats[0] if len(cats) > 0 else df.columns[0])
        pie_df = _group_pie(df, col)
        fig = px.pie(pie_df, names="category", values="count", hole=0.55)
        fig.update_traces(textposition="inside", textinfo="percent+label",
                          marker=dict(line=dict(color="white", width=2)))
        _style(fig, title=f"{col} — Donut (n={len(df):,})", showlegend=True)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Donut chart of '{col}'")

    # ── Grouped bar ───────────────────────────────────────────────────────────

    @staticmethod
    def handle_grouped_bar(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Grouped bar chart — compare across two categoricals."""
        cols = params.get("columns", [])
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        x_col = cols[0] if len(cols) > 0 and cols[0] in df.columns else (cat_cols[0] if len(cat_cols) > 0 else df.columns[0])
        color_col = cols[1] if len(cols) > 1 and cols[1] in df.columns else (cat_cols[1] if len(cat_cols) > 1 else None)
        if color_col:
            ct = pd.crosstab(df[x_col], df[color_col])
            fig = px.bar(ct, barmode="group")
        else:
            vc = df[x_col].value_counts().head(15).reset_index()
            vc.columns = [x_col, "count"]
            fig = px.bar(vc, x=x_col, y="count")
            fig.update_traces(marker_color="#FB8C3C")
        _style(fig, title=f"{x_col}" + (f" by {color_col}" if color_col else "") + " — Grouped",
               bargap=0.3)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Grouped bar: {x_col}" + (f" by {color_col}" if color_col else ""))

    # ── Percent bar (100% stacked) ────────────────────────────────────────────

    @staticmethod
    def handle_percent_bar(df: pd.DataFrame, params: dict) -> HandlerResult:
        """100% stacked bar chart — show proportions within each group."""
        cols = params.get("columns", [])
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        x_col = cols[0] if len(cols) > 0 and cols[0] in df.columns else (cat_cols[0] if len(cat_cols) > 0 else df.columns[0])
        color_col = cols[1] if len(cols) > 1 and cols[1] in df.columns else (cat_cols[1] if len(cat_cols) > 1 else None)
        if color_col:
            ct = pd.crosstab(df[x_col], df[color_col], normalize="index") * 100
            fig = px.bar(ct, barmode="stack")
            fig.update_layout(yaxis_ticksuffix="%")
        else:
            vc = df[x_col].value_counts(normalize=True).head(15).reset_index()
            vc.columns = [x_col, "pct"]
            vc["pct"] = (vc["pct"] * 100).round(1)
            fig = px.bar(vc, x=x_col, y="pct", text="pct")
            fig.update_traces(texttemplate="%{text:.1f}%", marker_color="#FB8C3C")
        _style(fig, title=f"{x_col} — 100% Stacked", bargap=0.3,
               yaxis_title="Percentage (%)")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Percent bar: {x_col}")

    # ── Pareto chart ──────────────────────────────────────────────────────────

    @staticmethod
    def handle_pareto_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Pareto chart — bar + cumulative line."""
        col = params.get("column")
        cats = df.select_dtypes(include=["object", "category"]).columns
        col = col if col and col in df.columns else (cats[0] if len(cats) > 0 else df.columns[0])
        vc = df[col].value_counts().head(15).reset_index()
        vc.columns = [col, "count"]
        vc["cumulative_pct"] = (vc["count"].cumsum() / vc["count"].sum() * 100).round(1)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=vc[col], y=vc["count"], name="Count",
                             marker_color="#FB8C3C"))
        fig.add_trace(go.Scatter(x=vc[col], y=vc["cumulative_pct"], name="Cumulative %",
                                 yaxis="y2", mode="lines+markers",
                                 line=dict(color="#457B9D", width=2)))
        fig.update_layout(yaxis2=dict(title="Cumulative %", overlaying="y", side="right",
                                      range=[0, 105], showgrid=False))
        _style(fig, title=f"{col} — Pareto", yaxis_title="Count", bargap=0.3)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Pareto chart of '{col}'")

    # ── Waterfall chart ───────────────────────────────────────────────────────

    @staticmethod
    def handle_waterfall_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Waterfall chart — show incremental changes."""
        col = params.get("column")
        cat_col = params.get("category")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        col = col if col and col in df.columns else (num_cols[0] if num_cols else df.columns[0])
        cat_col = cat_col if cat_col and cat_col in df.columns else (cat_cols[0] if cat_cols else None)
        if cat_col:
            agg = df.groupby(cat_col)[col].sum().head(10)
            labels, values = agg.index.tolist(), agg.values.tolist()
        else:
            data = df[col].dropna().head(15)
            labels = [str(i) for i in data.index]
            values = data.tolist()
        fig = go.Figure(go.Waterfall(x=labels, y=values,
                                     connector=dict(line=dict(color="#86868B")),
                                     increasing=dict(marker_color="#2EC4B6"),
                                     decreasing=dict(marker_color="#E71D36"),
                                     totals=dict(marker_color="#FB8C3C")))
        _style(fig, title=f"{col} — Waterfall", yaxis_title=col)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Waterfall chart of '{col}'")

    # ── Funnel chart ──────────────────────────────────────────────────────────

    @staticmethod
    def handle_funnel_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Funnel chart — show stages/flow."""
        col = params.get("column")
        cats = df.select_dtypes(include=["object", "category"]).columns
        col = col if col and col in df.columns else (cats[0] if len(cats) > 0 else df.columns[0])
        vc = df[col].value_counts().head(10).reset_index()
        vc.columns = [col, "count"]
        fig = px.funnel(vc, x="count", y=col)
        _style(fig, title=f"{col} — Funnel")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Funnel chart of '{col}'")

    # ── Radar chart ───────────────────────────────────────────────────────────

    @staticmethod
    def handle_radar_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Radar/spider chart — compare numeric columns for a row or aggregation."""
        num_cols = df.select_dtypes(include="number").columns[:8].tolist()
        if len(num_cols) < 3:
            return HandlerResult(success=False, error="Need at least 3 numeric columns for radar chart")
        means = df[num_cols].mean()
        normalized = ((means - means.min()) / (means.max() - means.min()) * 100).round(1)
        fig = go.Figure(go.Scatterpolar(
            r=normalized.tolist() + [normalized.tolist()[0]],
            theta=num_cols + [num_cols[0]],
            fill="toself", fillcolor="rgba(251,140,60,0.3)",
            line=dict(color="#FB8C3C")))
        _style(fig, title="Radar — Normalized Means")
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Radar chart of {len(num_cols)} columns")

    # ── Lollipop chart ────────────────────────────────────────────────────────

    @staticmethod
    def handle_lollipop_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Lollipop chart — dot + stem line."""
        col = params.get("column")
        cats = df.select_dtypes(include=["object", "category"]).columns
        col = col if col and col in df.columns else (cats[0] if len(cats) > 0 else df.columns[0])
        vc = df[col].value_counts().head(15).reset_index()
        vc.columns = [col, "count"]
        fig = go.Figure()
        for _, row in vc.iterrows():
            fig.add_trace(go.Scatter(x=[0, row["count"]], y=[row[col], row[col]],
                                     mode="lines", line=dict(color="#86868B", width=1.5),
                                     showlegend=False))
        fig.add_trace(go.Scatter(x=vc["count"], y=vc[col], mode="markers",
                                 marker=dict(color="#FB8C3C", size=10),
                                 name="Count"))
        _style(fig, title=f"{col} — Lollipop", xaxis_title="Count")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Lollipop chart of '{col}'")

    # ── Dual axis ─────────────────────────────────────────────────────────────

    @staticmethod
    def handle_dual_axis(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Dual Y-axis chart — two numeric columns on separate axes."""
        cols = params.get("columns", [])
        num_cols = df.select_dtypes(include="number").columns.tolist()
        y1 = cols[0] if len(cols) > 0 and cols[0] in df.columns else (num_cols[0] if len(num_cols) > 0 else None)
        y2 = cols[1] if len(cols) > 1 and cols[1] in df.columns else (num_cols[1] if len(num_cols) > 1 else None)
        if not y1 or not y2:
            return HandlerResult(success=False, error="Need at least 2 numeric columns for dual axis")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df[y1], mode="lines", name=y1,
                                 line=dict(color="#FB8C3C")))
        fig.add_trace(go.Scatter(y=df[y2], mode="lines", name=y2,
                                 yaxis="y2", line=dict(color="#457B9D")))
        fig.update_layout(yaxis2=dict(title=y2, overlaying="y", side="right",
                                      showgrid=False))
        _style(fig, title=f"{y1} & {y2} — Dual Axis", yaxis_title=y1)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Dual axis: {y1} vs {y2}")

    # ── 2D histogram ──────────────────────────────────────────────────────────

    @staticmethod
    def handle_histogram_2d(df: pd.DataFrame, params: dict) -> HandlerResult:
        """2D histogram / density heatmap of two numeric columns."""
        cols = params.get("columns", [])
        num_cols = df.select_dtypes(include="number").columns.tolist()
        x = cols[0] if len(cols) > 0 and cols[0] in df.columns else (num_cols[0] if len(num_cols) > 0 else None)
        y = cols[1] if len(cols) > 1 and cols[1] in df.columns else (num_cols[1] if len(num_cols) > 1 else None)
        if not x or not y:
            return HandlerResult(success=False, error="Need at least 2 numeric columns for 2D histogram")
        fig = px.density_heatmap(df, x=x, y=y, marginal_x="histogram", marginal_y="histogram",
                                 color_continuous_scale="Oranges")
        _style(fig, title=f"{x} vs {y} — 2D Density")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"2D histogram: {x} vs {y}")

    # ── ECDF plot ─────────────────────────────────────────────────────────────

    @staticmethod
    def handle_ecdf_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Empirical cumulative distribution function plot."""
        col = params.get("column")
        num_cols = df.select_dtypes(include="number").columns
        col = col if col and col in df.columns else (num_cols[0] if len(num_cols) > 0 else None)
        if not col:
            return HandlerResult(success=False, error="No numeric column for ECDF")
        fig = px.ecdf(df, x=col)
        fig.update_traces(line_color="#FB8C3C")
        _style(fig, title=f"ECDF of {col}", xaxis_title=col, yaxis_title="Cumulative Probability")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"ECDF of '{col}'")

    # ── Step chart ────────────────────────────────────────────────────────────

    @staticmethod
    def handle_step_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Step chart — discrete steps instead of smooth lines."""
        col = params.get("column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col = col if col and col in df.columns else (num_cols[0] if num_cols else df.columns[0])
        fig = px.line(df.reset_index(), x="index", y=col)
        fig.update_traces(line_shape="hv", line_color="#FB8C3C")
        _style(fig, title=f"{col} — Step Chart", xaxis_title="Index", yaxis_title=col)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Step chart of '{col}'")

    # ── Error bar chart ───────────────────────────────────────────────────────

    @staticmethod
    def handle_error_bar_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Bar chart with standard deviation error bars."""
        col = params.get("column")
        group_col = params.get("group")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        col = col if col and col in df.columns else (num_cols[0] if num_cols else None)
        group_col = group_col if group_col and group_col in df.columns else (cat_cols[0] if cat_cols else None)
        if not col or not group_col:
            return HandlerResult(success=False, error="Need a numeric column and a categorical column for error bars")
        agg = df.groupby(group_col)[col].agg(["mean", "std"]).reset_index()
        fig = go.Figure(go.Bar(x=agg[group_col], y=agg["mean"],
                               error_y=dict(type="data", array=agg["std"].fillna(0)),
                               marker_color="#FB8C3C"))
        _style(fig, title=f"{col} by {group_col} (mean +/- std)",
               xaxis_title=group_col, yaxis_title=col, bargap=0.3)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Error bar chart: {col} by {group_col}")

    # ── Ridgeline / joy plot ──────────────────────────────────────────────────

    @staticmethod
    def handle_ridgeline(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Ridgeline plot — overlapping distributions by group."""
        col = params.get("column")
        group_col = params.get("group")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        col = col if col and col in df.columns else (num_cols[0] if num_cols else None)
        group_col = group_col if group_col and group_col in df.columns else (cat_cols[0] if cat_cols else None)
        if not col or not group_col:
            return HandlerResult(success=False, error="Need a numeric column and a group column for ridgeline")
        groups = df[group_col].value_counts().head(8).index.tolist()
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, grp in enumerate(groups):
            data = df[df[group_col] == grp][col].dropna()
            fig.add_trace(go.Violin(x=data, name=str(grp), side="positive",
                                    line_color=colors[i % len(colors)], meanline_visible=True))
        fig.update_traces(orientation="h", width=1.8)
        _style(fig, title=f"{col} by {group_col} — Ridgeline", showlegend=True)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Ridgeline: {col} by {group_col}")

    # ── Dot plot (Cleveland) ──────────────────────────────────────────────────

    @staticmethod
    def handle_dot_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Cleveland dot plot — clean comparison of values across categories."""
        col = params.get("column")
        value_col = params.get("value")
        cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col = col if col and col in df.columns else (cats[0] if cats else df.columns[0])
        value_col = value_col if value_col and value_col in df.columns else (num_cols[0] if num_cols else None)
        if value_col:
            agg = df.groupby(col)[value_col].mean().sort_values().tail(15).reset_index()
            agg.columns = [col, "value"]
        else:
            agg = df[col].value_counts().tail(15).reset_index()
            agg.columns = [col, "value"]
        fig = go.Figure(go.Scatter(x=agg["value"], y=agg[col], mode="markers",
                                   marker=dict(color="#FB8C3C", size=10)))
        _style(fig, title=f"{col} — Dot Plot", xaxis_title=value_col or "Count")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Dot plot of '{col}'")

    # ── Polar chart ───────────────────────────────────────────────────────────

    @staticmethod
    def handle_polar_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Polar/radial chart for categorical data."""
        col = params.get("column")
        cats = df.select_dtypes(include=["object", "category"]).columns
        col = col if col and col in df.columns else (cats[0] if len(cats) > 0 else df.columns[0])
        vc = df[col].value_counts().head(12).reset_index()
        vc.columns = [col, "count"]
        fig = px.bar_polar(vc, r="count", theta=col)
        fig.update_traces(marker_color="#FB8C3C", opacity=0.85)
        _style(fig, title=f"{col} — Polar Chart")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Polar chart of '{col}'")

    # ── Sankey chart ──────────────────────────────────────────────────────────

    @staticmethod
    def handle_sankey_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Sankey flow diagram between two categorical columns."""
        cols = params.get("columns", [])
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        src_col = cols[0] if len(cols) > 0 and cols[0] in df.columns else (cat_cols[0] if len(cat_cols) > 0 else None)
        tgt_col = cols[1] if len(cols) > 1 and cols[1] in df.columns else (cat_cols[1] if len(cat_cols) > 1 else None)
        if not src_col or not tgt_col:
            return HandlerResult(success=False, error="Need 2 categorical columns for Sankey")
        flow = df.groupby([src_col, tgt_col]).size().reset_index(name="count")
        flow = flow.nlargest(20, "count")
        labels = list(set(flow[src_col].tolist() + flow[tgt_col].tolist()))
        label_idx = {l: i for i, l in enumerate(labels)}
        fig = go.Figure(go.Sankey(
            node=dict(label=labels, color="#FB8C3C", pad=15),
            link=dict(source=[label_idx[s] for s in flow[src_col]],
                      target=[label_idx[t] for t in flow[tgt_col]],
                      value=flow["count"].tolist())))
        _style(fig, title=f"{src_col} → {tgt_col} — Flow")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Sankey: {src_col} → {tgt_col}")

    # ── Candlestick chart ─────────────────────────────────────────────────────

    @staticmethod
    def handle_candlestick(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Candlestick / OHLC chart — needs open/high/low/close columns."""
        num_cols = df.select_dtypes(include="number").columns.tolist()
        o = params.get("open") or next((c for c in num_cols if "open" in c.lower()), None)
        h = params.get("high") or next((c for c in num_cols if "high" in c.lower()), None)
        lo = params.get("low") or next((c for c in num_cols if "low" in c.lower()), None)
        c = params.get("close") or next((c for c in num_cols if "close" in c.lower()), None)
        if not all([o, h, lo, c]):
            if len(num_cols) >= 4:
                o, h, lo, c = num_cols[:4]
            else:
                return HandlerResult(success=False, error="Need open/high/low/close columns for candlestick")
        dt_cols = df.select_dtypes(include="datetime").columns.tolist()
        x = dt_cols[0] if dt_cols else df.index
        fig = go.Figure(go.Candlestick(x=x if isinstance(x, pd.Index) else df[x],
                                       open=df[o], high=df[h], low=df[lo], close=df[c]))
        _style(fig, title="Candlestick Chart")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary="Candlestick chart")

    # ── Contour plot ──────────────────────────────────────────────────────────

    @staticmethod
    def handle_contour_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Contour density plot of two numeric columns."""
        cols = params.get("columns", [])
        num_cols = df.select_dtypes(include="number").columns.tolist()
        x = cols[0] if len(cols) > 0 and cols[0] in df.columns else (num_cols[0] if len(num_cols) > 0 else None)
        y = cols[1] if len(cols) > 1 and cols[1] in df.columns else (num_cols[1] if len(num_cols) > 1 else None)
        if not x or not y:
            return HandlerResult(success=False, error="Need at least 2 numeric columns for contour plot")
        fig = px.density_contour(df, x=x, y=y, marginal_x="histogram", marginal_y="histogram")
        fig.update_traces(contours_coloring="fill", colorscale="Oranges")
        _style(fig, title=f"{x} vs {y} — Contour Density")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Contour plot: {x} vs {y}")

    # ── Cumulative line ───────────────────────────────────────────────────────

    @staticmethod
    def handle_cumulative_line(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Cumulative sum line chart."""
        col = params.get("column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col = col if col and col in df.columns else (num_cols[0] if num_cols else None)
        if not col:
            return HandlerResult(success=False, error="No numeric column for cumulative line")
        cumsum = df[col].dropna().cumsum().reset_index(drop=True)
        fig = px.line(x=cumsum.index, y=cumsum.values)
        fig.update_traces(line_color="#FB8C3C")
        _style(fig, title=f"{col} — Cumulative Sum", xaxis_title="Index",
               yaxis_title=f"Cumulative {col}")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Cumulative line of '{col}'")

    # ── Null bar ──────────────────────────────────────────────────────────────

    @staticmethod
    def handle_null_bar(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Bar chart of null percentage per column."""
        null_pct = (df.isnull().sum() / len(df) * 100).round(2)
        null_pct = null_pct[null_pct > 0].sort_values(ascending=False)
        if len(null_pct) == 0:
            return HandlerResult(success=True, charts_plotly=[],
                                 summary="No null values found in any column")
        ndf = null_pct.reset_index()
        ndf.columns = ["column", "null_pct"]
        fig = px.bar(ndf, x="column", y="null_pct", text="null_pct")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                          marker_color="#E71D36")
        _style(fig, title="Null Percentage per Column", xaxis_title="Column",
               yaxis_title="Null %", bargap=0.3)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Null bar: {len(null_pct)} columns with nulls")

    # ── Top N bar ─────────────────────────────────────────────────────────────

    @staticmethod
    def handle_top_n_bar(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Top N values bar chart."""
        col = params.get("column")
        n = params.get("n", 10)
        cats = df.select_dtypes(include=["object", "category"]).columns
        col = col if col and col in df.columns else (cats[0] if len(cats) > 0 else df.columns[0])
        vc = df[col].value_counts().head(n).reset_index()
        vc.columns = [col, "count"]
        fig = px.bar(vc, x=col, y="count", text="count")
        fig.update_traces(texttemplate="%{text:,}", textposition="outside",
                          marker_color="#FB8C3C")
        _style(fig, title=f"Top {n} — {col}", xaxis_title=col,
               yaxis_title="Count", bargap=0.3)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Top {n} bar chart of '{col}'")

    # ── Correlation scatter ───────────────────────────────────────────────────

    @staticmethod
    def handle_correlation_scatter(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Scatter plot with OLS trendline and R-squared value."""
        cols = params.get("columns", [])
        num_cols = df.select_dtypes(include="number").columns.tolist()
        x = cols[0] if len(cols) > 0 and cols[0] in df.columns else (num_cols[0] if len(num_cols) > 0 else None)
        y = cols[1] if len(cols) > 1 and cols[1] in df.columns else (num_cols[1] if len(num_cols) > 1 else None)
        if not x or not y:
            return HandlerResult(success=False, error="Need at least 2 numeric columns")
        clean = df[[x, y]].dropna()
        r = clean[x].corr(clean[y])
        fig = px.scatter(clean, x=x, y=y, trendline="ols", opacity=0.6)
        fig.update_traces(marker_color="#FB8C3C", selector=dict(mode="markers"))
        _style(fig, title=f"{x} vs {y} (r = {r:.3f})",
               xaxis_title=x, yaxis_title=y)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Correlation scatter: {x} vs {y}, r={r:.3f}")

    # ── Range plot ────────────────────────────────────────────────────────────

    @staticmethod
    def handle_range_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Range/band plot showing min-max range over categories."""
        col = params.get("column")
        group_col = params.get("group")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        col = col if col and col in df.columns else (num_cols[0] if num_cols else None)
        group_col = group_col if group_col and group_col in df.columns else (cat_cols[0] if cat_cols else None)
        if not col or not group_col:
            return HandlerResult(success=False, error="Need a numeric and a categorical column for range plot")
        agg = df.groupby(group_col)[col].agg(["min", "max", "mean"]).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=agg[group_col], y=agg["max"], mode="lines",
                                 line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=agg[group_col], y=agg["min"], mode="lines",
                                 line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(251,140,60,0.25)", name="Min-Max Range"))
        fig.add_trace(go.Scatter(x=agg[group_col], y=agg["mean"], mode="lines+markers",
                                 line=dict(color="#FB8C3C", width=2), name="Mean"))
        _style(fig, title=f"{col} Range by {group_col}", yaxis_title=col)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Range plot: {col} by {group_col}")

    # ── Swarm / jitter plot ───────────────────────────────────────────────────

    @staticmethod
    def handle_swarm_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Jitter/swarm plot — better spread than strip plot."""
        col = params.get("column")
        group_col = params.get("group")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        col = col if col and col in df.columns else (num_cols[0] if num_cols else df.columns[0])
        group_col = group_col if group_col and group_col in df.columns else (cat_cols[0] if cat_cols else None)
        sample = df.sample(min(len(df), 500), random_state=42)
        if group_col:
            fig = px.strip(sample, x=group_col, y=col, color=group_col,
                           stripmode="overlay")
        else:
            fig = px.strip(sample, y=col)
        fig.update_traces(jitter=0.4, marker=dict(size=4, opacity=0.5))
        _style(fig, title=f"{col} — Swarm" + (f" by {group_col}" if group_col else ""),
               showlegend=False)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Swarm plot of '{col}'")

    # ── Comparison bar ────────────────────────────────────────────────────────

    @staticmethod
    def handle_comparison_bar(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Before/after or A vs B comparison bar chart."""
        cols = params.get("columns", [])
        num_cols = df.select_dtypes(include="number").columns.tolist()
        a = cols[0] if len(cols) > 0 and cols[0] in df.columns else (num_cols[0] if len(num_cols) > 0 else None)
        b = cols[1] if len(cols) > 1 and cols[1] in df.columns else (num_cols[1] if len(num_cols) > 1 else None)
        if not a or not b:
            return HandlerResult(success=False, error="Need 2 numeric columns for comparison")
        stats = ["mean", "median", "std", "min", "max"]
        vals_a = [df[a].mean(), df[a].median(), df[a].std(), df[a].min(), df[a].max()]
        vals_b = [df[b].mean(), df[b].median(), df[b].std(), df[b].min(), df[b].max()]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=stats, y=vals_a, name=a, marker_color="#FB8C3C"))
        fig.add_trace(go.Bar(x=stats, y=vals_b, name=b, marker_color="#457B9D"))
        fig.update_layout(barmode="group")
        _style(fig, title=f"{a} vs {b} — Comparison", bargap=0.3)
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Comparison: {a} vs {b}")

    # ── Marimekko / mosaic chart ──────────────────────────────────────────────

    @staticmethod
    def handle_marimekko(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Marimekko/mosaic chart — variable-width stacked bars."""
        cols = params.get("columns", [])
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        x_col = cols[0] if len(cols) > 0 and cols[0] in df.columns else (cat_cols[0] if len(cat_cols) > 0 else None)
        color_col = cols[1] if len(cols) > 1 and cols[1] in df.columns else (cat_cols[1] if len(cat_cols) > 1 else None)
        if not x_col or not color_col:
            return HandlerResult(success=False, error="Need 2 categorical columns for Marimekko")
        ct = pd.crosstab(df[x_col], df[color_col])
        widths = ct.sum(axis=1)
        widths_norm = widths / widths.sum()
        ct_pct = ct.div(ct.sum(axis=1), axis=0)
        colors = px.colors.qualitative.Set2
        fig = go.Figure()
        x_pos = 0.0
        for i, cat in enumerate(ct.index):
            w = float(widths_norm[cat])
            bottom = 0.0
            for j, sub in enumerate(ct.columns):
                h = float(ct_pct.loc[cat, sub])
                fig.add_trace(go.Bar(x=[x_pos + w / 2], y=[h], width=[w],
                                     base=bottom, name=str(sub),
                                     marker_color=colors[j % len(colors)],
                                     showlegend=(i == 0),
                                     hovertext=f"{cat} / {sub}: {ct.loc[cat, sub]}"))
                bottom += h
            x_pos += w
        fig.update_layout(barmode="stack", xaxis=dict(title=x_col, showticklabels=False),
                          yaxis=dict(title="Proportion", tickformat=".0%"))
        _style(fig, title=f"{x_col} x {color_col} — Marimekko")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Marimekko: {x_col} x {color_col}")

    # ── Gauge chart ───────────────────────────────────────────────────────────

    @staticmethod
    def handle_gauge_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Single-value gauge chart — shows a key metric."""
        col = params.get("column")
        agg = params.get("agg", "mean")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col = col if col and col in df.columns else (num_cols[0] if num_cols else None)
        if not col:
            return HandlerResult(success=False, error="No numeric column for gauge")
        data = df[col].dropna()
        agg_funcs = {"mean": data.mean, "median": data.median, "sum": data.sum,
                     "min": data.min, "max": data.max}
        value = float(agg_funcs.get(agg, data.mean)())
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title=dict(text=f"{col} ({agg})"),
            gauge=dict(axis=dict(range=[float(data.min()), float(data.max())]),
                       bar=dict(color="#FB8C3C"),
                       steps=[
                           dict(range=[float(data.min()), float(data.quantile(0.5))], color="#F0F0F0"),
                           dict(range=[float(data.quantile(0.5)), float(data.max())], color="#E8E8E8")
                       ])))
        _style(fig, title=f"{col} — Gauge ({agg}={value:,.2f})")
        return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                             summary=f"Gauge: {col} {agg}={value:,.2f}")
