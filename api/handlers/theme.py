"""Shared Plotly theme and helpers used across handler categories."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

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
