"""handle_compare_extremes handler."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_compare_extremes(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Compare the rows with the highest and lowest value of a column.
    Shows side-by-side comparison with all columns + a grouped bar chart."""
    col = params.get("column")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not col or col not in num_cols:
        col = num_cols[0] if num_cols else None
    if col is None:
        return HandlerResult(success=False, error="No numeric column found for comparison")

    max_idx = df[col].idxmax()
    min_idx = df[col].idxmin()
    max_row = df.loc[max_idx]
    min_row = df.loc[min_idx]

    comp = pd.DataFrame({
        "Column": df.columns.tolist(),
        f"Highest {col}": [max_row[c] for c in df.columns],
        f"Lowest {col}": [min_row[c] for c in df.columns],
    })

    # Build difference column for numerics
    diffs: list[str] = []
    for c in df.columns:
        if c in num_cols:
            high = float(max_row[c]) if pd.notna(max_row[c]) else 0
            low = float(min_row[c]) if pd.notna(min_row[c]) else 0
            diff = high - low
            if low != 0:
                pct = abs(diff / low) * 100
                diffs.append(f"{diff:+,.2f} ({pct:.0f}%)")
            else:
                diffs.append(f"{diff:+,.2f}")
        else:
            diffs.append("—")
    comp["Difference"] = diffs

    # Chart: grouped bar for numeric columns
    chart_cols = [c for c in num_cols if c != col][:8]
    if chart_cols:
        chart_data = pd.DataFrame({
            "Column": chart_cols * 2,
            "Value": [float(max_row[c]) if pd.notna(max_row[c]) else 0 for c in chart_cols]
                   + [float(min_row[c]) if pd.notna(min_row[c]) else 0 for c in chart_cols],
            "Type": [f"Highest {col}"] * len(chart_cols) + [f"Lowest {col}"] * len(chart_cols),
        })
        fig = px.bar(
            chart_data, x="Column", y="Value", color="Type", barmode="group",
            color_discrete_map={f"Highest {col}": "#FB8C3C", f"Lowest {col}": "#2EC4B6"},
            text="Value",
        )
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        _style(fig, title=f"Highest vs Lowest {col} — Side by Side")
        fig.update_layout(xaxis_title="Feature", yaxis_title="Value")
        charts = [fig.to_json()]
    else:
        charts = []

    # Rich summary
    summary_lines = [
        f"**Highest {col}**: {max_row[col]:,.2f}" if isinstance(max_row[col], (int, float)) else f"**Highest {col}**: {max_row[col]}",
        f"**Lowest {col}**: {min_row[col]:,.2f}" if isinstance(min_row[col], (int, float)) else f"**Lowest {col}**: {min_row[col]}",
    ]
    for c in num_cols:
        if c != col:
            high = max_row[c] if pd.notna(max_row[c]) else 0
            low = min_row[c] if pd.notna(min_row[c]) else 0
            summary_lines.append(f"  {c}: {high:,.2f} vs {low:,.2f}")

    return HandlerResult(
        success=True, result_df=comp, output_type="query",
        charts_plotly=charts,
        summary="\n".join(summary_lines),
    )
