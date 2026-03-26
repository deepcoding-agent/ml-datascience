"""handle_missing_value_analysis handler."""
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


def handle_missing_value_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Deep missing value pattern analysis: co-occurrence, MCAR hint, heatmap."""
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) == 0:
        return HandlerResult(
            success=True, result_df=pd.DataFrame({"status": ["No missing values"]}),
            output_type="query", summary="No missing values found in any column.",
        )

    rows: list[dict] = []
    for c in null_cols.index:
        n_miss = int(null_cols[c])
        pct = round(n_miss / len(df) * 100, 2)
        pattern = "Random" if pct < 5 else "Moderate" if pct < 30 else "Systematic"
        rows.append({"column": c, "missing": n_miss, "pct": pct, "pattern": pattern})

    result_df = pd.DataFrame(rows).sort_values("pct", ascending=False)

    # Co-occurrence: which columns tend to be missing together
    co_occ: list[str] = []
    null_matrix = df[null_cols.index].isnull()
    if len(null_cols) >= 2:
        corr_null = null_matrix.corr()
        for i, c1 in enumerate(null_cols.index):
            for j, c2 in enumerate(null_cols.index):
                if i < j and corr_null.loc[c1, c2] > 0.5:
                    co_occ.append(f"{c1} & {c2} (r={corr_null.loc[c1, c2]:.2f})")

    total_null = int(df.isnull().sum().sum())
    total_cells = len(df) * len(df.columns)

    fig = px.bar(result_df, x="pct", y="column", orientation="h", color="pattern",
                 text="pct", color_discrete_map={"Random": "#2EC4B6", "Moderate": "#FF9F1C", "Systematic": "#E71D36"})
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    _style(fig, title=f"Missing Value Analysis — {len(null_cols)} columns, {total_null:,} cells")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})

    summary = f"{len(null_cols)} columns with missing values ({total_null:,}/{total_cells:,} cells = {total_null/total_cells*100:.1f}%)."
    if co_occ:
        summary += f" Co-occurring: {', '.join(co_occ[:3])}."

    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=[fig.to_json()], summary=summary,
    )
