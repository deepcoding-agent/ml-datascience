"""handle_cohort_analysis handler."""
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


def handle_cohort_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")
    value_col = params.get("value_column")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    col = col if col and col in df.columns else (cat_cols[0] if cat_cols else None)
    value_col = value_col if value_col and value_col in num_cols else (num_cols[0] if num_cols else None)
    if not col or not value_col:
        return HandlerResult(success=False, error="Need categorical column + numeric value column")
    cohorts = df.groupby(col)[value_col].agg(["count", "mean", "median", "sum", "std"]).round(2).reset_index()
    cohorts = cohorts.sort_values("mean", ascending=False)
    fig = px.bar(cohorts, x=col, y="mean", text="count", color="mean", color_continuous_scale="YlOrRd")
    fig.update_traces(texttemplate="n=%{text}", textposition="outside")
    _style(fig, title=f"Cohort Analysis — {value_col} by {col}")
    return HandlerResult(success=True, result_df=cohorts, output_type="query", charts_plotly=[fig.to_json()],
                         summary=f"Analyzed {len(cohorts)} cohorts by '{col}'. Mean '{value_col}' range: {cohorts['mean'].min():.2f}–{cohorts['mean'].max():.2f}")
