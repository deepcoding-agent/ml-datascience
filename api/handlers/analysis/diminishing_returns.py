"""handle_diminishing_returns handler."""
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


def handle_diminishing_returns(df: pd.DataFrame, params: dict) -> HandlerResult:
    columns = params.get("columns", [])
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(columns) >= 2 and all(c in num_cols for c in columns[:2]):
        x_col, y_col = columns[0], columns[1]
    elif len(num_cols) >= 2:
        x_col, y_col = num_cols[0], num_cols[1]
    else:
        return HandlerResult(success=False, error="Need 2 numeric columns")
    clean = df[[x_col, y_col]].dropna().sort_values(x_col).reset_index(drop=True)
    n = len(clean)
    mid = n // 2
    r_first = float(clean.iloc[:mid][[x_col, y_col]].corr().iloc[0, 1])
    r_second = float(clean.iloc[mid:][[x_col, y_col]].corr().iloc[0, 1])
    diminishing = r_first > r_second and r_first > 0
    fig = px.scatter(clean, x=x_col, y=y_col, trendline="lowess", opacity=0.5)
    fig.update_traces(marker_color="#FB8C3C")
    _style(fig, title=f"Diminishing Returns? {x_col} → {y_col} ({'Yes' if diminishing else 'No'})")
    result_df = pd.DataFrame({"metric": ["r_first_half", "r_second_half", "diminishing"], "value": [round(r_first, 4), round(r_second, 4), diminishing]})
    return HandlerResult(success=True, result_df=result_df, output_type="query", charts_plotly=[fig.to_json()],
                         summary=f"{'Diminishing returns detected' if diminishing else 'No diminishing returns'}: r drops from {r_first:.3f} to {r_second:.3f}")
