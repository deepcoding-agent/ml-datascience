"""handle_rfm_analysis handler."""
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


def handle_rfm_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 3:
        return HandlerResult(success=False, error="Need ≥3 numeric columns for RFM scoring")
    cols = num_cols[:3]
    result = df.copy()
    for c in cols:
        result[f"{c}_score"] = pd.qcut(result[c].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    score_cols = [f"{c}_score" for c in cols]
    result["rfm_total"] = result[score_cols].sum(axis=1)
    summary = result[score_cols + ["rfm_total"]].describe().round(2).reset_index()
    fig = px.histogram(result, x="rfm_total", nbins=13, text_auto=True)
    fig.update_traces(marker_color="#FB8C3C")
    _style(fig, title=f"RFM Score Distribution (columns: {', '.join(cols)})")
    return HandlerResult(success=True, result_df=result, output_type="query", charts_plotly=[fig.to_json()],
                         summary=f"RFM scoring on {cols}. Score range: {int(result['rfm_total'].min())}–{int(result['rfm_total'].max())}")
