"""handle_correlation handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_correlation(df: pd.DataFrame, params: dict) -> HandlerResult:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return HandlerResult(success=False, error="Need at least 2 numeric columns for correlation")
    corr = df[num_cols].corr().round(4)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlBu_r",
                    aspect="auto")
    _style(fig, title="Correlation Matrix")
    return HandlerResult(
        success=True, result_df=corr.reset_index(),
        charts_plotly=[fig.to_json()],
        summary=f"Correlation matrix for {len(num_cols)} numeric columns",
    )
