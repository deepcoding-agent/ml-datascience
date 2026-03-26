"""handle_correlation_rank handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_correlation_rank(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Spearman rank correlation matrix + heatmap."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return HandlerResult(success=False, error="Need at least 2 numeric columns")
    corr = df[num_cols].corr(method="spearman").round(4)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlBu_r", aspect="auto")
    _style(fig, title="Spearman Rank Correlation")
    return HandlerResult(success=True, result_df=corr.reset_index(),
                         charts_plotly=[fig.to_json()],
                         summary=f"Spearman correlation for {len(num_cols)} columns")
