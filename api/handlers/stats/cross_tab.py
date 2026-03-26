"""handle_cross_tab handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_cross_tab(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Contingency table between two columns (count or normalized)."""
    cols = params.get("columns", [])
    normalize = params.get("normalize", False)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    col1 = cols[0] if len(cols) > 0 and cols[0] in df.columns else (cat_cols[0] if len(cat_cols) > 0 else None)
    col2 = cols[1] if len(cols) > 1 and cols[1] in df.columns else (cat_cols[1] if len(cat_cols) > 1 else None)

    if not col1 or not col2:
        return HandlerResult(success=False, error="Need 2 columns for cross-tabulation")

    ct = pd.crosstab(df[col1], df[col2], normalize="all" if normalize else False)
    if normalize:
        ct = (ct * 100).round(2)
    result = ct.reset_index()

    fig = px.imshow(ct, text_auto=True, aspect="auto",
                    color_continuous_scale=["#FFF5EB", "#FB8C3C"])
    _style(fig, title=f"{col1} vs {col2}")

    return HandlerResult(
        success=True, result_df=result, charts_plotly=[fig.to_json()],
        summary=f"Cross-tab: {col1} × {col2} ({ct.shape[0]}×{ct.shape[1]})",
    )
