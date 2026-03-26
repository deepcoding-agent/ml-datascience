"""handle_frequency_table handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_frequency_table(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Frequency table with cumulative percentage."""
    col = params.get("column")
    if not col or col not in df.columns:
        cats = df.select_dtypes(include=["object", "category"]).columns
        col = cats[0] if len(cats) > 0 else df.columns[0]
    vc = df[col].value_counts().reset_index()
    vc.columns = [col, "count"]
    vc["pct"] = (vc["count"] / vc["count"].sum() * 100).round(2)
    vc["cumulative_pct"] = vc["pct"].cumsum().round(2)
    return HandlerResult(success=True, result_df=vc,
                         summary=f"Frequency table for '{col}' ({len(vc)} unique values)")
