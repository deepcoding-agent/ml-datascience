"""handle_bin_numeric handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_bin_numeric(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Bin numeric column into custom bins with optional labels."""
    col = params.get("column")
    n_bins = params.get("n", 5)
    labels = params.get("labels")
    if not col or col not in df.columns:
        return HandlerResult(success=False, error=f"Column '{col}' not found")
    result = df.copy()
    try:
        if labels and len(labels) == n_bins:
            result[f"{col}_bin"] = pd.cut(result[col], bins=n_bins, labels=labels)
        else:
            result[f"{col}_bin"] = pd.cut(result[col], bins=n_bins)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Binned '{col}' into {n_bins} bins → '{col}_bin'",
        )
    except Exception as e:
        return HandlerResult(success=False, error=f"Bin numeric error: {e}")
