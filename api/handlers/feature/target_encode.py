"""handle_target_encode handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_target_encode(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Mean/target encoding: replace categorical values with the mean of the target."""
    target = params.get("target") or params.get("column")
    col = params.get("encode_column")
    if not target or target not in df.columns:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        target = num_cols[-1] if num_cols else None
    if target is None:
        return HandlerResult(success=False, error="No target column found")

    result = df.copy()
    cat_cols = [col] if col and col in result.columns else result.select_dtypes(include=["object", "category"]).columns.tolist()
    encoded = []

    global_mean = result[target].mean()
    for c in cat_cols:
        means = result.groupby(c)[target].mean()
        result[f"{c}_target_enc"] = result[c].map(means).fillna(global_mean)
        encoded.append(c)

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Target-encoded {len(encoded)} columns using target='{target}'",
    )
