"""handle_woe_encode handler."""
from __future__ import annotations
import numpy as np
import pandas as pd
from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger
log = get_logger(__name__)

def handle_woe_encode(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Weight of Evidence encoding for categorical columns (requires binary target)."""
    col = params.get("column")
    target = params.get("target")
    cats = BaseHandler.get_categorical_cols(df)
    if not col:
        col = cats[0] if cats else None
    if not col or col not in df.columns:
        return HandlerResult(success=False, error=f"Column required. Available: {cats}")
    if not target or target not in df.columns:
        return HandlerResult(success=False, error="Binary target column required")
    if df[target].nunique() != 2:
        return HandlerResult(success=False, error=f"Target \'{target}\' must be binary (has {df[target].nunique()} values)")
    result = df.copy()
    total_events = df[target].sum()
    total_non = len(df) - total_events
    woe_map = {}
    for cat in df[col].unique():
        mask = df[col] == cat
        events = df.loc[mask, target].sum()
        non_events = mask.sum() - events
        dist_events = (events / total_events) if total_events > 0 else 0.0001
        dist_non = (non_events / total_non) if total_non > 0 else 0.0001
        dist_events = max(dist_events, 0.0001)
        dist_non = max(dist_non, 0.0001)
        woe_map[cat] = round(np.log(dist_non / dist_events), 4)
    result[f"{col}_woe"] = result[col].map(woe_map)
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"WoE encoded \'{col}\' against target \'{target}\' ({len(woe_map)} categories)")
