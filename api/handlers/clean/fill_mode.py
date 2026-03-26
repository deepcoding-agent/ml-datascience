"""handle_fill_mode handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_fill_mode(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Fill nulls with mode (most frequent value)."""
    col = params.get("column")
    result = df.copy()
    cols = [col] if col and col in result.columns else result.columns.tolist()
    filled: dict[str, str] = {}

    for c in cols:
        if result[c].isnull().sum() == 0:
            continue
        mode = result[c].mode()
        if not mode.empty:
            result[c] = result[c].fillna(mode.iloc[0])
            filled[c] = str(mode.iloc[0])

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Filled nulls with mode in {len(filled)} column(s)",
        metadata={"mode_values": filled},
    )
