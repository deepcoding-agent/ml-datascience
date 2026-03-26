"""handle_cyclical_encode handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_cyclical_encode(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Encode cyclical features (month, dayofweek, hour) using sin/cos."""
    col = params.get("column")
    period = params.get("period")
    if not col or col not in df.columns:
        return HandlerResult(success=False, error=f"Column '{col}' not found")

    result = df.copy()
    values = pd.to_numeric(result[col], errors="coerce")

    if period is None:
        max_val = values.max()
        period = int(max_val) + 1 if max_val and max_val > 0 else 12

    result[f"{col}_sin"] = np.sin(2 * np.pi * values / period).round(4)
    result[f"{col}_cos"] = np.cos(2 * np.pi * values / period).round(4)

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Cyclical encoding '{col}' (period={period}) → {col}_sin, {col}_cos",
    )
