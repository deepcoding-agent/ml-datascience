"""handle_rank_transform handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_rank_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Rank values as percent rank (0-1) for numeric columns."""
    col = params.get("column")
    result = df.copy()
    if col and col in result.columns:
        result[f"{col}_pctrank"] = result[col].rank(pct=True).round(4)
        summary = f"Percent-ranked '{col}' → '{col}_pctrank'"
    else:
        transformed = []
        for c in result.select_dtypes(include="number").columns:
            result[f"{c}_pctrank"] = result[c].rank(pct=True).round(4)
            transformed.append(c)
        summary = f"Percent-ranked {len(transformed)} numeric columns"
    return HandlerResult(success=True, result_df=result, output_type="generate", summary=summary)
