"""handle_count_encode handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_count_encode(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Encode categorical values by their absolute count."""
    col = params.get("column")
    result = df.copy()
    cat_cols = (
        [col] if col and col in result.columns
        else result.select_dtypes(include=["object", "category"]).columns.tolist()
    )
    encoded = []
    for c in cat_cols:
        counts = result[c].value_counts()
        result[f"{c}_count"] = result[c].map(counts)
        encoded.append(c)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Count-encoded {len(encoded)} columns",
    )
