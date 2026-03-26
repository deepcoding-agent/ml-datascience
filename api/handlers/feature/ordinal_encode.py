"""handle_ordinal_encode handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_ordinal_encode(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Ordinal encoding with auto-detected order (sorted unique values)."""
    col = params.get("column")
    result = df.copy()
    cat_cols = (
        [col] if col and col in result.columns
        else result.select_dtypes(include=["object", "category"]).columns.tolist()
    )
    encoded = []
    for c in cat_cols:
        sorted_vals = sorted(result[c].dropna().unique())
        mapping = {v: i for i, v in enumerate(sorted_vals)}
        result[f"{c}_ordinal"] = result[c].map(mapping)
        encoded.append(c)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Ordinal-encoded {len(encoded)} columns (alphabetical order)",
    )
