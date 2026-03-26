"""handle_hash_encode handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_hash_encode(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Hash encoding for high-cardinality categorical columns."""
    col = params.get("column")
    n_features = params.get("n", 8)
    result = df.copy()
    cat_cols = (
        [col] if col and col in result.columns
        else result.select_dtypes(include=["object", "category"]).columns.tolist()
    )
    created = []
    for c in cat_cols:
        for i in range(n_features):
            result[f"{c}_hash{i}"] = result[c].astype(str).apply(
                lambda x, _i=i: (hash(x + str(_i)) % 2)
            )
            created.append(f"{c}_hash{i}")
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Hash-encoded {len(cat_cols)} columns into {n_features} features each ({len(created)} total)",
    )
