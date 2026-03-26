"""handle_ratio_features handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_ratio_features(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Create ratio features (col_a / col_b) for numeric column pairs."""
    cols = params.get("columns", [])
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cols = [c for c in cols if c in num_cols] or num_cols[:4]
    result = df.copy()
    created = []

    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            safe = result[c2].replace(0, np.nan)
            result[f"{c1}_per_{c2}"] = (result[c1] / safe).round(4)
            created.append(f"{c1}_per_{c2}")

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Created {len(created)} ratio features",
    )
