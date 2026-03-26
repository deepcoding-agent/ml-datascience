"""handle_interaction_features handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_interaction_features(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Multiply pairs of numeric columns to create interaction features."""
    cols = params.get("columns", [])
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cols = [c for c in cols if c in num_cols] or num_cols[:4]
    result = df.copy()
    created = []
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            name = f"{c1}_x_{c2}"
            result[name] = result[c1] * result[c2]
            created.append(name)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Created {len(created)} interaction features from {len(cols)} columns",
    )
