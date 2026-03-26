"""handle_quantile_transform handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_quantile_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Quantile transform — maps values to uniform or normal distribution."""
    col = params.get("column")
    distribution = params.get("distribution", "normal")
    result = df.copy()

    try:
        from sklearn.preprocessing import QuantileTransformer
        cols = (
            [col] if col and col in result.columns
            else result.select_dtypes(include="number").columns.tolist()
        )
        X = result[cols].dropna()
        if len(X) < 2:
            return HandlerResult(success=False, error="Not enough data for quantile transform")

        qt = QuantileTransformer(output_distribution=distribution, random_state=42)
        transformed = pd.DataFrame(qt.fit_transform(X), index=X.index, columns=cols)
        for c in cols:
            result[c] = result[c].astype(float)
        result.loc[X.index, cols] = transformed

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Quantile transform ({distribution}) on {len(cols)} columns",
        )
    except Exception as e:
        return HandlerResult(success=False, error=f"Quantile transform error: {e}")
