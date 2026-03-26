"""handle_power_transform handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_power_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Apply Yeo-Johnson or Box-Cox power transform to normalize distributions."""
    col = params.get("column")
    method = params.get("method", "yeo-johnson")
    result = df.copy()

    try:
        from sklearn.preprocessing import PowerTransformer
        cols = [col] if col and col in result.columns else result.select_dtypes(include="number").columns.tolist()
        X = result[cols].dropna()

        pt = PowerTransformer(method=method, standardize=True)
        transformed = pd.DataFrame(pt.fit_transform(X), index=X.index, columns=cols)
        for c in cols:
            result[c] = result[c].astype(float)
        result.loc[X.index, cols] = transformed

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Power transform ({method}) on {len(cols)} columns",
        )
    except Exception as e:
        return HandlerResult(success=False, error=f"Power transform error: {e}")
