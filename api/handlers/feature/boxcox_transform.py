"""handle_boxcox_transform handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_boxcox_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Box-Cox transform — requires strictly positive values."""
    col = params.get("column")
    result = df.copy()
    try:
        from scipy.stats import boxcox
        cols = (
            [col] if col and col in result.columns
            else result.select_dtypes(include="number").columns.tolist()
        )
        transformed = []
        for c in cols:
            vals = result[c].dropna()
            if (vals > 0).all() and len(vals) > 1:
                t_vals, lam = boxcox(vals.values)
                result[f"{c}_boxcox"] = np.nan
                result.loc[vals.index, f"{c}_boxcox"] = np.round(t_vals, 4)
                transformed.append(f"{c} (lambda={lam:.3f})")
        if not transformed:
            return HandlerResult(success=False, error="No columns with all-positive values found")
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Box-Cox transformed {len(transformed)} columns: {transformed}",
        )
    except Exception as e:
        return HandlerResult(success=False, error=f"Box-Cox error: {e}")
