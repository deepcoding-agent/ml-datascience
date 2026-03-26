"""handle_auto_feature_select handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_auto_feature_select(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Auto-select features: drop low-variance + highly-correlated columns."""
    var_threshold = params.get("var_threshold", 0.01)
    corr_threshold = params.get("corr_threshold", 0.95)
    result = df.copy()
    num_cols = result.select_dtypes(include="number").columns.tolist()
    dropped = []
    # Low variance
    variances = result[num_cols].var()
    low_var = variances[variances < var_threshold].index.tolist()
    result = result.drop(columns=low_var)
    dropped.extend(low_var)
    # High correlation
    remaining = [c for c in num_cols if c not in low_var]
    if len(remaining) >= 2:
        corr = result[remaining].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        high_corr = [c for c in upper.columns if upper[c].max() > corr_threshold]
        result = result.drop(columns=high_corr)
        dropped.extend(high_corr)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Auto-selected features: dropped {len(dropped)} columns (low-var: {low_var}, high-corr: {high_corr if len(remaining) >= 2 else []})",
    )
