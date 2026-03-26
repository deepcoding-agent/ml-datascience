"""handle_kurtosis handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_kurtosis(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Kurtosis per numeric column (complement to skewness)."""
    num_cols = df.select_dtypes(include="number").columns
    kurt = df[num_cols].kurtosis().round(4)
    result = kurt.reset_index()
    result.columns = ["column", "kurtosis"]
    result["shape"] = result["kurtosis"].apply(
        lambda k: "Normal-like" if abs(k) < 1 else ("Heavy-tailed" if k > 1 else "Light-tailed")
    )
    result = result.sort_values("kurtosis", key=abs, ascending=False).reset_index(drop=True)
    return HandlerResult(success=True, result_df=result, summary="Kurtosis per numeric column")
