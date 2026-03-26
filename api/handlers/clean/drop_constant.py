"""handle_drop_constant handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_drop_constant(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Drop columns where all values are the same (zero information)."""
    result = df.copy()
    nunique = result.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        result = result.drop(columns=constant_cols)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Dropped {len(constant_cols)} constant column(s): {constant_cols}" if constant_cols else "No constant columns found",
    )
