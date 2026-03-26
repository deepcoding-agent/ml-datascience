"""handle_lowercase_values handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_lowercase_values(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Lowercase all string values in specified or all string columns."""
    col = params.get("column")
    result = df.copy()
    if col and col in result.columns:
        cols = [col]
    else:
        cols = result.select_dtypes(include="object").columns.tolist()
    for c in cols:
        result[c] = result[c].str.lower()
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Lowercased values in {len(cols)} column(s): {cols}",
    )
