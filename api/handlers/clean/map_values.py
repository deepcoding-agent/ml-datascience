"""handle_map_values handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_map_values(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Map/recode values in a column using a mapping dict.

    params: column, mapping (dict e.g. {"M": "Male", "F": "Female"})
    """
    col = params.get("column")
    mapping = params.get("mapping", {})
    if not col or col not in df.columns:
        return HandlerResult(success=False, error=f"Column '{col}' not found")
    if not mapping:
        return HandlerResult(success=False, error="No mapping provided")
    result = df.copy()
    result[col] = result[col].replace(mapping)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Mapped {len(mapping)} values in '{col}': {mapping}",
    )
