"""handle_replace_values handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_replace_values(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")
    old_val = params.get("old_value", "?")
    new_val = params.get("new_value", np.nan)
    result = df.copy()
    if col and col in result.columns:
        result[col] = result[col].replace(old_val, new_val)
        summary = f"Replaced '{old_val}' with '{new_val}' in '{col}'"
    else:
        result = result.replace(old_val, new_val)
        summary = f"Replaced '{old_val}' with '{new_val}' in all columns"
    return HandlerResult(success=True, result_df=result, output_type="generate", summary=summary)
