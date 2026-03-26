"""handle_remove_empty_rows handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_remove_empty_rows(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Remove rows where all values are null or empty string."""
    result = df.copy()
    before = len(result)
    mask = result.apply(
        lambda row: row.isna().all() or (row.astype(str).str.strip() == "").all(),
        axis=1,
    )
    result = result[~mask].reset_index(drop=True)
    removed = before - len(result)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Removed {removed:,} completely empty rows ({before:,} → {len(result):,})",
    )
