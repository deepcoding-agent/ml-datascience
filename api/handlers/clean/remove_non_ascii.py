"""handle_remove_non_ascii handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_remove_non_ascii(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Remove non-ASCII characters from string columns."""
    col = params.get("column")
    result = df.copy()
    cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
    for c in cols:
        result[c] = result[c].astype(str).str.encode("ascii", errors="ignore").str.decode("ascii")
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Removed non-ASCII characters from {len(cols)} column(s)",
    )
