"""handle_remove_urls handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_remove_urls(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Remove URLs from text columns."""
    col = params.get("column")
    result = df.copy()
    cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
    url_re = re.compile(r"https?://\S+|www\.\S+")
    for c in cols:
        result[c] = result[c].astype(str).str.replace(url_re, "", regex=True).str.strip()
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Removed URLs from {len(cols)} column(s)",
    )
