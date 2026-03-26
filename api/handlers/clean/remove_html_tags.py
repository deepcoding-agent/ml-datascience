"""handle_remove_html_tags handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_remove_html_tags(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Strip HTML/XML tags from string columns."""
    col = params.get("column")
    result = df.copy()
    cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
    tag_re = re.compile(r"<[^>]+>")
    for c in cols:
        result[c] = result[c].astype(str).apply(lambda v: html.unescape(tag_re.sub("", v)))
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Stripped HTML tags from {len(cols)} column(s)",
    )
