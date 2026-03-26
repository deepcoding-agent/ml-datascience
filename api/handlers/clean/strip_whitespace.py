"""handle_strip_whitespace handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_strip_whitespace(df: pd.DataFrame, params: dict) -> HandlerResult:
    result = df.copy()
    str_cols = result.select_dtypes(include="object").columns
    for col in str_cols:
        result[col] = result[col].str.strip()
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Stripped whitespace from {len(str_cols)} string columns")
