"""handle_remove_duplicates handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_remove_duplicates(df: pd.DataFrame, params: dict) -> HandlerResult:
    before = len(df)
    result = df.drop_duplicates()
    removed = before - len(result)
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Removed {removed:,} duplicate rows ({before:,} → {len(result):,})")
