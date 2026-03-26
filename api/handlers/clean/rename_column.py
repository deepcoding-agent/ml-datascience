"""handle_rename_column handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_rename_column(df: pd.DataFrame, params: dict) -> HandlerResult:
    old_name = params.get("column")
    new_name = params.get("new_name") or params.get("value")
    if not old_name or not new_name:
        return HandlerResult(success=False, error="Need both old and new column names")
    if old_name not in df.columns:
        return HandlerResult(success=False, error=f"Column '{old_name}' not found")
    result = df.rename(columns={old_name: str(new_name)})
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Renamed '{old_name}' → '{new_name}'")
