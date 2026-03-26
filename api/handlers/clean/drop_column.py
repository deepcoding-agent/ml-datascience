"""handle_drop_column handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_drop_column(df: pd.DataFrame, params: dict) -> HandlerResult:
    cols = params.get("columns") or ([params["column"]] if params.get("column") else [])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return HandlerResult(success=False, error=f"Columns not found: {missing}")
    result = df.drop(columns=cols)
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Dropped {len(cols)} column(s): {cols}")
