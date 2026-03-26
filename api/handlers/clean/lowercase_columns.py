"""handle_lowercase_columns handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_lowercase_columns(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Normalize column names to lowercase snake_case."""
    result = df.copy()
    mapping: dict[str, str] = {}
    new_cols = []
    for c in result.columns:
        # CamelCase → snake_case, strip special chars, collapse underscores
        new = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(c))
        new = re.sub(r"[^a-zA-Z0-9_]", "_", new).lower().strip("_")
        new = re.sub(r"_+", "_", new)
        if new != c:
            mapping[c] = new
        new_cols.append(new)
    result.columns = new_cols
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Renamed {len(mapping)} columns to snake_case" if mapping else "All columns already snake_case",
        metadata={"renamed": mapping},
    )
