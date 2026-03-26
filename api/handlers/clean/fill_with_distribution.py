"""handle_fill_with_distribution handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_fill_with_distribution(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Fill nulls by sampling from the column's existing distribution."""
    col = params.get("column")
    seed = params.get("seed", 42)
    result = df.copy()
    rng = np.random.default_rng(seed)
    cols = [col] if col and col in result.columns else result.columns.tolist()
    filled_count = 0

    for c in cols:
        null_mask = result[c].isna()
        n_nulls = int(null_mask.sum())
        if n_nulls == 0:
            continue
        non_null = result[c].dropna().values
        if len(non_null) == 0:
            continue
        sampled = rng.choice(non_null, size=n_nulls, replace=True)
        result.loc[null_mask, c] = sampled
        filled_count += n_nulls

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Filled {filled_count:,} nulls by sampling from column distributions",
    )
