"""handle_systematic_sample handler."""
from __future__ import annotations

import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_systematic_sample(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Systematic sampling — select every N-th row from the dataset."""
    step = int(params.get("step", 10))
    offset = int(params.get("offset", 0))

    if step < 1:
        step = 10
    if offset < 0 or offset >= len(df):
        offset = 0

    result = df.iloc[offset::step].reset_index(drop=True)

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Systematic sample: every {step}th row (offset={offset}). {len(df)} → {len(result)} rows.",
    )
