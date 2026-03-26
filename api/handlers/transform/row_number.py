"""handle_row_number handler."""
from __future__ import annotations

import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_row_number(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Add a sequential row number column (1-based by default)."""
    try:
        new_name = params.get("new_name", "_row_number")
        start = int(params.get("start", 1))

        result = df.copy()
        result[new_name] = range(start, start + len(df))

        return HandlerResult(
            success=True,
            result_df=result,
            output_type="generate",
            summary=(
                f"Added row number column '{new_name}' "
                f"(range {start}..{start + len(df) - 1}, {len(df)} rows)"
            ),
        )
    except Exception as e:
        log.exception("row_number failed")
        return HandlerResult(success=False, error=f"row_number failed: {e}")
