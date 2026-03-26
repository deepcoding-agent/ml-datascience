"""handle_reset_index handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_reset_index(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Reset the DataFrame index to 0-based sequential."""
    result = df.reset_index(drop=True)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Reset index (0 to {len(result)-1})",
    )
