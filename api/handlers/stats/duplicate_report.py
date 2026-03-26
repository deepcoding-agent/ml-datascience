"""handle_duplicate_report handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_duplicate_report(df: pd.DataFrame, params: dict) -> HandlerResult:
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        sample = df[df.duplicated(keep="first")].head(5)
        return HandlerResult(success=True, result_df=sample,
                             summary=f"{dup_count:,} duplicate rows found (showing first 5)")
    return HandlerResult(success=True,
                         result_df=pd.DataFrame([{"duplicate_rows": 0}]),
                         summary="No duplicate rows found")
