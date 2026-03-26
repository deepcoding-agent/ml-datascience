"""handle_sample_rows handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_sample_rows(df: pd.DataFrame, params: dict) -> HandlerResult:
    n = min(params.get("n", 10), len(df))
    result = df.sample(n=n, random_state=42)
    return HandlerResult(success=True, result_df=result, output_type="query",
                         summary=f"Random sample of {n} rows")
