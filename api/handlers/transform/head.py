"""handle_head handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_head(df: pd.DataFrame, params: dict) -> HandlerResult:
    n = params.get("n", 10)
    result = df.head(n)
    return HandlerResult(success=True, result_df=result, output_type="query",
                         summary=f"First {n} rows")
