"""handle_encode_onehot handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_encode_onehot(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")
    result = df.copy()
    cols = [col] if col and col in result.columns else result.select_dtypes(include=["object", "category"]).columns.tolist()
    result = pd.get_dummies(result, columns=cols, drop_first=True, dtype=int)
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"One-hot encoded {len(cols)} columns → {result.shape[1]} total cols")
