"""handle_encode_label handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_encode_label(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")
    result = df.copy()
    cols = [col] if col and col in result.columns else result.select_dtypes(include=["object", "category"]).columns.tolist()
    mappings = {}
    for c in cols:
        le = LabelEncoder()
        result[c] = le.fit_transform(result[c].astype(str))
        mappings[c] = {str(cls): int(i) for i, cls in enumerate(le.classes_)}
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Label-encoded {len(cols)} columns", metadata={"mappings": mappings})
