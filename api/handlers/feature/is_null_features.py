"""handle_is_null_features handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_is_null_features(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Create binary is_null indicator columns for each column with nulls."""
    result = df.copy()
    created = []
    cols_with_nulls = [c for c in result.columns if result[c].isnull().any()]
    for c in cols_with_nulls:
        result[f"{c}_is_null"] = result[c].isnull().astype(int)
        created.append(f"{c}_is_null")
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Created {len(created)} is_null indicator columns",
    )
