"""handle_aggregation_features handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_aggregation_features(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Create group-by aggregation features (mean/std/count per group)."""
    group_col = params.get("column")
    agg_col = params.get("agg_column")
    result = df.copy()

    cat_cols = result.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = result.select_dtypes(include="number").columns.tolist()

    if not group_col or group_col not in result.columns:
        group_col = cat_cols[0] if cat_cols else None
    if group_col is None:
        return HandlerResult(success=False, error="No categorical column found for grouping")

    agg_cols = [agg_col] if agg_col and agg_col in num_cols else num_cols[:3]
    created = []

    for c in agg_cols:
        for agg in ["mean", "std", "count"]:
            col_name = f"{c}_{agg}_by_{group_col}"
            result[col_name] = result.groupby(group_col)[c].transform(agg)
            if agg != "count":
                result[col_name] = result[col_name].round(4)
            created.append(col_name)

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Created {len(created)} aggregation features grouped by '{group_col}'",
    )
