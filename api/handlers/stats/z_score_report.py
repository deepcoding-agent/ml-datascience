"""handle_z_score_report handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_z_score_report(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Z-score analysis: flag extreme z-scores per column."""
    col = params.get("column")
    num_cols = [col] if col and col in df.columns else df.select_dtypes(include="number").columns.tolist()
    rows = []
    for c in num_cols:
        data = df[c].dropna()
        if data.std() == 0:
            rows.append({"column": c, "max_abs_z": 0, "n_beyond_2": 0, "n_beyond_3": 0})
            continue
        z = ((data - data.mean()) / data.std()).abs()
        rows.append({
            "column": c,
            "max_abs_z": round(float(z.max()), 4),
            "n_beyond_2": int((z > 2).sum()),
            "n_beyond_3": int((z > 3).sum()),
        })
    result = pd.DataFrame(rows).sort_values("max_abs_z", ascending=False).reset_index(drop=True)
    return HandlerResult(success=True, result_df=result, summary=f"Z-score analysis for {len(num_cols)} columns")
