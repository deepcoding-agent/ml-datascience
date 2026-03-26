"""handle_sparsity_report handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_sparsity_report(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Sparsity analysis: zeros + nulls per column."""
    rows = []
    total_cells = len(df)
    for c in df.columns:
        null_n = int(df[c].isna().sum())
        zero_n = int((df[c] == 0).sum()) if pd.api.types.is_numeric_dtype(df[c]) else 0
        sparse_n = null_n + zero_n
        rows.append({
            "column": c, "null_count": null_n, "zero_count": zero_n,
            "sparse_count": sparse_n,
            "sparsity_pct": round(sparse_n / total_cells * 100, 2) if total_cells > 0 else 0,
        })
    result = pd.DataFrame(rows).sort_values("sparsity_pct", ascending=False).reset_index(drop=True)
    overall = sum(r["sparse_count"] for r in rows) / (total_cells * len(df.columns)) * 100 if total_cells > 0 else 0
    return HandlerResult(success=True, result_df=result,
                         summary=f"Overall sparsity: {overall:.1f}% (zeros + nulls)")
