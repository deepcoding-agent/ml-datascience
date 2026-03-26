"""handle_cardinality_report handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_cardinality_report(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Analyze unique ratio per column — identify high/low cardinality."""
    rows = []
    for c in df.columns:
        n_unique = df[c].nunique()
        ratio = round(n_unique / len(df) * 100, 2) if len(df) > 0 else 0
        if ratio > 90:
            label = "ID-like"
        elif n_unique <= 2:
            label = "Binary"
        elif n_unique <= 10:
            label = "Low"
        elif ratio > 50:
            label = "High"
        else:
            label = "Medium"
        rows.append({"column": c, "unique": n_unique, "unique_pct": ratio,
                     "dtype": str(df[c].dtype), "cardinality": label})

    result = pd.DataFrame(rows).sort_values("unique", ascending=False).reset_index(drop=True)
    return HandlerResult(
        success=True, result_df=result,
        summary=f"Cardinality: {sum(1 for r in rows if r['cardinality'] == 'ID-like')} ID-like, "
                f"{sum(1 for r in rows if r['cardinality'] == 'Binary')} binary, "
                f"{sum(1 for r in rows if r['cardinality'] == 'High')} high, "
                f"{sum(1 for r in rows if r['cardinality'] in ('Low', 'Medium'))} low/medium",
    )
