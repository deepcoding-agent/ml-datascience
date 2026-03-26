"""handle_entropy_report handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_entropy_report(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Shannon entropy per column (higher = more diverse)."""
    from scipy import stats as sp_stats

    rows = []
    for c in df.columns:
        vc = df[c].dropna().value_counts(normalize=True)
        ent = round(float(sp_stats.entropy(vc, base=2)), 4)
        rows.append({"column": c, "entropy": ent, "unique": int(df[c].nunique()),
                     "dtype": str(df[c].dtype)})
    result = pd.DataFrame(rows).sort_values("entropy", ascending=False).reset_index(drop=True)
    return HandlerResult(success=True, result_df=result, summary="Shannon entropy per column")
