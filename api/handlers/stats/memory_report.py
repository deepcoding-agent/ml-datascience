"""handle_memory_report handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_memory_report(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Detailed memory usage per column."""
    mem = df.memory_usage(deep=True)
    total = mem.sum()
    rows = []
    for c in df.columns:
        m = int(mem[c])
        rows.append({
            "column": c, "dtype": str(df[c].dtype),
            "memory_bytes": m, "memory_kb": round(m / 1024, 2),
            "pct_of_total": round(m / total * 100, 2),
        })
    result = pd.DataFrame(rows).sort_values("memory_bytes", ascending=False).reset_index(drop=True)
    total_mb = round(total / 1024 ** 2, 2)
    return HandlerResult(success=True, result_df=result,
                         summary=f"Total memory: {total_mb} MB across {len(df.columns)} columns")
