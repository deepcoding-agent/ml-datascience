"""handle_mode_report handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_mode_report(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Mode values per column."""
    rows = []
    for c in df.columns:
        modes = df[c].mode()
        mode_val = str(modes.iloc[0]) if len(modes) > 0 else "N/A"
        mode_count = int((df[c] == modes.iloc[0]).sum()) if len(modes) > 0 else 0
        rows.append({
            "column": c, "mode": mode_val, "mode_count": mode_count,
            "mode_pct": round(mode_count / len(df) * 100, 2) if len(df) > 0 else 0,
            "n_modes": len(modes),
        })
    result = pd.DataFrame(rows)
    return HandlerResult(success=True, result_df=result, summary="Mode values per column")
