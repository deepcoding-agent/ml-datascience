"""handle_anomaly_detect handler."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_anomaly_detect(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Detect anomalies using IQR and Z-score methods across numeric columns.
    Returns flagged rows + anomaly summary + scatter chart."""
    col = params.get("column")
    method = params.get("method", "iqr")  # iqr | zscore
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if col and col in num_cols:
        check_cols = [col]
    else:
        check_cols = num_cols[:5]

    if not check_cols:
        return HandlerResult(success=False, error="No numeric columns for anomaly detection")

    result = df.copy()
    anomaly_flags = pd.Series(False, index=df.index)
    anomaly_cols: dict[str, int] = {}

    for c in check_cols:
        s = result[c].dropna()
        if method == "zscore":
            z = (s - s.mean()) / s.std()
            mask = z.abs() > 3
        else:  # iqr
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)

        col_anomalies = mask.sum()
        if col_anomalies > 0:
            anomaly_cols[c] = int(col_anomalies)
        anomaly_flags |= mask.reindex(df.index, fill_value=False)

    result["_is_anomaly"] = anomaly_flags
    total_anomalies = int(anomaly_flags.sum())
    anomaly_rows = result[result["_is_anomaly"]]

    # Build summary table
    summary_rows = [{"column": c, "anomaly_count": cnt, "anomaly_pct": round(cnt / len(df) * 100, 2)}
                    for c, cnt in anomaly_cols.items()]
    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame(columns=["column", "anomaly_count", "anomaly_pct"])

    # Scatter chart of first two anomaly columns
    charts: list[str] = []
    if len(check_cols) >= 2:
        fig = px.scatter(
            result, x=check_cols[0], y=check_cols[1],
            color="_is_anomaly",
            color_discrete_map={True: "#E71D36", False: "#86868B"},
        )
        _style(fig, title=f"Anomaly Detection ({method.upper()}) — {total_anomalies} anomalies in {len(df)} rows")
        fig.update_layout(xaxis_title=check_cols[0], yaxis_title=check_cols[1])
        charts.append(fig.to_json())

    return HandlerResult(
        success=True, result_df=summary_df, output_type="query",
        charts_plotly=charts,
        summary=f"Found {total_anomalies} anomalous rows ({total_anomalies/max(len(df),1)*100:.1f}%) using {method.upper()} method across {len(check_cols)} columns",
        metadata={"anomaly_details": anomaly_cols, "total_anomalies": total_anomalies},
    )
