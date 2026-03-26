"""handle_multicollinearity_check handler."""
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


def handle_multicollinearity_check(df: pd.DataFrame, params: dict) -> HandlerResult:
    """VIF-based multicollinearity detection."""
    from sklearn.linear_model import LinearRegression

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return HandlerResult(success=False, error="Need at least 2 numeric columns")

    cols = num_cols[:15]
    clean = df[cols].dropna()
    if len(clean) < 10:
        return HandlerResult(success=False, error="Need at least 10 non-null rows")

    rows: list[dict] = []
    X = clean[cols].values
    for i, c in enumerate(cols):
        y_i = X[:, i]
        X_i = np.delete(X, i, axis=1)
        if X_i.shape[1] == 0:
            continue
        r2 = float(LinearRegression().fit(X_i, y_i).score(X_i, y_i))
        vif = 1.0 / (1.0 - r2) if r2 < 1 else float("inf")
        concern = "High" if vif > 10 else "Moderate" if vif > 5 else "Low"
        rows.append({"feature": c, "VIF": round(vif, 2), "R2_other": round(r2, 4), "concern": concern})

    result_df = pd.DataFrame(rows).sort_values("VIF", ascending=False)

    fig = px.bar(result_df, x="VIF", y="feature", orientation="h", color="concern",
                 text="VIF", color_discrete_map={"Low": "#2EC4B6", "Moderate": "#FF9F1C", "High": "#E71D36"})
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    _style(fig, title=f"Multicollinearity Check (VIF) — {len(cols)} features")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})

    high_vif = [r for r in rows if r["VIF"] > 10]
    worst = rows[0] if rows else {"feature": "N/A", "VIF": 0}
    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"VIF analysis: {len(high_vif)} features with high multicollinearity (VIF>10). Worst: {worst['feature']} (VIF={worst['VIF']:.1f})",
    )
