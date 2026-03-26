"""handle_outlier_isolation_forest handler."""
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


def handle_outlier_isolation_forest(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Isolation Forest anomaly detection + scatter visualization."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    contamination = float(params.get("contamination", 0.05))
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return HandlerResult(success=False, error="Need at least 2 numeric columns")

    cols = num_cols[:10]
    X = df[cols].dropna()
    if len(X) < 10:
        return HandlerResult(success=False, error="Need at least 10 non-null rows")

    X_scaled = StandardScaler().fit_transform(X)
    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    preds = iso.fit_predict(X_scaled)

    X_out = X.copy()
    X_out["anomaly"] = ["Anomaly" if p == -1 else "Normal" for p in preds]
    n_anomalies = int((preds == -1).sum())

    fig = px.scatter(
        X_out, x=cols[0], y=cols[1], color="anomaly",
        color_discrete_map={"Anomaly": "#E71D36", "Normal": "#86868B"},
    )
    fig.update_traces(marker_size=4)
    _style(fig, title=f"Isolation Forest — {n_anomalies} anomalies ({n_anomalies/len(X)*100:.1f}%)")

    result_df = pd.DataFrame({"label": ["Normal", "Anomaly"], "count": [len(X) - n_anomalies, n_anomalies]})
    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"Isolation Forest found {n_anomalies} anomalies ({n_anomalies/len(X)*100:.1f}%) in {len(X)} rows",
    )
