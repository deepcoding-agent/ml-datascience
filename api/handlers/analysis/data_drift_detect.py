"""handle_data_drift_detect handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
    breakpoints = np.linspace(min(expected.min(), actual.min()),
                              max(expected.max(), actual.max()), bins + 1)
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    # Avoid zero
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def handle_data_drift_detect(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Compare two halves of dataset for distribution drift using KS test and PSI."""
    from scipy.stats import ks_2samp

    split = params.get("split", "half")  # "half" or column name for temporal split
    column = params.get("column")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if column and column in num_cols:
        num_cols = [column]
    if not num_cols:
        return HandlerResult(success=False, error="No numeric columns found for drift detection.")

    mid = len(df) // 2
    df1 = df.iloc[:mid]
    df2 = df.iloc[mid:]

    results = []
    for col in num_cols:
        s1 = df1[col].dropna().values
        s2 = df2[col].dropna().values
        if len(s1) < 5 or len(s2) < 5:
            continue
        stat, p_value = ks_2samp(s1, s2)
        psi_val = _psi(s1, s2)
        drift = "No drift" if p_value > 0.05 and psi_val < 0.1 else ("Moderate drift" if psi_val < 0.25 else "Significant drift")
        results.append({
            "column": col,
            "ks_statistic": round(float(stat), 4),
            "ks_p_value": round(float(p_value), 6),
            "psi": round(psi_val, 4),
            "drift_status": drift,
        })

    if not results:
        return HandlerResult(success=False, error="Insufficient data for drift detection.")

    result_df = pd.DataFrame(results).sort_values("psi", ascending=False)
    drifted = result_df[result_df["drift_status"] != "No drift"]

    # Chart: top drifted columns by PSI
    top = result_df.head(10)
    fig = go.Figure(go.Bar(
        x=top["psi"], y=top["column"], orientation="h",
        marker_color=["#E8500A" if d != "No drift" else "#6B6B6B" for d in top["drift_status"]],
        text=top["drift_status"], textposition="auto",
    ))
    _style(fig, title=f"Data Drift (PSI) — {len(drifted)} of {len(results)} columns drifted")

    return HandlerResult(
        success=True, result_df=result_df, charts_plotly=[fig.to_json()],
        summary=f"Drift detection: {len(drifted)}/{len(results)} columns show drift. Top drifted: {drifted['column'].tolist()[:5]}",
        metadata={"drifted_columns": drifted["column"].tolist()},
    )
