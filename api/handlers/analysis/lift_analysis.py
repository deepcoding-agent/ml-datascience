"""handle_lift_analysis handler."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger
log = get_logger(__name__)

def handle_lift_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Lift / Gain analysis for a score column vs binary target."""
    score_col = params.get("column") or params.get("score")
    target_col = params.get("target")
    nums = BaseHandler.get_numeric_cols(df)
    if not score_col:
        score_col = nums[0] if nums else None
    if not target_col:
        binary = [c for c in df.columns if df[c].dropna().nunique() == 2]
        target_col = binary[0] if binary else None
    if not score_col or not target_col:
        return HandlerResult(success=False, error="Need score column and binary target column")
    tmp = df[[score_col, target_col]].dropna().sort_values(score_col, ascending=False)
    tmp["_decile"] = pd.qcut(range(len(tmp)), q=10, labels=False) + 1
    total_pos = tmp[target_col].sum()
    base_rate = total_pos / len(tmp)
    rows = []
    for d in range(1, 11):
        mask = tmp["_decile"] <= d
        cum_pct = mask.sum() / len(tmp) * 100
        cum_pos = tmp.loc[mask, target_col].sum()
        gain = cum_pos / total_pos * 100 if total_pos > 0 else 0
        lift = gain / cum_pct if cum_pct > 0 else 1
        rows.append({"decile": d, "cum_pct": round(cum_pct, 1), "gain": round(gain, 1), "lift": round(lift, 2)})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["cum_pct"] for r in rows], y=[r["gain"] for r in rows],
                             mode="lines+markers", name="Gain", line=dict(color="#FB8C3C")))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines",
                             name="Random", line=dict(dash="dash", color="#86868B")))
    _style(fig, title=f"Gain Chart: {score_col} → {target_col}",
           xaxis_title="% Population", yaxis_title="% Gain")
    return HandlerResult(success=True, result_df=pd.DataFrame(rows), charts_plotly=[fig.to_json()],
                         output_type="query", summary=f"Lift analysis: top 10% captures {rows[0]['gain']:.0f}% of positives (lift={rows[0]['lift']:.1f}x)")
