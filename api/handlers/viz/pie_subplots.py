"""handle_pie_subplots handler."""
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger
log = get_logger(__name__)

def handle_pie_subplots(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Multiple pie charts side by side for comparing categorical columns."""
    columns = params.get("columns", [])
    cats = BaseHandler.get_categorical_cols(df)
    if not columns:
        columns = [c for c in cats if df[c].nunique() <= 10][:4]
    if not columns:
        return HandlerResult(success=False, error="No suitable categorical columns (<=10 unique)")
    n = len(columns)
    fig = make_subplots(rows=1, cols=n, specs=[[{"type": "pie"}]*n],
                        subplot_titles=columns)
    colors = ["#FB8C3C", "#2EC4B6", "#457B9D", "#E71D36", "#FF9F1C", "#A8DADC"]
    for i, col in enumerate(columns):
        vc = df[col].value_counts().head(8)
        fig.add_trace(go.Pie(labels=vc.index.tolist(), values=vc.values.tolist(),
                             name=col, marker=dict(colors=colors)), row=1, col=i+1)
    _style(fig, title="Categorical Distribution Comparison", showlegend=False)
    return HandlerResult(success=True, charts_plotly=[fig.to_json()], output_type="query",
                         summary=f"Pie subplots for {columns}")
