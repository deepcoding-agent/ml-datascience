"""handle_timeline_chart handler."""
from __future__ import annotations
import pandas as pd
import plotly.express as px
from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger
log = get_logger(__name__)

def handle_timeline_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Timeline / Gantt chart — requires start and end date columns."""
    start_col = params.get("start") or params.get("column")
    end_col = params.get("end")
    label_col = params.get("label")
    dt_cols = BaseHandler.get_datetime_cols(df)
    cats = BaseHandler.get_categorical_cols(df)
    if not start_col and len(dt_cols) >= 1:
        start_col = dt_cols[0]
    if not end_col and len(dt_cols) >= 2:
        end_col = dt_cols[1]
    if not label_col and cats:
        label_col = cats[0]
    if not start_col:
        return HandlerResult(success=False, error="Need a start date column")
    tmp = df.copy()
    tmp[start_col] = pd.to_datetime(tmp[start_col], format="mixed", errors="coerce")
    if end_col:
        tmp[end_col] = pd.to_datetime(tmp[end_col], format="mixed", errors="coerce")
    else:
        end_col = "_end"
        tmp[end_col] = tmp[start_col] + pd.Timedelta(days=1)
    if not label_col:
        label_col = "_label"
        tmp[label_col] = tmp.index.astype(str)
    tmp = tmp.dropna(subset=[start_col, end_col])
    fig = px.timeline(tmp, x_start=start_col, x_end=end_col, y=label_col)
    _style(fig, title="Timeline Chart")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()], output_type="query",
                         summary=f"Timeline chart: {start_col} → {end_col}")
