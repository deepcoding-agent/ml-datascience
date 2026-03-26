"""handle_density_heatmap handler."""
from __future__ import annotations
import pandas as pd
import plotly.express as px
from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger
log = get_logger(__name__)

def handle_density_heatmap(df: pd.DataFrame, params: dict) -> HandlerResult:
    """2D density heatmap between two numeric columns."""
    nums = BaseHandler.get_numeric_cols(df)
    x = params.get("x") or params.get("column") or (nums[0] if len(nums) >= 2 else None)
    y = params.get("y") or (nums[1] if len(nums) >= 2 else None)
    if not x or not y:
        return HandlerResult(success=False, error=f"Need 2 numeric columns. Available: {nums}")
    nbins = int(params.get("bins", 30))
    fig = px.density_heatmap(df, x=x, y=y, nbinsx=nbins, nbinsy=nbins,
                             color_continuous_scale="Oranges")
    _style(fig, title=f"Density Heatmap: {x} vs {y}")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()], output_type="query",
                         summary=f"2D density heatmap of {x} vs {y}")
