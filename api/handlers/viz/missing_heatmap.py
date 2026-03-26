"""handle_missing_heatmap handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_missing_heatmap(df: pd.DataFrame, params: dict) -> HandlerResult:
    null_matrix = df.isnull().astype(int)
    fig = px.imshow(null_matrix, color_continuous_scale=["#F8F8F8", "#FB8C3C"],
                    aspect="auto")
    _style(fig, title="Missing Values Pattern")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary="Missing values heatmap")
