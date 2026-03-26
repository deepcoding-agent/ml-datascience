"""handle_categorical_target_crosstab handler."""
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


def handle_categorical_target_crosstab(df: pd.DataFrame, params: dict) -> HandlerResult:
    target = params.get("column")
    feature = params.get("feature_column")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not target or target not in cat_cols:
        target = cat_cols[0] if cat_cols else None
    if not feature or feature not in cat_cols:
        feature = cat_cols[1] if len(cat_cols) > 1 else None
    if not target or not feature:
        return HandlerResult(success=False, error="Need 2 categorical columns")
    ct = pd.crosstab(df[feature], df[target], margins=True, margins_name="Total")
    ct_pct = pd.crosstab(df[feature], df[target], normalize="index").round(4) * 100
    fig = px.imshow(ct_pct.values[:-1] if "Total" in ct_pct.index else ct_pct.values,
                    x=ct_pct.columns.tolist(), y=ct_pct.index.tolist(),
                    text_auto=".1f", color_continuous_scale="YlOrRd")
    _style(fig, title=f"Crosstab: {feature} × {target} (%)")
    return HandlerResult(success=True, result_df=ct.reset_index(), output_type="query", charts_plotly=[fig.to_json()],
                         summary=f"Crosstab: {feature} ({df[feature].nunique()} levels) × {target} ({df[target].nunique()} levels)")
