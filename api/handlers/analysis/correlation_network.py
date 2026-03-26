"""handle_correlation_network handler."""
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


def handle_correlation_network(df: pd.DataFrame, params: dict) -> HandlerResult:
    threshold = float(params.get("threshold", 0.5))
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return HandlerResult(success=False, error="Need ≥2 numeric columns")
    corr = df[num_cols].corr()
    edges = []
    for i, c1 in enumerate(num_cols):
        for j, c2 in enumerate(num_cols):
            if i < j and abs(corr.loc[c1, c2]) >= threshold:
                edges.append({"source": c1, "target": c2, "correlation": round(float(corr.loc[c1, c2]), 4),
                              "strength": "strong" if abs(corr.loc[c1, c2]) >= 0.7 else "moderate"})
    result_df = pd.DataFrame(edges) if edges else pd.DataFrame(columns=["source", "target", "correlation"])
    return HandlerResult(success=True, result_df=result_df, output_type="query",
                         summary=f"Correlation network: {len(edges)} edges above |r|≥{threshold} among {len(num_cols)} features")
