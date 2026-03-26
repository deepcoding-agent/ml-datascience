"""handle_feature_interaction handler."""
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


def handle_feature_interaction(df: pd.DataFrame, params: dict) -> HandlerResult:
    target = params.get("column")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not target or target not in num_cols:
        target = num_cols[-1] if num_cols else None
    if target is None or len(num_cols) < 3:
        return HandlerResult(success=False, error="Need target + ≥2 features")
    feats = [c for c in num_cols if c != target][:8]
    rows = []
    y = df[target].dropna()
    for i, f1 in enumerate(feats):
        for f2 in feats[i + 1:]:
            clean = df[[f1, f2, target]].dropna()
            interaction = clean[f1] * clean[f2]
            r_individual = max(abs(clean[f1].corr(clean[target])), abs(clean[f2].corr(clean[target])))
            r_interaction = abs(interaction.corr(clean[target]))
            if r_interaction > r_individual:
                rows.append({"feature_1": f1, "feature_2": f2, "individual_max_r": round(r_individual, 4),
                             "interaction_r": round(r_interaction, 4), "lift": round(r_interaction - r_individual, 4)})
    rows.sort(key=lambda x: x["lift"], reverse=True)
    result_df = pd.DataFrame(rows[:15]) if rows else pd.DataFrame(columns=["feature_1", "feature_2", "interaction_r", "lift"])
    return HandlerResult(success=True, result_df=result_df, output_type="query",
                         summary=f"Found {len(rows)} feature interactions improving correlation with '{target}'")
