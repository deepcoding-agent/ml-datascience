"""handle_effect_size handler."""
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


def handle_effect_size(df: pd.DataFrame, params: dict) -> HandlerResult:
    group_col = params.get("column")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    group_col = group_col if group_col and group_col in cat_cols else (cat_cols[0] if cat_cols else None)
    if not group_col or not num_cols:
        return HandlerResult(success=False, error="Need categorical group column + numeric features")
    groups = df[group_col].dropna().unique()[:2]
    if len(groups) < 2:
        return HandlerResult(success=False, error=f"Need ≥2 groups in '{group_col}'")
    g1, g2 = df[df[group_col] == groups[0]], df[df[group_col] == groups[1]]
    rows = []
    for c in num_cols[:10]:
        m1, m2 = float(g1[c].mean()), float(g2[c].mean())
        s_pooled = np.sqrt((float(g1[c].std()) ** 2 + float(g2[c].std()) ** 2) / 2)
        d = (m1 - m2) / max(s_pooled, 1e-10)
        size = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small"
        rows.append({"feature": c, "mean_1": round(m1, 4), "mean_2": round(m2, 4),
                      "cohens_d": round(d, 4), "effect_size": size})
    result_df = pd.DataFrame(rows)
    return HandlerResult(success=True, result_df=result_df, output_type="query",
                         summary=f"Effect sizes ({groups[0]} vs {groups[1]}): {sum(1 for r in rows if r['effect_size']!='small')}/{len(rows)} features have medium/large effect")
