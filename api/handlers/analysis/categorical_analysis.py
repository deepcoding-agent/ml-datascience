"""handle_categorical_analysis handler."""
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


def handle_categorical_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Deep analysis of all categorical columns: cardinality, mode, entropy."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        return HandlerResult(success=False, error="No categorical columns found")

    rows: list[dict] = []
    for c in cat_cols:
        s = df[c].dropna()
        n_unique = int(s.nunique())
        mode_val = str(s.mode().iloc[0]) if len(s) > 0 else "N/A"
        mode_freq = int(s.value_counts().iloc[0]) if len(s) > 0 else 0
        mode_pct = round(mode_freq / max(len(s), 1) * 100, 1)

        probs = s.value_counts(normalize=True)
        entropy = round(float(-(probs * np.log2(probs.clip(lower=1e-10))).sum()), 3)

        card_type = "Binary" if n_unique == 2 else "Low" if n_unique <= 10 else "Medium" if n_unique <= 50 else "High"
        rows.append({
            "column": c, "unique": n_unique, "cardinality": card_type,
            "mode": mode_val, "mode_pct": mode_pct, "entropy": entropy,
            "null_pct": round(df[c].isnull().sum() / len(df) * 100, 1),
        })

    result_df = pd.DataFrame(rows)

    fig = px.bar(result_df, x="unique", y="column", orientation="h",
                 color="cardinality", text="unique",
                 color_discrete_map={"Binary": "#2EC4B6", "Low": "#FB8C3C", "Medium": "#FF9F1C", "High": "#E71D36"})
    fig.update_traces(textposition="outside")
    _style(fig, title=f"Categorical Column Analysis — {len(cat_cols)} columns")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})

    highest_card = result_df.loc[result_df["unique"].idxmax()]
    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"Analyzed {len(cat_cols)} categorical columns. Highest cardinality: {highest_card['column']} ({highest_card['unique']} unique)",
    )
