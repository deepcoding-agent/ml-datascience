"""handle_pca_2d handler."""
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


def handle_pca_2d(df: pd.DataFrame, params: dict) -> HandlerResult:
    """PCA 2D projection with explained variance + scatter plot."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return HandlerResult(success=False, error="Need at least 2 numeric columns for PCA")

    cols = num_cols[:20]
    X = df[cols].dropna()
    if len(X) < 5:
        return HandlerResult(success=False, error="Need at least 5 non-null rows")

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1]})
    var = pca.explained_variance_ratio_

    fig = px.scatter(pca_df, x="PC1", y="PC2", opacity=0.5)
    fig.update_traces(marker_color="#FB8C3C", marker_size=4)
    _style(fig, title=f"PCA 2D — Var explained: PC1={var[0]:.1%}, PC2={var[1]:.1%} (total={sum(var):.1%})")

    result_df = pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": [round(float(v), 4) for v in var],
        "cumulative": [round(float(var[0]), 4), round(float(sum(var)), 4)],
    })
    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"PCA 2D on {len(cols)} features: {sum(var):.1%} variance explained",
    )
