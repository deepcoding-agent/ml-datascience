"""handle_cluster_kmeans handler."""
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


def handle_cluster_kmeans(df: pd.DataFrame, params: dict) -> HandlerResult:
    """K-Means clustering with auto k selection (silhouette) + 2D scatter."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return HandlerResult(success=False, error="Need at least 2 numeric columns for clustering")

    cols = num_cols[:10]
    X = df[cols].dropna()
    if len(X) < 10:
        return HandlerResult(success=False, error="Need at least 10 non-null rows for clustering")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    max_k = min(int(params.get("max_k", 8)), len(X) - 1, 10)
    min_k = 2
    best_k, best_score = 2, -1.0
    scores: list[dict] = []
    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
        labels = km.fit_predict(X_scaled)
        s = silhouette_score(X_scaled, labels)
        scores.append({"k": k, "silhouette": round(s, 4)})
        if s > best_score:
            best_k, best_score = k, s

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42, max_iter=300)
    labels = km.fit_predict(X_scaled)
    X = X.copy()
    X["cluster"] = labels

    fig = px.scatter(
        X, x=cols[0], y=cols[1], color="cluster",
        color_continuous_scale="Viridis",
    )
    fig.update_traces(marker_size=5)
    _style(fig, title=f"K-Means Clustering (k={best_k}, silhouette={best_score:.3f})")

    result_df = pd.DataFrame(scores)
    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"K-Means: best k={best_k} (silhouette={best_score:.3f}) on {len(X)} rows, {len(cols)} features",
    )
