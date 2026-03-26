"""handle_cluster_tendency handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_cluster_tendency(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Hopkins statistic to check if data is clusterable (< 0.5 = clusterable)."""
    from sklearn.neighbors import NearestNeighbors

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return HandlerResult(success=False, error="Need at least 2 numeric columns")

    X = df[num_cols].dropna()
    if len(X) < 10:
        return HandlerResult(success=False, error="Need at least 10 rows")

    n = min(len(X), 200)
    rng = np.random.RandomState(42)
    sample = X.sample(n, random_state=42).values
    # Uniform random points in feature space
    rand_pts = rng.uniform(X.min().values, X.max().values, size=(n, X.shape[1]))

    nn = NearestNeighbors(n_neighbors=2).fit(X.values)
    u_dist = nn.kneighbors(rand_pts, return_distance=True)[0][:, 1].sum()
    w_dist = nn.kneighbors(sample, return_distance=True)[0][:, 1].sum()
    h = round(float(u_dist / (u_dist + w_dist)), 4)

    label = "Clusterable" if h > 0.5 else "Uniform (not clusterable)"
    result = pd.DataFrame([{"hopkins_stat": h, "interpretation": label, "n_features": len(num_cols)}])
    return HandlerResult(success=True, result_df=result,
                         summary=f"Hopkins statistic: {h} — {label}")
