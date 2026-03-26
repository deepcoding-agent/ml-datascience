"""handle_cluster_profile handler."""
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


def handle_cluster_profile(df: pd.DataFrame, params: dict) -> HandlerResult:
    n_clusters = int(params.get("n", 3))
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return HandlerResult(success=False, error="Need ≥2 numeric columns")
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        X = df[num_cols].dropna()
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        X_copy = X.copy()
        X_copy["_cluster"] = labels
        profile = X_copy.groupby("_cluster")[num_cols].mean().round(3).reset_index()
        counts = X_copy["_cluster"].value_counts().sort_index()
        profile["_count"] = counts.values
        fig = px.bar(profile.melt(id_vars=["_cluster", "_count"], value_vars=num_cols[:6]),
                     x="variable", y="value", color="_cluster", barmode="group", text="value")
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        _style(fig, title=f"Cluster Profiles — {n_clusters} clusters")
        return HandlerResult(success=True, result_df=profile, output_type="query", charts_plotly=[fig.to_json()],
                             summary=f"Profiled {n_clusters} clusters across {len(num_cols)} features. Sizes: {dict(counts)}")
    except Exception as e:
        return HandlerResult(success=False, error=f"Cluster profile error: {e}")
