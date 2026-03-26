"""handle_hierarchical_clustering handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_hierarchical_clustering(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Agglomerative clustering with dendrogram chart."""
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler
        from scipy.cluster.hierarchy import dendrogram, linkage

        n_clusters = int(params.get("n_clusters", 3))
        method = params.get("method", "ward")
        if method not in ("ward", "complete", "average"):
            method = "ward"

        num_cols = BaseHandler.get_numeric_cols(df)
        if len(num_cols) < 2:
            return HandlerResult(
                success=False,
                error="Need at least 2 numeric columns for hierarchical clustering.",
            )

        cols = num_cols[:10]  # Cap at 10 features
        X = df[cols].dropna()
        if len(X) < n_clusters + 1:
            return HandlerResult(
                success=False,
                error=f"Need at least {n_clusters + 1} non-null rows for {n_clusters} clusters.",
            )

        # Cap rows for dendrogram readability
        max_rows = min(len(X), 500)
        if len(X) > max_rows:
            X = X.sample(n=max_rows, random_state=42)

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit agglomerative clustering
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
        labels = agg.fit_predict(X_scaled)

        result_df = X.copy()
        result_df["_cluster"] = labels

        # Build dendrogram data via scipy linkage
        Z = linkage(X_scaled, method=method)
        dendro = dendrogram(Z, no_plot=True, truncate_mode="lastp", p=min(30, len(X)))

        # Build Plotly dendrogram figure
        fig = go.Figure()

        icoord = dendro["icoord"]
        dcoord = dendro["dcoord"]
        colors = dendro.get("color_list", [])

        # Map scipy dendrogram colors to hex
        color_map = {
            "C0": "#FB8C3C", "C1": "#2EC4B6", "C2": "#457B9D",
            "C3": "#E71D36", "C4": "#FF9F1C", "C5": "#A8DADC",
            "C6": "#1D3557", "C7": "#6B4226", "b": "#457B9D",
            "g": "#2EC4B6", "r": "#E71D36", "c": "#A8DADC",
            "m": "#6B4226", "y": "#FF9F1C", "k": "#1D1D1F",
        }

        for i, (x_seg, y_seg) in enumerate(zip(icoord, dcoord)):
            c = colors[i] if i < len(colors) else "C0"
            hex_c = color_map.get(c, "#FB8C3C")
            fig.add_trace(go.Scatter(
                x=x_seg,
                y=y_seg,
                mode="lines",
                line=dict(color=hex_c, width=1.5),
                showlegend=False,
                hoverinfo="skip",
            ))

        _style(
            fig,
            title=f"Dendrogram (method={method}, n_clusters={n_clusters}, n={len(X)})",
            xaxis_title="Sample index",
            yaxis_title="Distance",
        )

        cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()
        size_str = ", ".join(f"C{k}: {v}" for k, v in cluster_sizes.items())

        return HandlerResult(
            success=True,
            result_df=result_df,
            output_type="generate",
            charts_plotly=[fig.to_json()],
            summary=(
                f"Hierarchical clustering ({method}): {n_clusters} clusters on "
                f"{len(X)} rows, {len(cols)} features. Sizes: {size_str}"
            ),
        )
    except Exception as e:
        log.exception("hierarchical_clustering failed")
        return HandlerResult(success=False, error=f"hierarchical_clustering failed: {e}")
