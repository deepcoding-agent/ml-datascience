"""handle_dbscan_clustering handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_dbscan_clustering(df: pd.DataFrame, params: dict) -> HandlerResult:
    """DBSCAN density-based clustering on numeric columns with PCA 2D scatter."""
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler

        eps = float(params.get("eps", 0.5))
        min_samples = int(params.get("min_samples", 5))
        target_col = params.get("column")

        # Select numeric columns
        num_cols = BaseHandler.get_numeric_cols(df)
        if target_col and target_col in df.columns:
            num_cols = [target_col] if target_col in num_cols else num_cols
            if target_col not in num_cols:
                return HandlerResult(
                    success=False,
                    error=f"Column '{target_col}' is not numeric.",
                )
        if len(num_cols) < 2:
            return HandlerResult(
                success=False,
                error="Need at least 2 numeric columns for DBSCAN clustering.",
            )

        cols = num_cols[:10]  # Cap at 10 features
        X = df[cols].dropna()
        if len(X) < max(min_samples + 1, 10):
            return HandlerResult(
                success=False,
                error=f"Need at least {max(min_samples + 1, 10)} non-null rows for DBSCAN.",
            )

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)

        n_clusters = len(set(labels) - {-1})
        n_noise = int((labels == -1).sum())

        # Build result DataFrame with _cluster column
        result_df = X.copy()
        result_df["_cluster"] = labels

        # Generate 2D scatter chart
        if len(cols) > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_scaled)
            plot_df = pd.DataFrame({
                "PC1": coords[:, 0],
                "PC2": coords[:, 1],
                "cluster": labels.astype(str),
            })
            x_col, y_col = "PC1", "PC2"
            explained = pca.explained_variance_ratio_
            title_suffix = f" (PCA: {explained[0]:.1%}+{explained[1]:.1%} variance)"
        else:
            plot_df = pd.DataFrame({
                cols[0]: X[cols[0]].values,
                cols[1]: X[cols[1]].values,
                "cluster": labels.astype(str),
            })
            x_col, y_col = cols[0], cols[1]
            title_suffix = ""

        # Mark noise as "Noise" for clarity
        plot_df["cluster"] = plot_df["cluster"].replace({"-1": "Noise"})

        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color="cluster",
            opacity=0.7,
        )
        fig.update_traces(marker_size=5)
        _style(
            fig,
            title=f"DBSCAN Clustering (eps={eps}, min_samples={min_samples}, "
                  f"{n_clusters} clusters, {n_noise} noise){title_suffix}",
        )

        return HandlerResult(
            success=True,
            result_df=result_df,
            output_type="generate",
            charts_plotly=[fig.to_json()],
            summary=(
                f"DBSCAN found {n_clusters} clusters and {n_noise} noise points "
                f"on {len(X)} rows, {len(cols)} features (eps={eps}, min_samples={min_samples})"
            ),
        )
    except Exception as e:
        log.exception("dbscan_clustering failed")
        return HandlerResult(success=False, error=f"dbscan_clustering failed: {e}")
