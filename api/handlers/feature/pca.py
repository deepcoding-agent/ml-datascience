"""handle_pca handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_pca(df: pd.DataFrame, params: dict) -> HandlerResult:
    n_components = params.get("n", 2)
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        num_cols = df.select_dtypes(include="number").columns.tolist()
        X = df[num_cols].dropna()
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=min(n_components, len(num_cols)))
        components = pca.fit_transform(X_scaled)
        col_names = [f"PC{i+1}" for i in range(components.shape[1])]
        result = pd.DataFrame(components, columns=col_names, index=X.index)
        var_explained = pca.explained_variance_ratio_
        fig = px.scatter(result, x="PC1", y="PC2", opacity=0.7)
        _style(fig, title=f"PCA (PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%})")
        return HandlerResult(success=True, result_df=result, charts_plotly=[fig.to_json()],
                             output_type="generate",
                             summary=f"PCA: {len(num_cols)} features → {n_components} components, {sum(var_explained):.1%} variance")
    except Exception as e:
        return HandlerResult(success=False, error=f"PCA error: {e}")
