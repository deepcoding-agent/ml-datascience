"""handle_svd_features handler."""
from __future__ import annotations
import numpy as np
import pandas as pd
from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger
log = get_logger(__name__)

def handle_svd_features(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Truncated SVD dimensionality reduction (works with sparse data too)."""
    from sklearn.decomposition import TruncatedSVD
    n_components = int(params.get("n_components", params.get("value", 3)))
    nums = BaseHandler.get_numeric_cols(df)
    if len(nums) < 2:
        return HandlerResult(success=False, error="Need at least 2 numeric columns")
    n_components = min(n_components, len(nums), len(df) - 1)
    X = df[nums].fillna(0).values
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    components = svd.fit_transform(X)
    result = df.copy()
    for i in range(n_components):
        result[f"svd_{i+1}"] = components[:, i]
    explained = svd.explained_variance_ratio_.sum() * 100
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"SVD: {len(nums)} features → {n_components} components ({explained:.1f}% variance explained)")
