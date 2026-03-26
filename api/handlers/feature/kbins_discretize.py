"""handle_kbins_discretize handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_kbins_discretize(df: pd.DataFrame, params: dict) -> HandlerResult:
    """KBins discretizer (uniform/quantile/kmeans) via sklearn."""
    col = params.get("column")
    n_bins = params.get("n", 5)
    strategy = params.get("strategy", "quantile")
    try:
        from sklearn.preprocessing import KBinsDiscretizer
        result = df.copy()
        cols = (
            [col] if col and col in result.columns
            else result.select_dtypes(include="number").columns.tolist()
        )
        X = result[cols].dropna()
        kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
        binned = pd.DataFrame(kbd.fit_transform(X), index=X.index, columns=[f"{c}_kbin" for c in cols])
        for c in binned.columns:
            result[c] = np.nan
            result.loc[X.index, c] = binned[c].astype(int)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"KBins discretized {len(cols)} columns (n={n_bins}, strategy={strategy})",
        )
    except Exception as e:
        return HandlerResult(success=False, error=f"KBins discretize error: {e}")
