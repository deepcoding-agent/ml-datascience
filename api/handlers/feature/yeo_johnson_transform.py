"""handle_yeo_johnson_transform handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_yeo_johnson_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Yeo-Johnson transform — handles negative values unlike Box-Cox."""
    col = params.get("column")
    result = df.copy()
    try:
        from sklearn.preprocessing import PowerTransformer
        cols = (
            [col] if col and col in result.columns
            else result.select_dtypes(include="number").columns.tolist()
        )
        X = result[cols].dropna()
        pt = PowerTransformer(method="yeo-johnson", standardize=True)
        transformed = pd.DataFrame(pt.fit_transform(X), index=X.index, columns=cols)
        for c in cols:
            result[f"{c}_yeojohnson"] = np.nan
            result.loc[X.index, f"{c}_yeojohnson"] = transformed[c].round(4)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Yeo-Johnson transformed {len(cols)} columns",
        )
    except Exception as e:
        return HandlerResult(success=False, error=f"Yeo-Johnson error: {e}")
