"""handle_regression_quick handler."""
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


def handle_regression_quick(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Quick OLS linear regression + scatter + R-squared + coefficients."""
    from sklearn.linear_model import LinearRegression

    target = params.get("column") or params.get("target")
    feature = params.get("feature")
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if not target or target not in num_cols:
        target = num_cols[-1] if num_cols else None
    if target is None:
        return HandlerResult(success=False, error="No numeric target column found")

    features = [c for c in num_cols if c != target]
    if not features:
        return HandlerResult(success=False, error="Need at least 1 feature column")

    clean = df[features + [target]].dropna()
    if len(clean) < 5:
        return HandlerResult(success=False, error="Need at least 5 non-null rows")

    X = clean[features].values
    y = clean[target].values
    model = LinearRegression().fit(X, y)
    r2 = round(float(model.score(X, y)), 4)
    y_pred = model.predict(X)

    coef_df = pd.DataFrame({"feature": features, "coefficient": [round(float(c), 4) for c in model.coef_]})
    coef_df["abs_coeff"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coeff", ascending=False).drop(columns=["abs_coeff"])
    coef_df.loc[len(coef_df)] = {"feature": "(intercept)", "coefficient": round(float(model.intercept_), 4)}

    fig = px.scatter(x=y, y=y_pred, opacity=0.5, labels={"x": "Actual", "y": "Predicted"})
    fig.update_traces(marker_color="#FB8C3C", marker_size=4)
    fig.add_trace(go.Scatter(x=[float(y.min()), float(y.max())], y=[float(y.min()), float(y.max())],
                             mode="lines", line=dict(dash="dash", color="#E71D36"), name="Perfect fit"))
    _style(fig, title=f"OLS Regression — {target} (R\u00b2={r2})")

    return HandlerResult(
        success=True, result_df=coef_df, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"OLS on {len(features)} features \u2192 R\u00b2={r2}. Top predictor: {coef_df.iloc[0]['feature']} (coeff={coef_df.iloc[0]['coefficient']:.4f})",
    )
