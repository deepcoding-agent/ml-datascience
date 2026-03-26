"""handle_bootstrap_ci handler."""
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


def handle_bootstrap_ci(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")
    n_boot = int(params.get("n_bootstrap", 1000))
    ci = float(params.get("confidence", 0.95))
    num_cols = df.select_dtypes(include="number").columns.tolist()
    col = col if col and col in num_cols else (num_cols[0] if num_cols else None)
    if col is None:
        return HandlerResult(success=False, error="No numeric column found")
    s = df[col].dropna().values
    rng = np.random.RandomState(42)
    means = [float(np.mean(rng.choice(s, size=len(s), replace=True))) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    result_df = pd.DataFrame({"metric": ["mean", "ci_lower", "ci_upper", "ci_width", "n_bootstrap"],
                               "value": [round(float(np.mean(s)), 4), round(float(lo), 4), round(float(hi), 4),
                                         round(float(hi - lo), 4), n_boot]})
    fig = px.histogram(x=means, nbins=40)
    fig.update_traces(marker_color="#FB8C3C")
    fig.add_vline(x=lo, line_dash="dash", line_color="#E71D36")
    fig.add_vline(x=hi, line_dash="dash", line_color="#E71D36")
    _style(fig, title=f"Bootstrap {ci*100:.0f}% CI — {col}: [{lo:.4f}, {hi:.4f}]")
    return HandlerResult(success=True, result_df=result_df, output_type="query", charts_plotly=[fig.to_json()],
                         summary=f"Bootstrap {ci*100:.0f}% CI for '{col}' mean: [{lo:.4f}, {hi:.4f}] (n={n_boot})")
