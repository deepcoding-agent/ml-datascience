"""handle_label_binarize handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_label_binarize(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Multi-label binarization — one binary column per unique value."""
    col = params.get("column")
    if not col or col not in df.columns:
        return HandlerResult(success=False, error=f"Column '{col}' not found")
    result = df.copy()
    try:
        from sklearn.preprocessing import LabelBinarizer
        lb = LabelBinarizer()
        vals = result[col].fillna("__MISSING__")
        binarized = lb.fit_transform(vals)
        classes = lb.classes_
        for i, cls in enumerate(classes):
            result[f"{col}_{cls}"] = binarized[:, i] if binarized.ndim > 1 else binarized.ravel()
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Binarized '{col}' into {len(classes)} binary columns",
        )
    except Exception as e:
        return HandlerResult(success=False, error=f"Label binarize error: {e}")
