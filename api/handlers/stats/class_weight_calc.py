"""handle_class_weight_calc handler."""
from __future__ import annotations

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_class_weight_calc(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Calculate balanced class weights for model training (sklearn-compatible)."""
    from sklearn.utils.class_weight import compute_class_weight

    target_col = params.get("column") or params.get("target")

    if not target_col or target_col not in df.columns:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_low = [c for c in df.select_dtypes(include="number").columns if df[c].nunique() <= 10]
        target_col = (num_low + cat_cols + [df.columns[-1]])[0]

    y = df[target_col].dropna()
    classes = np.array(sorted(y.unique()))
    weights = compute_class_weight("balanced", classes=classes, y=y)

    result_df = pd.DataFrame({
        "class": classes.astype(str),
        "count": [int((y == c).sum()) for c in classes],
        "percentage": [round(float((y == c).mean()) * 100, 2) for c in classes],
        "weight": [round(float(w), 4) for w in weights],
    })

    weight_dict = {str(c): round(float(w), 4) for c, w in zip(classes, weights)}
    return HandlerResult(
        success=True, result_df=result_df,
        summary=f"Class weights for '{target_col}' ({len(classes)} classes): {weight_dict}. Use these in model training via class_weight parameter.",
        metadata={"class_weights": weight_dict},
    )
