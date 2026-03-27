"""handle_frequency_encode_transform handler."""
from __future__ import annotations

import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_frequency_encode_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Frequency encoding — replace each category with its count or proportion."""
    column = params.get("column")
    normalize = params.get("normalize", False)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if column and column in cat_cols:
        cat_cols = [column]
    elif column and column in df.columns:
        cat_cols = [column]
    elif not cat_cols:
        return HandlerResult(success=False, error="No categorical columns found.")

    result = df.copy()
    encoded = []

    for col in cat_cols:
        vc = result[col].value_counts(normalize=bool(normalize))
        mapping = vc.to_dict()
        suffix = "freq_pct" if normalize else "freq"
        new_col = f"{col}_{suffix}"
        result[new_col] = result[col].map(mapping).fillna(0)
        encoded.append(new_col)

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Frequency-encoded {len(encoded)} columns ({'normalized' if normalize else 'counts'}): {encoded}",
    )
