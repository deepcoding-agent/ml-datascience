"""handle_describe handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_describe(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")

    # Single column describe
    if col and col in df.columns:
        desc = df[col].describe().round(4)
        result = desc.to_frame().reset_index()
        result.columns = ["stat", "value"]
        return HandlerResult(success=True, result_df=result, summary=f"Descriptive statistics for '{col}'")

    # Full dataset describe — build a rich summary table
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    rows = []

    for c in df.columns:
        row: dict = {
            "column": c,
            "dtype": str(df[c].dtype),
            "non_null": int(df[c].notna().sum()),
            "null_count": int(df[c].isna().sum()),
            "null_pct": round(df[c].isna().mean() * 100, 1),
            "unique": int(df[c].nunique()),
        }
        if c in num_cols:
            row["mean"] = round(float(df[c].mean()), 2)
            row["std"] = round(float(df[c].std()), 2)
            row["min"] = round(float(df[c].min()), 2)
            row["25%"] = round(float(df[c].quantile(0.25)), 2)
            row["50%"] = round(float(df[c].quantile(0.50)), 2)
            row["75%"] = round(float(df[c].quantile(0.75)), 2)
            row["max"] = round(float(df[c].max()), 2)
        else:
            top = df[c].mode()
            row["top"] = str(top.iloc[0]) if len(top) > 0 else ""
            row["freq"] = int(df[c].value_counts().iloc[0]) if df[c].notna().any() else 0

        rows.append(row)

    result = pd.DataFrame(rows).fillna("")

    # Build summary text
    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    null_total = int(df.isna().sum().sum())
    dup_count = int(df.duplicated().sum())
    parts = [
        f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns ({mem_mb:.2f} MB)",
        f"Numeric: {len(num_cols)} cols | Categorical: {len(cat_cols)} cols",
    ]
    if null_total > 0:
        null_cols_count = int((df.isna().sum() > 0).sum())
        parts.append(f"Missing: {null_total:,} values across {null_cols_count} columns")
    else:
        parts.append("Missing: none")
    if dup_count > 0:
        parts.append(f"Duplicates: {dup_count:,} rows")
    summary = " | ".join(parts)

    return HandlerResult(success=True, result_df=result, summary=summary)
