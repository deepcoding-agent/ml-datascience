"""handle_data_profiling_report handler."""
from __future__ import annotations
import numpy as np
import pandas as pd
from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger
log = get_logger(__name__)

def handle_data_profiling_report(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Comprehensive data profiling report — column-by-column stats."""
    rows = []
    for col in df.columns:
        s = df[col]
        row = {
            "column": col,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "null_count": int(s.isna().sum()),
            "null_pct": round(s.isna().mean() * 100, 1),
            "unique": int(s.nunique()),
            "unique_pct": round(s.nunique() / max(len(s), 1) * 100, 1),
        }
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe()
            row["mean"] = round(float(desc.get("mean", 0)), 4)
            row["std"] = round(float(desc.get("std", 0)), 4)
            row["min"] = round(float(desc.get("min", 0)), 4)
            row["max"] = round(float(desc.get("max", 0)), 4)
            row["median"] = round(float(s.median()), 4)
            row["skewness"] = round(float(s.skew()), 4)
            row["kurtosis"] = round(float(s.kurtosis()), 4)
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            outliers = ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()
            row["outliers"] = int(outliers)
            row["zeros"] = int((s == 0).sum())
        else:
            top = s.value_counts().head(1)
            row["top_value"] = str(top.index[0]) if len(top) > 0 else ""
            row["top_freq"] = int(top.values[0]) if len(top) > 0 else 0
            row["avg_length"] = round(s.astype(str).str.len().mean(), 1)
        rows.append(row)
    result = pd.DataFrame(rows)
    mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    return HandlerResult(success=True, result_df=result, output_type="query",
                         summary=f"Data profile: {len(df):,} rows × {len(df.columns)} cols, {mem_mb:.2f} MB memory")
