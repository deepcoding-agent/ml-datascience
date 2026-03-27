"""handle_datetime_decompose handler."""
from __future__ import annotations

import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_datetime_decompose(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Extract year, month, day, weekday, hour, quarter, is_weekend from datetime columns."""
    column = params.get("column")
    result = df.copy()

    dt_cols = result.select_dtypes(include="datetime").columns.tolist()
    if column and column in result.columns:
        try:
            result[column] = pd.to_datetime(result[column], format="mixed", errors="coerce")
            if column not in dt_cols:
                dt_cols = [column]
        except Exception:
            pass

    if not dt_cols:
        # Try to find columns that can be parsed as datetime
        for c in result.columns:
            if result[c].dtype == "object":
                try:
                    parsed = pd.to_datetime(result[c], format="mixed", errors="coerce")
                    if parsed.notna().sum() > len(result) * 0.5:
                        result[c] = parsed
                        dt_cols.append(c)
                        break
                except Exception:
                    continue

    if not dt_cols:
        return HandlerResult(success=False, error="No datetime columns found. Specify a column or ensure data has datetime values.")

    created = []
    for col in dt_cols:
        dt = result[col]
        result[f"{col}_year"] = dt.dt.year
        result[f"{col}_month"] = dt.dt.month
        result[f"{col}_day"] = dt.dt.day
        result[f"{col}_weekday"] = dt.dt.dayofweek
        result[f"{col}_quarter"] = dt.dt.quarter
        result[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        created.extend([f"{col}_year", f"{col}_month", f"{col}_day",
                        f"{col}_weekday", f"{col}_quarter", f"{col}_is_weekend"])
        # Only add hour if time component exists
        if dt.dt.hour.sum() > 0:
            result[f"{col}_hour"] = dt.dt.hour
            created.append(f"{col}_hour")

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Decomposed {len(dt_cols)} datetime columns into {len(created)} features: {created[:6]}{'...' if len(created) > 6 else ''}",
    )
