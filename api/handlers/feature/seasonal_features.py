"""handle_seasonal_features handler."""
from __future__ import annotations

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_seasonal_features(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Create cyclical sin/cos encoding for periodic features (month, day of week, hour)."""
    column = params.get("column")
    result = df.copy()
    created = []

    dt_cols = result.select_dtypes(include="datetime").columns.tolist()
    if column and column in result.columns:
        try:
            result[column] = pd.to_datetime(result[column], format="mixed", errors="coerce")
            dt_cols = [column]
        except Exception:
            pass

    if dt_cols:
        for col in dt_cols:
            dt = result[col]
            # Month (period = 12)
            result[f"{col}_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
            result[f"{col}_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
            created.extend([f"{col}_month_sin", f"{col}_month_cos"])
            # Day of week (period = 7)
            result[f"{col}_dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
            result[f"{col}_dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
            created.extend([f"{col}_dow_sin", f"{col}_dow_cos"])
            # Day of month (period = 31)
            result[f"{col}_day_sin"] = np.sin(2 * np.pi * dt.dt.day / 31)
            result[f"{col}_day_cos"] = np.cos(2 * np.pi * dt.dt.day / 31)
            created.extend([f"{col}_day_sin", f"{col}_day_cos"])
            # Hour (period = 24) — only if time component exists
            if dt.dt.hour.sum() > 0:
                result[f"{col}_hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
                result[f"{col}_hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
                created.extend([f"{col}_hour_sin", f"{col}_hour_cos"])
    else:
        # Try numeric columns that might represent periods
        num_cols = result.select_dtypes(include="number").columns.tolist()
        periodic_hints = {"month": 12, "day": 31, "hour": 24, "weekday": 7, "dow": 7, "quarter": 4, "minute": 60}
        for col in num_cols:
            col_lower = col.lower()
            for hint, period in periodic_hints.items():
                if hint in col_lower:
                    result[f"{col}_sin"] = np.sin(2 * np.pi * result[col] / period)
                    result[f"{col}_cos"] = np.cos(2 * np.pi * result[col] / period)
                    created.extend([f"{col}_sin", f"{col}_cos"])
                    break

    if not created:
        return HandlerResult(success=False, error="No datetime or periodic numeric columns found for seasonal encoding.")

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Created {len(created)} cyclical sin/cos features: {created[:6]}{'...' if len(created) > 6 else ''}",
    )
