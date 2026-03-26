"""handle_date_diff handler."""
from __future__ import annotations

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_date_diff(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Calculate days/months/years between two date columns."""
    try:
        col_start = params.get("column")
        col_end = params.get("column2")
        unit = params.get("unit", "days")
        new_name = params.get("new_name")

        # Auto-detect date columns if not specified
        dt_cols = BaseHandler.get_datetime_cols(df)

        # Also check object columns that look like dates
        if len(dt_cols) < 2:
            for c in df.select_dtypes(include="object").columns:
                try:
                    pd.to_datetime(df[c].dropna().head(20), format="mixed")
                    dt_cols.append(c)
                except Exception:
                    pass

        if not col_start:
            if len(dt_cols) >= 1:
                col_start = dt_cols[0]
            else:
                return HandlerResult(
                    success=False,
                    error="No start date column specified and no date columns detected.",
                )

        if not col_end:
            if len(dt_cols) >= 2:
                col_end = [c for c in dt_cols if c != col_start][0]
            else:
                return HandlerResult(
                    success=False,
                    error="No end date column specified and fewer than 2 date columns detected.",
                )

        # Validate columns exist
        col_start, err = BaseHandler.require_column(df, col_start, col_start or "")
        if err:
            return err
        col_end, err = BaseHandler.require_column(df, col_end, col_end or "")
        if err:
            return err

        result = df.copy()

        # Parse to datetime
        start_dt = pd.to_datetime(result[col_start], format="mixed", errors="coerce")
        end_dt = pd.to_datetime(result[col_end], format="mixed", errors="coerce")

        # Default new column name
        if not new_name:
            new_name = f"_diff_{unit}"

        if unit == "days":
            result[new_name] = (end_dt - start_dt).dt.days
        elif unit == "months":
            # Approximate months via days / 30.44
            result[new_name] = np.round((end_dt - start_dt).dt.days / 30.44, 2)
        elif unit == "years":
            # Approximate years via days / 365.25
            result[new_name] = np.round((end_dt - start_dt).dt.days / 365.25, 2)
        else:
            return HandlerResult(
                success=False,
                error=f"Unknown unit '{unit}'. Use 'days', 'months', or 'years'.",
            )

        valid_count = result[new_name].notna().sum()
        return HandlerResult(
            success=True,
            result_df=result,
            output_type="generate",
            summary=(
                f"Calculated {unit} difference between '{col_start}' and '{col_end}' "
                f"as '{new_name}' ({valid_count}/{len(df)} valid rows)"
            ),
        )
    except Exception as e:
        log.exception("date_diff failed")
        return HandlerResult(success=False, error=f"date_diff failed: {e}")
