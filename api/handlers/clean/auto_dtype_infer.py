"""handle_auto_dtype_infer handler."""
from __future__ import annotations

import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_auto_dtype_infer(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Smart data type inference and auto-conversion for all columns."""
    result = df.copy()
    changes: list[dict] = []

    for col in result.columns:
        original_dtype = str(result[col].dtype)

        # Skip already well-typed columns
        if result[col].dtype in ("datetime64[ns]", "bool"):
            continue

        series = result[col].dropna()
        if series.empty:
            continue

        # Try numeric conversion
        if result[col].dtype == "object":
            numeric = pd.to_numeric(result[col], errors="coerce")
            non_null_orig = result[col].notna().sum()
            non_null_numeric = numeric.notna().sum()
            if non_null_numeric > 0 and non_null_numeric >= non_null_orig * 0.8:
                # Check if integer or float
                if (numeric.dropna() == numeric.dropna().astype(int)).all():
                    result[col] = numeric.astype("Int64")
                else:
                    result[col] = numeric
                changes.append({"column": col, "from": original_dtype, "to": str(result[col].dtype)})
                continue

        # Try boolean conversion
        if result[col].dtype == "object":
            lower = series.astype(str).str.lower().unique()
            bool_vals = {"true", "false", "yes", "no", "1", "0", "t", "f", "y", "n"}
            if set(lower).issubset(bool_vals) and len(lower) <= 3:
                mapping = {"true": True, "false": False, "yes": True, "no": False,
                           "1": True, "0": False, "t": True, "f": False, "y": True, "n": False}
                result[col] = result[col].astype(str).str.lower().map(mapping)
                changes.append({"column": col, "from": original_dtype, "to": "bool"})
                continue

        # Try datetime conversion
        if result[col].dtype == "object":
            try:
                dt = pd.to_datetime(result[col], format="mixed", errors="coerce")
                if dt.notna().sum() >= series.shape[0] * 0.7:
                    result[col] = dt
                    changes.append({"column": col, "from": original_dtype, "to": "datetime64[ns]"})
                    continue
            except Exception:
                pass

        # Try category for low-cardinality strings
        if result[col].dtype == "object" and result[col].nunique() < len(result) * 0.05:
            result[col] = result[col].astype("category")
            changes.append({"column": col, "from": original_dtype, "to": "category"})

    if not changes:
        return HandlerResult(
            success=True, result_df=result,
            summary="Auto dtype inference: no changes needed — all columns already have appropriate types.",
        )

    change_df = pd.DataFrame(changes)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Auto dtype inference: converted {len(changes)} columns: {', '.join(c['column'] + ' (' + c['from'] + ' → ' + c['to'] + ')' for c in changes[:5])}{'...' if len(changes) > 5 else ''}",
        metadata={"changes": changes},
    )
