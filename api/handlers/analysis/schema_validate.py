"""handle_schema_validate handler."""
from __future__ import annotations

import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_schema_validate(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Validate column types, null constraints, and value ranges against expected schema."""
    expected_types = params.get("types", {})  # {"col": "int64"}
    not_null = params.get("not_null", [])     # ["col1", "col2"]
    ranges = params.get("ranges", {})         # {"col": {"min": 0, "max": 100}}

    if isinstance(not_null, str):
        not_null = [s.strip() for s in not_null.split(",")]

    violations: list[dict] = []

    # Auto-detect schema if no params provided
    if not expected_types and not not_null and not ranges:
        # Generate schema report from current data
        rows = []
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            null_pct = round(null_count / len(df) * 100, 1) if len(df) > 0 else 0
            dtype = str(df[col].dtype)
            nunique = int(df[col].nunique())
            row = {"column": col, "dtype": dtype, "nulls": null_count, "null_pct": null_pct,
                   "unique": nunique, "nullable": null_count > 0}
            if pd.api.types.is_numeric_dtype(df[col]):
                row["min"] = round(float(df[col].min()), 4) if df[col].notna().any() else None
                row["max"] = round(float(df[col].max()), 4) if df[col].notna().any() else None
            rows.append(row)
        result_df = pd.DataFrame(rows)
        return HandlerResult(
            success=True, result_df=result_df,
            summary=f"Schema report: {len(df.columns)} columns, {len(df)} rows. Use types/not_null/ranges params to validate against expected schema.",
        )

    # Validate types
    for col, expected in expected_types.items():
        if col not in df.columns:
            violations.append({"column": col, "check": "exists", "expected": "present", "actual": "missing"})
            continue
        actual = str(df[col].dtype)
        if expected not in actual:
            violations.append({"column": col, "check": "dtype", "expected": expected, "actual": actual})

    # Validate not-null
    for col in not_null:
        if col not in df.columns:
            continue
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            violations.append({"column": col, "check": "not_null", "expected": "0 nulls", "actual": f"{null_count} nulls"})

    # Validate ranges
    for col, bounds in ranges.items():
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        col_min = float(df[col].min()) if df[col].notna().any() else None
        col_max = float(df[col].max()) if df[col].notna().any() else None
        if "min" in bounds and col_min is not None and col_min < bounds["min"]:
            violations.append({"column": col, "check": "min_range", "expected": f">= {bounds['min']}", "actual": str(col_min)})
        if "max" in bounds and col_max is not None and col_max > bounds["max"]:
            violations.append({"column": col, "check": "max_range", "expected": f"<= {bounds['max']}", "actual": str(col_max)})

    result_df = pd.DataFrame(violations) if violations else pd.DataFrame({"result": ["All schema checks passed"]})
    status = "FAIL" if violations else "PASS"
    return HandlerResult(
        success=True, result_df=result_df,
        summary=f"Schema validation: {status}. {len(violations)} violations found." if violations else "Schema validation: PASS. All checks passed.",
        metadata={"violations": violations, "status": status},
    )
