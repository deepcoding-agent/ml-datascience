"""handle_constraint_check handler."""
from __future__ import annotations

import re

import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_constraint_check(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Check uniqueness, value range, regex format, and allowed values constraints."""
    unique_cols = params.get("unique", [])
    ranges = params.get("ranges", {})          # {"col": {"min": 0, "max": 100}}
    patterns = params.get("patterns", {})      # {"col": "^[A-Z]{2}\\d+$"}
    allowed = params.get("allowed_values", {}) # {"col": ["A", "B", "C"]}

    if isinstance(unique_cols, str):
        unique_cols = [s.strip() for s in unique_cols.split(",")]

    violations: list[dict] = []

    # Auto-detect if no constraints provided
    if not unique_cols and not ranges and not patterns and not allowed:
        checks = []
        for col in df.columns:
            is_unique = df[col].nunique() == len(df.dropna(subset=[col]))
            null_pct = round(df[col].isnull().mean() * 100, 1)
            checks.append({
                "column": col,
                "is_unique": is_unique,
                "null_pct": null_pct,
                "dtype": str(df[col].dtype),
                "nunique": int(df[col].nunique()),
                "sample_values": str(df[col].dropna().head(3).tolist()),
            })
        result_df = pd.DataFrame(checks)
        return HandlerResult(
            success=True, result_df=result_df,
            summary=f"Constraint report: {len(checks)} columns analyzed. Use unique/ranges/patterns/allowed_values params to run specific checks.",
        )

    # Uniqueness check
    for col in unique_cols:
        if col not in df.columns:
            continue
        dup_count = int(df[col].duplicated().sum())
        if dup_count > 0:
            violations.append({"column": col, "check": "unique", "expected": "all unique",
                               "actual": f"{dup_count} duplicates", "rows_affected": dup_count})

    # Range check
    for col, bounds in ranges.items():
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if "min" in bounds:
            below = int((df[col] < bounds["min"]).sum())
            if below > 0:
                violations.append({"column": col, "check": "min_range", "expected": f">= {bounds['min']}",
                                   "actual": f"{below} values below", "rows_affected": below})
        if "max" in bounds:
            above = int((df[col] > bounds["max"]).sum())
            if above > 0:
                violations.append({"column": col, "check": "max_range", "expected": f"<= {bounds['max']}",
                                   "actual": f"{above} values above", "rows_affected": above})

    # Pattern check
    for col, pattern in patterns.items():
        if col not in df.columns:
            continue
        series = df[col].dropna().astype(str)
        non_match = int((~series.str.match(pattern)).sum())
        if non_match > 0:
            violations.append({"column": col, "check": "pattern", "expected": pattern,
                               "actual": f"{non_match} non-matching", "rows_affected": non_match})

    # Allowed values check
    for col, vals in allowed.items():
        if col not in df.columns:
            continue
        invalid = df[~df[col].isin(vals) & df[col].notna()]
        if len(invalid) > 0:
            violations.append({"column": col, "check": "allowed_values", "expected": str(vals),
                               "actual": f"{len(invalid)} invalid values", "rows_affected": len(invalid)})

    result_df = pd.DataFrame(violations) if violations else pd.DataFrame({"result": ["All constraints passed"]})
    status = "FAIL" if violations else "PASS"
    return HandlerResult(
        success=True, result_df=result_df,
        summary=f"Constraint check: {status}. {len(violations)} violations found." if violations else "Constraint check: PASS. All checks passed.",
        metadata={"violations": violations, "status": status},
    )
