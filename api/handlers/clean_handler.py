"""Cleaning handler — drop/fill nulls, remove duplicates, fix dtypes, rename, drop columns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


class CleanHandler(BaseHandler):

    @staticmethod
    def handle_drop_nulls(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        result = df.copy()
        before = len(result)
        if col and col in result.columns:
            result = result.dropna(subset=[col])
            summary = f"Dropped {before - len(result):,} rows with null in '{col}'"
        else:
            threshold = params.get("threshold", 0.5)
            # Drop columns with >threshold nulls, then drop rows with any remaining nulls
            null_pct = result.isnull().mean()
            cols_to_drop = null_pct[null_pct > threshold].index.tolist()
            if cols_to_drop:
                result = result.drop(columns=cols_to_drop)
            result = result.dropna()
            summary = f"Dropped {len(cols_to_drop)} columns (>{threshold*100:.0f}% null), then {before - len(result):,} rows with remaining nulls"
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=summary, metadata={"rows_before": before, "rows_after": len(result)})

    @staticmethod
    def handle_fill_nulls(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        strategy = params.get("strategy", "auto")
        result = df.copy()
        filled: dict[str, str] = {}

        cols_to_fill = [col] if col and col in result.columns else result.columns.tolist()

        for c in cols_to_fill:
            if result[c].isnull().sum() == 0:
                continue
            if pd.api.types.is_numeric_dtype(result[c]):
                if strategy == "auto":
                    s = "median" if abs(result[c].skew()) > 1 else "mean"
                else:
                    s = strategy
                if s == "median":
                    result[c] = result[c].fillna(result[c].median())
                elif s == "mean":
                    result[c] = result[c].fillna(result[c].mean())
                elif s == "zero":
                    result[c] = result[c].fillna(0)
                else:
                    result[c] = result[c].fillna(result[c].median())
                filled[c] = s
            else:
                mode = result[c].mode()
                result[c] = result[c].fillna(mode.iloc[0] if not mode.empty else "Unknown")
                filled[c] = "mode"

        remaining = int(result.isnull().sum().sum())
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Filled nulls in {len(filled)} columns. Remaining nulls: {remaining}",
                             metadata={"strategies": filled})

    @staticmethod
    def handle_remove_duplicates(df: pd.DataFrame, params: dict) -> HandlerResult:
        before = len(df)
        result = df.drop_duplicates()
        removed = before - len(result)
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Removed {removed:,} duplicate rows ({before:,} → {len(result):,})")

    @staticmethod
    def handle_fix_dtypes(df: pd.DataFrame, params: dict) -> HandlerResult:
        result = df.copy()
        converted: dict[str, str] = {}
        for col in result.select_dtypes(include="object").columns:
            sample = result[col].dropna().head(100)
            if sample.empty:
                continue
            # Try numeric
            coerced = pd.to_numeric(sample, errors="coerce")
            if coerced.notna().mean() >= 0.8:
                result[col] = pd.to_numeric(result[col], errors="coerce")
                converted[col] = "numeric"
                continue
            # Try datetime
            try:
                coerced_dt = pd.to_datetime(sample, errors="coerce", format="mixed")
                if coerced_dt.notna().mean() >= 0.8:
                    result[col] = pd.to_datetime(result[col], errors="coerce", format="mixed")
                    converted[col] = "datetime"
            except Exception:
                pass
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Converted {len(converted)} columns: {converted}",
                             metadata={"converted": converted})

    @staticmethod
    def handle_rename_column(df: pd.DataFrame, params: dict) -> HandlerResult:
        old_name = params.get("column")
        new_name = params.get("new_name") or params.get("value")
        if not old_name or not new_name:
            return HandlerResult(success=False, error="Need both old and new column names")
        if old_name not in df.columns:
            return HandlerResult(success=False, error=f"Column '{old_name}' not found")
        result = df.rename(columns={old_name: str(new_name)})
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Renamed '{old_name}' → '{new_name}'")

    @staticmethod
    def handle_drop_column(df: pd.DataFrame, params: dict) -> HandlerResult:
        cols = params.get("columns") or ([params["column"]] if params.get("column") else [])
        missing = [c for c in cols if c not in df.columns]
        if missing:
            return HandlerResult(success=False, error=f"Columns not found: {missing}")
        result = df.drop(columns=cols)
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Dropped {len(cols)} column(s): {cols}")

    @staticmethod
    def handle_strip_whitespace(df: pd.DataFrame, params: dict) -> HandlerResult:
        result = df.copy()
        str_cols = result.select_dtypes(include="object").columns
        for col in str_cols:
            result[col] = result[col].str.strip()
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Stripped whitespace from {len(str_cols)} string columns")

    @staticmethod
    def handle_replace_values(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        old_val = params.get("old_value", "?")
        new_val = params.get("new_value", np.nan)
        result = df.copy()
        if col and col in result.columns:
            result[col] = result[col].replace(old_val, new_val)
            summary = f"Replaced '{old_val}' with '{new_val}' in '{col}'"
        else:
            result = result.replace(old_val, new_val)
            summary = f"Replaced '{old_val}' with '{new_val}' in all columns"
        return HandlerResult(success=True, result_df=result, output_type="generate", summary=summary)
