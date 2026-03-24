"""Cleaning handler — drop/fill nulls, remove duplicates, fix dtypes, rename, drop, outliers, etc."""
from __future__ import annotations

import re

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

    @staticmethod
    def handle_lowercase_columns(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Normalize column names to lowercase snake_case."""
        result = df.copy()
        mapping: dict[str, str] = {}
        new_cols = []
        for c in result.columns:
            # CamelCase → snake_case, strip special chars, collapse underscores
            new = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(c))
            new = re.sub(r"[^a-zA-Z0-9_]", "_", new).lower().strip("_")
            new = re.sub(r"_+", "_", new)
            if new != c:
                mapping[c] = new
            new_cols.append(new)
        result.columns = new_cols
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Renamed {len(mapping)} columns to snake_case" if mapping else "All columns already snake_case",
            metadata={"renamed": mapping},
        )

    @staticmethod
    def handle_drop_constant(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Drop columns where all values are the same (zero information)."""
        result = df.copy()
        nunique = result.nunique(dropna=False)
        constant_cols = nunique[nunique <= 1].index.tolist()
        if constant_cols:
            result = result.drop(columns=constant_cols)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Dropped {len(constant_cols)} constant column(s): {constant_cols}" if constant_cols else "No constant columns found",
        )

    @staticmethod
    def handle_clip_outliers(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Clip outliers using IQR or z-score method."""
        method = params.get("method", "iqr")
        col = params.get("column")
        result = df.copy()
        cols = [col] if col and col in result.columns else result.select_dtypes(include="number").columns.tolist()
        clipped_info: dict[str, int] = {}

        for c in cols:
            before_outliers = 0
            if method == "zscore":
                mean, std = result[c].mean(), result[c].std()
                if std == 0:
                    continue
                z = (result[c] - mean) / std
                mask = z.abs() > 3
                before_outliers = int(mask.sum())
                result.loc[mask & (z > 0), c] = mean + 3 * std
                result.loc[mask & (z < 0), c] = mean - 3 * std
            else:  # iqr
                q1 = result[c].quantile(0.25)
                q3 = result[c].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                mask = (result[c] < lower) | (result[c] > upper)
                before_outliers = int(mask.sum())
                result[c] = result[c].clip(lower=lower, upper=upper)
            if before_outliers > 0:
                clipped_info[c] = before_outliers

        total = sum(clipped_info.values())
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Clipped {total:,} outliers ({method.upper()}) across {len(clipped_info)} columns",
            metadata={"clipped": clipped_info},
        )

    @staticmethod
    def handle_change_dtype(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Cast a column to a specific dtype (int, float, str, bool, datetime, category)."""
        col = params.get("column")
        dtype = params.get("dtype", "str")
        if not col or col not in df.columns:
            return HandlerResult(success=False, error=f"Column '{col}' not found")
        result = df.copy()
        try:
            if dtype in ("datetime", "date"):
                result[col] = pd.to_datetime(result[col], errors="coerce", format="mixed")
            elif dtype == "category":
                result[col] = result[col].astype("category")
            elif dtype == "bool":
                result[col] = result[col].astype(bool)
            elif dtype in ("int", "integer"):
                result[col] = pd.to_numeric(result[col], errors="coerce").astype("Int64")
            elif dtype in ("float", "numeric"):
                result[col] = pd.to_numeric(result[col], errors="coerce")
            else:
                result[col] = result[col].astype(str)
            return HandlerResult(success=True, result_df=result, output_type="generate",
                                 summary=f"Changed '{col}' dtype to {dtype}")
        except Exception as e:
            return HandlerResult(success=False, error=f"Cannot convert '{col}' to {dtype}: {e}")

    @staticmethod
    def handle_fill_interpolate(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Fill nulls via interpolation — linear, ffill (forward), or bfill (backward)."""
        method = params.get("method", "linear")
        col = params.get("column")
        result = df.copy()
        before_nulls = int(result.isna().sum().sum())

        if method == "ffill":
            if col and col in result.columns:
                result[col] = result[col].ffill()
            else:
                result = result.ffill()
        elif method == "bfill":
            if col and col in result.columns:
                result[col] = result[col].bfill()
            else:
                result = result.bfill()
        else:  # linear
            num_cols = [col] if col and col in result.columns else result.select_dtypes(include="number").columns.tolist()
            for c in num_cols:
                result[c] = result[c].interpolate(method="linear")

        after_nulls = int(result.isna().sum().sum())
        filled = before_nulls - after_nulls
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Interpolated ({method}): filled {filled:,} nulls ({before_nulls:,} → {after_nulls:,})",
        )

    @staticmethod
    def handle_remove_outliers(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Remove rows containing outlier values (IQR or z-score)."""
        method = params.get("method", "iqr")
        col = params.get("column")
        result = df.copy()
        cols = [col] if col and col in result.columns else result.select_dtypes(include="number").columns.tolist()
        mask = pd.Series(True, index=result.index)

        for c in cols:
            if method == "zscore":
                mean, std = result[c].mean(), result[c].std()
                if std == 0:
                    continue
                z = ((result[c] - mean) / std).abs()
                mask &= z <= 3
            else:  # iqr
                q1 = result[c].quantile(0.25)
                q3 = result[c].quantile(0.75)
                iqr = q3 - q1
                mask &= (result[c] >= q1 - 1.5 * iqr) & (result[c] <= q3 + 1.5 * iqr)

        before = len(result)
        result = result[mask].reset_index(drop=True)
        removed = before - len(result)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Removed {removed:,} outlier rows ({method.upper()}): {before:,} → {len(result):,}",
        )

    @staticmethod
    def handle_lowercase_values(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Lowercase all string values in specified or all string columns."""
        col = params.get("column")
        result = df.copy()
        if col and col in result.columns:
            cols = [col]
        else:
            cols = result.select_dtypes(include="object").columns.tolist()
        for c in cols:
            result[c] = result[c].str.lower()
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Lowercased values in {len(cols)} column(s): {cols}",
        )

    @staticmethod
    def handle_map_values(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Map/recode values in a column using a mapping dict.

        params: column, mapping (dict e.g. {"M": "Male", "F": "Female"})
        """
        col = params.get("column")
        mapping = params.get("mapping", {})
        if not col or col not in df.columns:
            return HandlerResult(success=False, error=f"Column '{col}' not found")
        if not mapping:
            return HandlerResult(success=False, error="No mapping provided")
        result = df.copy()
        result[col] = result[col].replace(mapping)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Mapped {len(mapping)} values in '{col}': {mapping}",
        )

    @staticmethod
    def handle_reset_index(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Reset the DataFrame index to 0-based sequential."""
        result = df.reset_index(drop=True)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Reset index (0 to {len(result)-1})",
        )

    @staticmethod
    def handle_fill_with_value(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Fill nulls with a specific constant value (e.g. -1, 0, 'Unknown', 'N/A')."""
        col = params.get("column")
        value = params.get("value", 0)
        result = df.copy()
        before_nulls = int(result.isna().sum().sum())

        if col and col in result.columns:
            result[col] = result[col].fillna(value)
            summary_target = f"in '{col}'"
        else:
            result = result.fillna(value)
            summary_target = "in all columns"

        after_nulls = int(result.isna().sum().sum())
        filled = before_nulls - after_nulls
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Filled {filled:,} nulls with {repr(value)} {summary_target}",
        )

    @staticmethod
    def handle_deduplicate_by(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Remove duplicates based on specific column(s), keeping first or last."""
        cols = params.get("columns", [])
        col = params.get("column")
        keep = params.get("keep", "first")

        subset = cols if cols else ([col] if col and col in df.columns else None)
        if subset:
            subset = [c for c in subset if c in df.columns]
            if not subset:
                return HandlerResult(success=False, error="No valid columns specified for deduplication")

        before = len(df)
        result = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
        removed = before - len(result)
        col_desc = f" by {subset}" if subset else ""
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Removed {removed:,} duplicates{col_desc} (keep={keep}): {before:,} → {len(result):,}",
        )

    @staticmethod
    def handle_drop_id_columns(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Auto-detect and drop ID-like columns (high unique ratio, sequential, generic names)."""
        result = df.copy()
        id_cols = []

        for c in result.columns:
            col_lower = c.lower().strip("_")
            n_unique = result[c].nunique()
            unique_ratio = n_unique / len(result) if len(result) > 0 else 0

            # Name-based: short generic ID names
            is_id_name = (
                col_lower in ("id", "index", "idx", "key", "row", "num", "number", "seq", "serial")
                or (col_lower.endswith("id") and len(col_lower) <= 6)
                or col_lower == "unnamed: 0"
            )
            # Value-based: nearly all unique + numeric + sequential
            is_sequential = False
            if pd.api.types.is_numeric_dtype(result[c]) and unique_ratio > 0.95:
                sorted_vals = result[c].dropna().sort_values()
                if len(sorted_vals) > 1:
                    diffs = sorted_vals.diff().dropna()
                    is_sequential = (diffs == diffs.iloc[0]).mean() > 0.95

            if is_id_name or (unique_ratio > 0.95 and is_sequential):
                id_cols.append(c)

        if id_cols:
            result = result.drop(columns=id_cols)

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Dropped {len(id_cols)} ID-like column(s): {id_cols}" if id_cols else "No ID-like columns detected",
        )
