"""Cleaning handler — drop/fill nulls, remove duplicates, fix dtypes, rename, drop, outliers, etc."""
from __future__ import annotations

import html
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

    # ── NEW HANDLERS (21–50) ─────────────────────────────────────────────────

    @staticmethod
    def handle_fix_numeric_strings(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Convert formatted numeric strings ('$1,234' / '1.234,56') to float."""
        col = params.get("column")
        result = df.copy()
        cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
        converted: list[str] = []

        for c in cols:
            sample = result[c].dropna().head(100).astype(str)
            if sample.empty:
                continue
            cleaned = sample.str.replace(r"[^\d.,\-]", "", regex=True)
            # Detect European format (1.234,56) vs US format (1,234.56)
            has_european = cleaned.str.contains(r"\d\.\d{3},", regex=True).any()
            if has_european:
                cleaned = cleaned.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
            else:
                cleaned = cleaned.str.replace(",", "", regex=False)
            coerced = pd.to_numeric(cleaned, errors="coerce")
            if coerced.notna().mean() >= 0.6:
                full = result[c].astype(str).str.replace(r"[^\d.,\-]", "", regex=True)
                if has_european:
                    full = full.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
                else:
                    full = full.str.replace(",", "", regex=False)
                result[c] = pd.to_numeric(full, errors="coerce")
                converted.append(c)

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Converted {len(converted)} column(s) from formatted strings to numeric: {converted}" if converted else "No numeric-string columns detected",
        )

    @staticmethod
    def handle_clean_column_names(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Remove special chars, spaces to underscore, lowercase all column names."""
        result = df.copy()
        mapping: dict[str, str] = {}
        new_cols = []
        for c in result.columns:
            new = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(c))
            new = re.sub(r"[^a-zA-Z0-9_]", "_", new).lower().strip("_")
            new = re.sub(r"_+", "_", new)
            if new != c:
                mapping[c] = new
            new_cols.append(new)
        result.columns = new_cols
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Cleaned {len(mapping)} column names to snake_case" if mapping else "All column names already clean",
            metadata={"renamed": mapping},
        )

    @staticmethod
    def handle_remove_empty_rows(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Remove rows where all values are null or empty string."""
        result = df.copy()
        before = len(result)
        mask = result.apply(
            lambda row: row.isna().all() or (row.astype(str).str.strip() == "").all(),
            axis=1,
        )
        result = result[~mask].reset_index(drop=True)
        removed = before - len(result)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Removed {removed:,} completely empty rows ({before:,} → {len(result):,})",
        )

    @staticmethod
    def handle_fill_mode(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Fill nulls with mode (most frequent value)."""
        col = params.get("column")
        result = df.copy()
        cols = [col] if col and col in result.columns else result.columns.tolist()
        filled: dict[str, str] = {}

        for c in cols:
            if result[c].isnull().sum() == 0:
                continue
            mode = result[c].mode()
            if not mode.empty:
                result[c] = result[c].fillna(mode.iloc[0])
                filled[c] = str(mode.iloc[0])

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Filled nulls with mode in {len(filled)} column(s)",
            metadata={"mode_values": filled},
        )

    @staticmethod
    def handle_fill_forward_backward(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Fill nulls using forward fill then backward fill."""
        col = params.get("column")
        result = df.copy()
        before_nulls = int(result.isna().sum().sum())

        if col and col in result.columns:
            result[col] = result[col].ffill().bfill()
        else:
            result = result.ffill().bfill()

        after_nulls = int(result.isna().sum().sum())
        filled = before_nulls - after_nulls
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Forward+backward fill: resolved {filled:,} nulls ({before_nulls:,} → {after_nulls:,})",
        )

    @staticmethod
    def handle_fix_boolean(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Standardize boolean-like values (yes/no, true/false, Y/N, 1/0) to bool."""
        col = params.get("column")
        result = df.copy()
        true_vals = {"yes", "y", "true", "t", "1", "1.0", "si", "on"}
        false_vals = {"no", "n", "false", "f", "0", "0.0", "off"}
        converted: list[str] = []

        cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
        for c in cols:
            unique = result[c].dropna().astype(str).str.strip().str.lower().unique()
            all_bool = all(v in true_vals | false_vals for v in unique)
            if all_bool and len(unique) > 0:
                result[c] = result[c].astype(str).str.strip().str.lower().map(
                    lambda v, t=true_vals: True if v in t else (False if v in false_vals else None)
                )
                converted.append(c)

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Converted {len(converted)} column(s) to boolean: {converted}" if converted else "No boolean-like columns detected",
        )

    @staticmethod
    def handle_fix_encoding(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Fix common mojibake/encoding issues in string columns."""
        col = params.get("column")
        result = df.copy()
        replacements = {
            "\xc3\xa2\xe2\x82\xac\xe2\x84\xa2": "'",
            "\xc3\xa2\xe2\x82\xac\xcb\x9c": "'",
            "\xc3\xa2\xe2\x82\xac\xc5\x93": '"',
            "\xc3\xa2\xe2\x82\xac\xc2\x9d": '"',
            "\xc3\xa2\xe2\x82\xac\xe2\x80\x9c": "\u2014",
            "\xc3\xa2\xe2\x82\xac\xe2\x80\x9d": "\u2013",
            "\xc3\xa2\xe2\x82\xac\xc2\xa6": "\u2026",
            "\xc3\x83\xc2\xa9": "\u00e9",
            "\xc3\x83\xc2\xa8": "\u00e8",
            "\xc3\x83\xc2\xbc": "\u00fc",
            "\xc3\x83\xc2\xb6": "\u00f6",
            "\xc3\x83\xc2\xa4": "\u00e4",
            "\xc3\x83\xc2\xb1": "\u00f1",
            "\xc2\xc2": "",
            "\x00": "",
        }
        cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
        for c in cols:
            for bad, good in replacements.items():
                result[c] = result[c].astype(str).str.replace(bad, good, regex=False)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Fixed encoding issues in {len(cols)} string column(s)",
        )

    @staticmethod
    def handle_remove_html_tags(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Strip HTML/XML tags from string columns."""
        col = params.get("column")
        result = df.copy()
        cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
        tag_re = re.compile(r"<[^>]+>")
        for c in cols:
            result[c] = result[c].astype(str).apply(lambda v: html.unescape(tag_re.sub("", v)))
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Stripped HTML tags from {len(cols)} column(s)",
        )

    @staticmethod
    def handle_clean_currency(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Clean currency strings ($, EUR, ¥, commas) to float."""
        col = params.get("column")
        result = df.copy()
        cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
        converted: list[str] = []

        currency_re = re.compile(r"[$€£¥₹₩₫฿,\s]")
        for c in cols:
            sample = result[c].dropna().head(100).astype(str)
            if sample.empty:
                continue
            has_currency = sample.str.contains(r"[$€£¥₹₩₫฿]", regex=True).any()
            if not has_currency:
                continue
            cleaned = result[c].astype(str).apply(lambda v: currency_re.sub("", v))
            coerced = pd.to_numeric(cleaned, errors="coerce")
            if coerced.notna().mean() >= 0.5:
                result[c] = coerced
                converted.append(c)

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Cleaned currency in {len(converted)} column(s): {converted}" if converted else "No currency columns detected",
        )

    @staticmethod
    def handle_standardize_dates(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Parse and standardize mixed date formats to consistent datetime."""
        col = params.get("column")
        date_format = params.get("format", "%Y-%m-%d")
        result = df.copy()
        cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
        converted: list[str] = []

        for c in cols:
            sample = result[c].dropna().head(100)
            try:
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                if parsed.notna().mean() >= 0.7:
                    result[c] = pd.to_datetime(result[c], errors="coerce", format="mixed")
                    result[c] = result[c].dt.strftime(date_format).replace("NaT", None)
                    converted.append(c)
            except Exception:
                pass

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Standardized dates in {len(converted)} column(s) to {date_format}" if converted else "No date columns detected",
        )

    @staticmethod
    def handle_remove_non_ascii(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Remove non-ASCII characters from string columns."""
        col = params.get("column")
        result = df.copy()
        cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
        for c in cols:
            result[c] = result[c].astype(str).str.encode("ascii", errors="ignore").str.decode("ascii")
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Removed non-ASCII characters from {len(cols)} column(s)",
        )

    @staticmethod
    def handle_remove_special_chars(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Remove special characters, keep alphanumeric + spaces."""
        col = params.get("column")
        keep_pattern = params.get("keep", r"[^a-zA-Z0-9\s]")
        result = df.copy()
        cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
        for c in cols:
            result[c] = result[c].astype(str).str.replace(keep_pattern, "", regex=True)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Removed special characters from {len(cols)} column(s)",
        )

    @staticmethod
    def handle_normalize_text_case(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Normalize text to title/upper/lower/sentence case."""
        col = params.get("column")
        case = params.get("case", "lower")
        result = df.copy()
        cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()

        for c in cols:
            if case == "upper":
                result[c] = result[c].str.upper()
            elif case == "title":
                result[c] = result[c].str.title()
            elif case == "sentence":
                result[c] = result[c].str.capitalize()
            else:  # lower
                result[c] = result[c].str.lower()

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Normalized {len(cols)} column(s) to {case} case",
        )

    @staticmethod
    def handle_cap_outliers_percentile(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Cap outliers at Nth percentile (e.g. 1st and 99th)."""
        col = params.get("column")
        lower_pct = params.get("lower", 1)
        upper_pct = params.get("upper", 99)
        result = df.copy()
        cols = [col] if col and col in result.columns else result.select_dtypes(include="number").columns.tolist()
        capped: dict[str, int] = {}

        for c in cols:
            lo = result[c].quantile(lower_pct / 100)
            hi = result[c].quantile(upper_pct / 100)
            mask = (result[c] < lo) | (result[c] > hi)
            count = int(mask.sum())
            if count > 0:
                result[c] = result[c].clip(lower=lo, upper=hi)
                capped[c] = count

        total = sum(capped.values())
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Capped {total:,} values at p{lower_pct}/p{upper_pct} across {len(capped)} column(s)",
            metadata={"capped": capped},
        )

    @staticmethod
    def handle_fill_median_by_group(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Fill nulls with group-level median."""
        group_col = params.get("group_column") or params.get("column")
        value_col = params.get("value_column")
        if not group_col or group_col not in df.columns:
            return HandlerResult(success=False, error=f"Group column '{group_col}' not found")
        result = df.copy()
        before_nulls = int(result.isna().sum().sum())

        if value_col and value_col in result.columns:
            fill_cols = [value_col]
        else:
            fill_cols = result.select_dtypes(include="number").columns.tolist()

        for c in fill_cols:
            group_median = result.groupby(group_col)[c].transform("median")
            result[c] = result[c].fillna(group_median)

        after_nulls = int(result.isna().sum().sum())
        filled = before_nulls - after_nulls
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Filled {filled:,} nulls with group median (grouped by '{group_col}')",
        )

    @staticmethod
    def handle_remove_zero_rows(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Remove rows where specified column(s) are zero."""
        col = params.get("column")
        result = df.copy()
        before = len(result)
        cols = [col] if col and col in result.columns else result.select_dtypes(include="number").columns.tolist()
        mask = result[cols].eq(0).all(axis=1)
        result = result[~mask].reset_index(drop=True)
        removed = before - len(result)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Removed {removed:,} rows with all-zero values ({before:,} → {len(result):,})",
        )

    @staticmethod
    def handle_remove_negative(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Remove rows with negative values in numeric columns."""
        col = params.get("column")
        result = df.copy()
        before = len(result)
        cols = [col] if col and col in result.columns else result.select_dtypes(include="number").columns.tolist()
        mask = result[cols].lt(0).any(axis=1)
        result = result[~mask].reset_index(drop=True)
        removed = before - len(result)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Removed {removed:,} rows with negative values ({before:,} → {len(result):,})",
        )

    @staticmethod
    def handle_standardize_categories(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Merge similar categories by stripping, lowering, and mapping common variants."""
        col = params.get("column")
        mapping = params.get("mapping", {})
        if not col or col not in df.columns:
            return HandlerResult(success=False, error=f"Column '{col}' not found")
        result = df.copy()
        before_unique = result[col].nunique()
        # Strip + lowercase
        result[col] = result[col].astype(str).str.strip().str.lower()
        # Apply explicit mapping if provided
        if mapping:
            result[col] = result[col].replace(mapping)
        after_unique = result[col].nunique()
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Standardized '{col}' categories: {before_unique} → {after_unique} unique values",
        )

    @staticmethod
    def handle_remove_high_null_cols(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Drop columns above a null-percentage threshold."""
        threshold = params.get("threshold", 0.5)
        result = df.copy()
        null_pct = result.isnull().mean()
        cols_to_drop = null_pct[null_pct > threshold].index.tolist()
        if cols_to_drop:
            result = result.drop(columns=cols_to_drop)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Dropped {len(cols_to_drop)} column(s) with >{threshold*100:.0f}% nulls: {cols_to_drop}" if cols_to_drop else f"No columns exceed {threshold*100:.0f}% null threshold",
        )

    @staticmethod
    def handle_clean_phone_numbers(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Standardize phone numbers to digits-only format."""
        col = params.get("column")
        if not col or col not in df.columns:
            return HandlerResult(success=False, error=f"Column '{col}' not found")
        result = df.copy()
        result[col] = (
            result[col].astype(str)
            .str.replace(r"[^\d+]", "", regex=True)
            .str.strip("+")
        )
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Cleaned phone numbers in '{col}' to digits-only",
        )

    @staticmethod
    def handle_split_name(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Split 'John Doe' into first_name and last_name columns."""
        col = params.get("column")
        if not col or col not in df.columns:
            return HandlerResult(success=False, error=f"Column '{col}' not found")
        result = df.copy()
        parts = result[col].astype(str).str.strip().str.split(r"\s+", n=1, expand=True)
        result["first_name"] = parts[0] if 0 in parts.columns else ""
        result["last_name"] = parts[1] if 1 in parts.columns else ""
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Split '{col}' into first_name and last_name columns",
        )

    @staticmethod
    def handle_fix_whitespace_names(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Fix excess whitespace in name/text: ' John  Doe ' → 'John Doe'."""
        col = params.get("column")
        result = df.copy()
        cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
        for c in cols:
            result[c] = result[c].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Fixed whitespace in {len(cols)} column(s)",
        )

    @staticmethod
    def handle_remove_urls(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Remove URLs from text columns."""
        col = params.get("column")
        result = df.copy()
        cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
        url_re = re.compile(r"https?://\S+|www\.\S+")
        for c in cols:
            result[c] = result[c].astype(str).str.replace(url_re, "", regex=True).str.strip()
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Removed URLs from {len(cols)} column(s)",
        )

    @staticmethod
    def handle_remove_emails(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Remove email addresses from text columns."""
        col = params.get("column")
        result = df.copy()
        cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
        email_re = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
        for c in cols:
            result[c] = result[c].astype(str).str.replace(email_re, "", regex=True).str.strip()
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Removed email addresses from {len(cols)} column(s)",
        )

    @staticmethod
    def handle_fix_mixed_types(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Convert mixed-type columns to a consistent type (numeric preferred, else string)."""
        col = params.get("column")
        result = df.copy()
        cols = [col] if col and col in result.columns else result.columns.tolist()
        converted: dict[str, str] = {}

        for c in cols:
            if result[c].apply(type).nunique() <= 1:
                continue
            # Try numeric first
            coerced = pd.to_numeric(result[c], errors="coerce")
            if coerced.notna().mean() >= 0.7:
                result[c] = coerced
                converted[c] = "numeric"
            else:
                result[c] = result[c].astype(str)
                converted[c] = "string"

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Fixed mixed types in {len(converted)} column(s): {converted}" if converted else "No mixed-type columns found",
            metadata={"converted": converted},
        )

    @staticmethod
    def handle_fill_with_distribution(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Fill nulls by sampling from the column's existing distribution."""
        col = params.get("column")
        seed = params.get("seed", 42)
        result = df.copy()
        rng = np.random.default_rng(seed)
        cols = [col] if col and col in result.columns else result.columns.tolist()
        filled_count = 0

        for c in cols:
            null_mask = result[c].isna()
            n_nulls = int(null_mask.sum())
            if n_nulls == 0:
                continue
            non_null = result[c].dropna().values
            if len(non_null) == 0:
                continue
            sampled = rng.choice(non_null, size=n_nulls, replace=True)
            result.loc[null_mask, c] = sampled
            filled_count += n_nulls

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Filled {filled_count:,} nulls by sampling from column distributions",
        )

    @staticmethod
    def handle_remove_rare_categories(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Replace categories with fewer than N occurrences with 'Other'."""
        col = params.get("column")
        min_count = params.get("min_count", 5)
        replacement = params.get("replacement", "Other")
        if not col or col not in df.columns:
            return HandlerResult(success=False, error=f"Column '{col}' not found")
        result = df.copy()
        counts = result[col].value_counts()
        rare = counts[counts < min_count].index.tolist()
        result[col] = result[col].replace(rare, replacement)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Replaced {len(rare)} rare categories (count < {min_count}) with '{replacement}' in '{col}'",
        )

    @staticmethod
    def handle_dedup_keep_latest(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Deduplicate by a column, keeping the row with the latest value in another column."""
        key_col = params.get("column") or params.get("key_column")
        date_col = params.get("date_column") or params.get("sort_column")
        if not key_col or key_col not in df.columns:
            return HandlerResult(success=False, error=f"Key column '{key_col}' not found")
        if not date_col or date_col not in df.columns:
            return HandlerResult(success=False, error=f"Date/sort column '{date_col}' not found — needed to determine 'latest'")
        result = df.sort_values(date_col, ascending=True).drop_duplicates(subset=[key_col], keep="last").reset_index(drop=True)
        removed = len(df) - len(result)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Deduped by '{key_col}' keeping latest by '{date_col}': removed {removed:,} rows",
        )

    @staticmethod
    def handle_fix_date_outliers(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Remove or clip dates outside a valid range."""
        col = params.get("column")
        min_date = params.get("min_date", "1900-01-01")
        max_date = params.get("max_date", "2099-12-31")
        action = params.get("action", "remove")
        if not col or col not in df.columns:
            return HandlerResult(success=False, error=f"Column '{col}' not found")
        result = df.copy()
        result[col] = pd.to_datetime(result[col], errors="coerce", format="mixed")
        lo = pd.Timestamp(min_date)
        hi = pd.Timestamp(max_date)
        before = len(result)

        if action == "clip":
            result[col] = result[col].clip(lower=lo, upper=hi)
            summary = f"Clipped dates in '{col}' to [{min_date}, {max_date}]"
        else:  # remove
            mask = (result[col] >= lo) & (result[col] <= hi) | result[col].isna()
            result = result[mask].reset_index(drop=True)
            removed = before - len(result)
            summary = f"Removed {removed:,} rows with dates outside [{min_date}, {max_date}] in '{col}'"

        return HandlerResult(success=True, result_df=result, output_type="generate", summary=summary)

    @staticmethod
    def handle_clean_text_whitespace(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Normalize all whitespace: double spaces, tabs, newlines → single space."""
        col = params.get("column")
        result = df.copy()
        cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
        for c in cols:
            result[c] = (
                result[c].astype(str)
                .str.replace(r"[\t\n\r]+", " ", regex=True)
                .str.replace(r"\s{2,}", " ", regex=True)
                .str.strip()
            )
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Normalized whitespace in {len(cols)} column(s)",
        )
