"""Transform handler — filter, sort, groupby, assign, encode, scale, pivot, melt, rank, rolling, etc."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


class TransformHandler(BaseHandler):

    @staticmethod
    def handle_filter(df: pd.DataFrame, params: dict) -> HandlerResult:
        col, err = BaseHandler.require_column(df, params.get("column"), params.get("column", ""))
        if err:
            return err
        op = params.get("operator", "==")
        val = params.get("value")
        if val is None:
            return HandlerResult(success=False, error="No filter value provided")
        ops = {"==": "eq", "!=": "ne", ">": "gt", "<": "lt", ">=": "ge", "<=": "le"}
        method = ops.get(op, "eq")
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                val = float(val)
            result = df[getattr(df[col], method)(val)]
        except Exception as e:
            return HandlerResult(success=False, error=f"Filter error: {e}")
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Filtered {col} {op} {val}: {len(df):,} → {len(result):,} rows")

    @staticmethod
    def handle_assign_value(df: pd.DataFrame, params: dict) -> HandlerResult:
        col, err = BaseHandler.require_column(df, params.get("column"), params.get("column", ""))
        if err:
            return err
        val = params.get("value")
        if val is None:
            return HandlerResult(success=False, error="No value to assign")
        result = df.copy()
        result[col] = val
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Set all {len(result):,} rows of '{col}' to {val}")

    @staticmethod
    def handle_sort(df: pd.DataFrame, params: dict) -> HandlerResult:
        col, err = BaseHandler.require_column(df, params.get("column"), params.get("column", ""))
        if err:
            return err
        ascending = params.get("ascending", True)
        result = df.sort_values(col, ascending=ascending).reset_index(drop=True)
        order = "ascending" if ascending else "descending"
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Sorted by '{col}' {order}")

    @staticmethod
    def handle_groupby_agg(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        agg = params.get("agg", "count")
        if not col or col not in df.columns:
            return HandlerResult(success=False, error=f"Column '{col}' not found")
        if agg == "count":
            result = df.groupby(col, dropna=False).size().reset_index(name="count")
        else:
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if not num_cols:
                return HandlerResult(success=False, error="No numeric columns to aggregate")
            result = df.groupby(col, dropna=False)[num_cols].agg(agg).reset_index()
        result = result.sort_values(result.columns[-1], ascending=False).reset_index(drop=True)
        return HandlerResult(success=True, result_df=result, output_type="query",
                             summary=f"Grouped by '{col}' with {agg}")

    @staticmethod
    def handle_add_column(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        expression = params.get("expression", "")
        if not col:
            return HandlerResult(success=False, error="No column name provided")
        result = df.copy()
        try:
            result[col] = result.eval(expression) if expression else 0
        except Exception as e:
            return HandlerResult(success=False, error=f"Expression error: {e}")
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Added column '{col}'")

    @staticmethod
    def handle_encode_label(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        result = df.copy()
        cols = [col] if col and col in result.columns else result.select_dtypes(include=["object", "category"]).columns.tolist()
        mappings = {}
        for c in cols:
            le = LabelEncoder()
            result[c] = le.fit_transform(result[c].astype(str))
            mappings[c] = {str(cls): int(i) for i, cls in enumerate(le.classes_)}
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Label-encoded {len(cols)} columns", metadata={"mappings": mappings})

    @staticmethod
    def handle_encode_onehot(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        result = df.copy()
        cols = [col] if col and col in result.columns else result.select_dtypes(include=["object", "category"]).columns.tolist()
        result = pd.get_dummies(result, columns=cols, drop_first=True, dtype=int)
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"One-hot encoded {len(cols)} columns → {result.shape[1]} total cols")

    @staticmethod
    def handle_scale_minmax(df: pd.DataFrame, params: dict) -> HandlerResult:
        result = df.copy()
        num_cols = result.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return HandlerResult(success=False, error="No numeric columns to scale")
        result[num_cols] = MinMaxScaler().fit_transform(result[num_cols])
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"MinMax scaled {len(num_cols)} numeric columns to [0,1]")

    @staticmethod
    def handle_scale_standard(df: pd.DataFrame, params: dict) -> HandlerResult:
        result = df.copy()
        num_cols = result.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return HandlerResult(success=False, error="No numeric columns to scale")
        result[num_cols] = StandardScaler().fit_transform(result[num_cols])
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Standard scaled {len(num_cols)} numeric columns (mean=0, std=1)")

    @staticmethod
    def handle_bin_column(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        bins = params.get("n", 5)
        if not col or col not in df.columns:
            return HandlerResult(success=False, error=f"Column '{col}' not found")
        result = df.copy()
        result[f"{col}_bin"] = pd.cut(result[col], bins=bins, labels=False)
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Binned '{col}' into {bins} bins → '{col}_bin'")

    @staticmethod
    def handle_inject_null(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Inject random NaN values into a copy of the DataFrame."""
        fraction = params.get("value", 15)
        if fraction > 1:
            fraction = fraction / 100.0  # convert 15 → 0.15
        result = df.copy()
        for col in result.columns:
            n_nulls = int(len(result) * fraction)
            if n_nulls > 0:
                null_indices = np.random.choice(result.index, size=n_nulls, replace=False)
                result.loc[null_indices, col] = np.nan
        total_nulls = int(result.isnull().sum().sum())
        total_cells = result.shape[0] * result.shape[1]
        actual_pct = total_nulls / total_cells * 100 if total_cells > 0 else 0
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Injected ~{fraction*100:.0f}% null values. Actual: {actual_pct:.1f}% ({total_nulls:,}/{total_cells:,} cells)",
        )

    @staticmethod
    def handle_sample_rows(df: pd.DataFrame, params: dict) -> HandlerResult:
        n = min(params.get("n", 10), len(df))
        result = df.sample(n=n, random_state=42)
        return HandlerResult(success=True, result_df=result, output_type="query",
                             summary=f"Random sample of {n} rows")

    @staticmethod
    def handle_head(df: pd.DataFrame, params: dict) -> HandlerResult:
        n = params.get("n", 10)
        result = df.head(n)
        return HandlerResult(success=True, result_df=result, output_type="query",
                             summary=f"First {n} rows")

    @staticmethod
    def handle_tail(df: pd.DataFrame, params: dict) -> HandlerResult:
        n = params.get("n", 10)
        result = df.tail(n)
        return HandlerResult(success=True, result_df=result, output_type="query",
                             summary=f"Last {n} rows")

    # ── 1. Pivot table ───────────────────────────────────────────────────────

    @staticmethod
    def handle_pivot(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Pivot table: rows=index, columns=columns, values=values, aggfunc=agg."""
        index = params.get("index") or params.get("column")
        columns = params.get("columns")
        values = params.get("values")
        agg = params.get("agg", "mean")

        if not index or index not in df.columns:
            return HandlerResult(success=False, error=f"Index column '{index}' not found")

        try:
            if columns and columns in df.columns and values and values in df.columns:
                result = pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=agg)
                result = result.reset_index()
                result.columns = [str(c) for c in result.columns]
            elif values and values in df.columns:
                result = pd.pivot_table(df, index=index, values=values, aggfunc=agg).reset_index()
            else:
                num_cols = df.select_dtypes(include="number").columns.tolist()
                result = pd.pivot_table(df, index=index, values=num_cols, aggfunc=agg).reset_index()

            return HandlerResult(success=True, result_df=result, output_type="query",
                                 summary=f"Pivot table: index='{index}', agg={agg} ({result.shape[0]}×{result.shape[1]})")
        except Exception as e:
            return HandlerResult(success=False, error=f"Pivot error: {e}")

    # ── 2. Melt (wide → long) ────────────────────────────────────────────────

    @staticmethod
    def handle_melt(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Unpivot/melt: wide → long format."""
        id_vars = params.get("id_vars", [])
        value_vars = params.get("value_vars", [])

        if not id_vars:
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            id_vars = cat_cols[:2] if cat_cols else [df.columns[0]]
        id_vars = [c for c in id_vars if c in df.columns]

        if not value_vars:
            value_vars = [c for c in df.columns if c not in id_vars]
        value_vars = [c for c in value_vars if c in df.columns]

        result = pd.melt(df, id_vars=id_vars, value_vars=value_vars,
                         var_name="variable", value_name="value")
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Melted: {len(value_vars)} columns → long format ({len(result):,} rows)")

    # ── 3. Robust scaler ─────────────────────────────────────────────────────

    @staticmethod
    def handle_scale_robust(df: pd.DataFrame, params: dict) -> HandlerResult:
        """RobustScaler — scales using median/IQR, resistant to outliers."""
        result = df.copy()
        num_cols = result.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return HandlerResult(success=False, error="No numeric columns to scale")
        result[num_cols] = RobustScaler().fit_transform(result[num_cols])
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Robust scaled {len(num_cols)} numeric columns (median-centered, IQR-scaled)")

    # ── 4. N largest ──────────────────────────────────────────────────────────

    @staticmethod
    def handle_nlargest(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Top N rows by column value."""
        col = params.get("column")
        n = params.get("n", 10)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col = col if col and col in df.columns else (num_cols[0] if num_cols else df.columns[0])
        result = df.nlargest(min(n, len(df)), col)
        return HandlerResult(success=True, result_df=result, output_type="query",
                             summary=f"Top {min(n, len(df))} rows by '{col}'")

    # ── 5. N smallest ────────────────────────────────────────────────────────

    @staticmethod
    def handle_nsmallest(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Bottom N rows by column value."""
        col = params.get("column")
        n = params.get("n", 10)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col = col if col and col in df.columns else (num_cols[0] if num_cols else df.columns[0])
        result = df.nsmallest(min(n, len(df)), col)
        return HandlerResult(success=True, result_df=result, output_type="query",
                             summary=f"Bottom {min(n, len(df))} rows by '{col}'")

    # ── 6. Rank ──────────────────────────────────────────────────────────────

    @staticmethod
    def handle_rank(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Add rank column based on a numeric column."""
        col = params.get("column")
        ascending = params.get("ascending", True)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col = col if col and col in df.columns else (num_cols[0] if num_cols else None)
        if not col:
            return HandlerResult(success=False, error="No column to rank")
        result = df.copy()
        result[f"{col}_rank"] = result[col].rank(ascending=ascending, method="min").astype(int)
        result = result.sort_values(f"{col}_rank").reset_index(drop=True)
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Ranked by '{col}' {'ascending' if ascending else 'descending'}")

    # ── 7. Cumulative ────────────────────────────────────────────────────────

    @staticmethod
    def handle_cumulative(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Cumulative sum, count, max, or min."""
        col = params.get("column")
        func = params.get("agg", "sum")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col = col if col and col in df.columns else (num_cols[0] if num_cols else None)
        if not col:
            return HandlerResult(success=False, error="No column for cumulative")
        result = df.copy()
        if func == "sum":
            result[f"{col}_cumsum"] = result[col].cumsum()
        elif func == "max":
            result[f"{col}_cummax"] = result[col].cummax()
        elif func == "min":
            result[f"{col}_cummin"] = result[col].cummin()
        elif func == "count":
            result[f"{col}_cumcount"] = range(1, len(result) + 1)
        else:
            result[f"{col}_cumsum"] = result[col].cumsum()
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Cumulative {func} of '{col}'")

    # ── 8. Rolling window ────────────────────────────────────────────────────

    @staticmethod
    def handle_rolling(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Rolling/moving window: mean, sum, std, min, max."""
        col = params.get("column")
        window = params.get("window", 3)
        func = params.get("agg", "mean")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col = col if col and col in df.columns else (num_cols[0] if num_cols else None)
        if not col:
            return HandlerResult(success=False, error="No column for rolling")
        result = df.copy()
        roller = result[col].rolling(window=window, min_periods=1)
        result[f"{col}_rolling_{func}_{window}"] = getattr(roller, func)().round(4)
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Rolling {func} (window={window}) of '{col}'")

    # ── 9. Round values ──────────────────────────────────────────────────────

    @staticmethod
    def handle_round_values(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Round numeric columns to N decimal places."""
        col = params.get("column")
        decimals = params.get("decimals", 2)
        result = df.copy()
        if col and col in result.columns:
            result[col] = result[col].round(decimals)
            summary = f"Rounded '{col}' to {decimals} decimals"
        else:
            num_cols = result.select_dtypes(include="number").columns.tolist()
            result[num_cols] = result[num_cols].round(decimals)
            summary = f"Rounded {len(num_cols)} numeric columns to {decimals} decimals"
        return HandlerResult(success=True, result_df=result, output_type="generate", summary=summary)

    # ── 10. Split column ─────────────────────────────────────────────────────

    @staticmethod
    def handle_split_column(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Split a string column by delimiter into multiple columns."""
        col = params.get("column")
        delimiter = params.get("delimiter", ",")
        if not col or col not in df.columns:
            return HandlerResult(success=False, error=f"Column '{col}' not found")
        result = df.copy()
        split_df = result[col].astype(str).str.split(delimiter, expand=True)
        split_df.columns = [f"{col}_{i+1}" for i in range(split_df.shape[1])]
        for c in split_df.columns:
            split_df[c] = split_df[c].str.strip()
        result = pd.concat([result, split_df], axis=1)
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Split '{col}' by '{delimiter}' → {split_df.shape[1]} new columns")

    # ── 11. Concat columns ───────────────────────────────────────────────────

    @staticmethod
    def handle_concat_columns(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Concatenate string columns into a new column."""
        cols = params.get("columns", [])
        separator = params.get("separator", "_")
        new_name = params.get("new_name", "combined")
        cols = [c for c in cols if c in df.columns]
        if len(cols) < 2:
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            cols = cat_cols[:2] if len(cat_cols) >= 2 else df.columns[:2].tolist()
        result = df.copy()
        result[new_name] = result[cols[0]].astype(str)
        for c in cols[1:]:
            result[new_name] = result[new_name] + separator + result[c].astype(str)
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Concatenated {cols} → '{new_name}'")

    # ── 12. Quantile-based binning ───────────────────────────────────────────

    @staticmethod
    def handle_qcut(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Quantile-based binning (equal-frequency bins)."""
        col = params.get("column")
        q = params.get("n", 4)
        if not col or col not in df.columns:
            return HandlerResult(success=False, error=f"Column '{col}' not found")
        result = df.copy()
        try:
            result[f"{col}_qbin"] = pd.qcut(result[col], q=q, labels=False, duplicates="drop")
            actual_bins = result[f"{col}_qbin"].nunique()
            return HandlerResult(success=True, result_df=result, output_type="generate",
                                 summary=f"Quantile-binned '{col}' into {actual_bins} bins → '{col}_qbin'")
        except Exception as e:
            return HandlerResult(success=False, error=f"Quantile binning error: {e}")
