"""Transform handler — filter, sort, groupby, assign, add column, encode, scale, sample."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

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
