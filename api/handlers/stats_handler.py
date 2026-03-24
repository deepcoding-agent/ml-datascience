"""Statistics handler — describe, shape, nulls, value counts, correlation, outliers, duplicates."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


class StatsHandler(BaseHandler):

    @staticmethod
    def handle_describe(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        if col and col in df.columns:
            desc = df[col].describe().round(4)
            result = desc.to_frame().reset_index()
            result.columns = ["stat", "value"]
        else:
            result = df.describe(include="all").round(4).T.reset_index()
            result.columns = ["column"] + list(result.columns[1:])
        return HandlerResult(success=True, result_df=result, summary=f"Descriptive statistics for {col or 'all columns'}")

    @staticmethod
    def handle_shape(df: pd.DataFrame, params: dict) -> HandlerResult:
        mem = df.memory_usage(deep=True).sum() / 1024**2
        result = pd.DataFrame([{
            "rows": df.shape[0], "columns": df.shape[1],
            "memory_mb": round(mem, 2), "duplicates": int(df.duplicated().sum()),
        }])
        return HandlerResult(success=True, result_df=result, summary=f"Shape: {df.shape[0]:,} rows x {df.shape[1]} cols, {mem:.2f} MB")

    @staticmethod
    def handle_null_report(df: pd.DataFrame, params: dict) -> HandlerResult:
        null_counts = df.isnull().sum()
        null_pct = (df.isnull().mean() * 100).round(2)
        result = pd.DataFrame({
            "column": df.columns,
            "null_count": null_counts.values,
            "null_pct": null_pct.values,
            "dtype": [str(d) for d in df.dtypes.values],
        }).sort_values("null_count", ascending=False).reset_index(drop=True)
        total = int(null_counts.sum())
        return HandlerResult(success=True, result_df=result, summary=f"Total nulls: {total:,} across {(null_counts > 0).sum()} columns")

    @staticmethod
    def handle_value_counts(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        if not col or col not in df.columns:
            # Pick first categorical or first column
            cats = df.select_dtypes(include=["object", "category"]).columns
            col = cats[0] if len(cats) > 0 else df.columns[0]
        n = params.get("n", 10)
        vc = df[col].value_counts().head(n).reset_index()
        vc.columns = [col, "count"]
        return HandlerResult(success=True, result_df=vc, summary=f"Top {n} values in '{col}'")

    @staticmethod
    def handle_unique_values(df: pd.DataFrame, params: dict) -> HandlerResult:
        result = pd.DataFrame({
            "column": df.columns,
            "unique_count": [df[c].nunique() for c in df.columns],
            "dtype": [str(d) for d in df.dtypes.values],
        }).sort_values("unique_count", ascending=False).reset_index(drop=True)
        return HandlerResult(success=True, result_df=result, summary="Unique value counts per column")

    @staticmethod
    def handle_dtypes(df: pd.DataFrame, params: dict) -> HandlerResult:
        result = pd.DataFrame({
            "column": df.columns,
            "dtype": [str(d) for d in df.dtypes.values],
            "null_pct": (df.isnull().mean() * 100).round(1).values,
            "unique": [df[c].nunique() for c in df.columns],
        })
        return HandlerResult(success=True, result_df=result, summary="Column data types")

    @staticmethod
    def handle_correlation(df: pd.DataFrame, params: dict) -> HandlerResult:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return HandlerResult(success=False, error="Need at least 2 numeric columns for correlation")
        corr = df[num_cols].corr().round(4)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlBu_r",
                        title="Correlation Matrix")
        return HandlerResult(
            success=True, result_df=corr.reset_index(),
            charts_plotly=[fig.to_json()],
            summary=f"Correlation matrix for {len(num_cols)} numeric columns",
        )

    @staticmethod
    def handle_skewness(df: pd.DataFrame, params: dict) -> HandlerResult:
        num_cols = df.select_dtypes(include="number").columns
        skew = df[num_cols].skew().round(4)
        result = skew.reset_index()
        result.columns = ["column", "skewness"]
        result = result.sort_values("skewness", key=abs, ascending=False).reset_index(drop=True)
        return HandlerResult(success=True, result_df=result, summary="Skewness per numeric column")

    @staticmethod
    def handle_outlier_report(df: pd.DataFrame, params: dict) -> HandlerResult:
        num_cols = df.select_dtypes(include="number").columns
        rows = []
        for col in num_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            count = int(((df[col] < lower) | (df[col] > upper)).sum())
            rows.append({"column": col, "outlier_count": count, "lower_bound": round(lower, 2), "upper_bound": round(upper, 2)})
        result = pd.DataFrame(rows).sort_values("outlier_count", ascending=False).reset_index(drop=True)
        total = result["outlier_count"].sum()
        return HandlerResult(success=True, result_df=result, summary=f"Total outliers (IQR): {total:,}")

    @staticmethod
    def handle_duplicate_report(df: pd.DataFrame, params: dict) -> HandlerResult:
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            sample = df[df.duplicated(keep="first")].head(5)
            return HandlerResult(success=True, result_df=sample,
                                 summary=f"{dup_count:,} duplicate rows found (showing first 5)")
        return HandlerResult(success=True,
                             result_df=pd.DataFrame([{"duplicate_rows": 0}]),
                             summary="No duplicate rows found")
