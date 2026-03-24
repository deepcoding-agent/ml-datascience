"""Statistics handler — describe, shape, nulls, value counts, correlation, outliers, duplicates."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.viz_handler import _style
from api.logger import get_logger

log = get_logger(__name__)


class StatsHandler(BaseHandler):

    @staticmethod
    def handle_describe(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")

        # Single column describe
        if col and col in df.columns:
            desc = df[col].describe().round(4)
            result = desc.to_frame().reset_index()
            result.columns = ["stat", "value"]
            return HandlerResult(success=True, result_df=result, summary=f"Descriptive statistics for '{col}'")

        # Full dataset describe — build a rich summary table
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        rows = []

        for c in df.columns:
            row: dict = {
                "column": c,
                "dtype": str(df[c].dtype),
                "non_null": int(df[c].notna().sum()),
                "null_count": int(df[c].isna().sum()),
                "null_pct": round(df[c].isna().mean() * 100, 1),
                "unique": int(df[c].nunique()),
            }
            if c in num_cols:
                row["mean"] = round(float(df[c].mean()), 2)
                row["std"] = round(float(df[c].std()), 2)
                row["min"] = round(float(df[c].min()), 2)
                row["25%"] = round(float(df[c].quantile(0.25)), 2)
                row["50%"] = round(float(df[c].quantile(0.50)), 2)
                row["75%"] = round(float(df[c].quantile(0.75)), 2)
                row["max"] = round(float(df[c].max()), 2)
            else:
                top = df[c].mode()
                row["top"] = str(top.iloc[0]) if len(top) > 0 else ""
                row["freq"] = int(df[c].value_counts().iloc[0]) if df[c].notna().any() else 0

            rows.append(row)

        result = pd.DataFrame(rows).fillna("")

        # Build summary text
        mem_mb = df.memory_usage(deep=True).sum() / 1024**2
        null_total = int(df.isna().sum().sum())
        dup_count = int(df.duplicated().sum())
        parts = [
            f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns ({mem_mb:.2f} MB)",
            f"Numeric: {len(num_cols)} cols | Categorical: {len(cat_cols)} cols",
        ]
        if null_total > 0:
            null_cols_count = int((df.isna().sum() > 0).sum())
            parts.append(f"Missing: {null_total:,} values across {null_cols_count} columns")
        else:
            parts.append("Missing: none")
        if dup_count > 0:
            parts.append(f"Duplicates: {dup_count:,} rows")
        summary = " | ".join(parts)

        return HandlerResult(success=True, result_df=result, summary=summary)

    @staticmethod
    def handle_shape(df: pd.DataFrame, params: dict) -> HandlerResult:
        mem = df.memory_usage(deep=True).sum() / 1024**2
        num_cols = len(df.select_dtypes(include="number").columns)
        cat_cols = len(df.select_dtypes(include=["object", "category"]).columns)
        null_total = int(df.isna().sum().sum())
        dup_count = int(df.duplicated().sum())
        result = pd.DataFrame([{
            "rows": df.shape[0], "columns": df.shape[1],
            "numeric_cols": num_cols, "categorical_cols": cat_cols,
            "total_nulls": null_total, "duplicates": dup_count,
            "memory_mb": round(mem, 2),
        }])
        summary = (
            f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns ({mem:.2f} MB) | "
            f"Numeric: {num_cols} | Categorical: {cat_cols} | "
            f"Nulls: {null_total:,} | Duplicates: {dup_count:,}"
        )
        return HandlerResult(success=True, result_df=result, summary=summary)

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
                        aspect="auto")
        _style(fig, title="Correlation Matrix")
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

    # ── 1. Cross-tabulation ──────────────────────────────────────────────────

    @staticmethod
    def handle_cross_tab(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Contingency table between two columns (count or normalized)."""
        cols = params.get("columns", [])
        normalize = params.get("normalize", False)
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        col1 = cols[0] if len(cols) > 0 and cols[0] in df.columns else (cat_cols[0] if len(cat_cols) > 0 else None)
        col2 = cols[1] if len(cols) > 1 and cols[1] in df.columns else (cat_cols[1] if len(cat_cols) > 1 else None)

        if not col1 or not col2:
            return HandlerResult(success=False, error="Need 2 columns for cross-tabulation")

        ct = pd.crosstab(df[col1], df[col2], normalize="all" if normalize else False)
        if normalize:
            ct = (ct * 100).round(2)
        result = ct.reset_index()

        fig = px.imshow(ct, text_auto=True, aspect="auto",
                        color_continuous_scale=["#FFF5EB", "#FB8C3C"])
        _style(fig, title=f"{col1} vs {col2}")

        return HandlerResult(
            success=True, result_df=result, charts_plotly=[fig.to_json()],
            summary=f"Cross-tab: {col1} × {col2} ({ct.shape[0]}×{ct.shape[1]})",
        )

    # ── 2. Custom percentile analysis ────────────────────────────────────────

    @staticmethod
    def handle_percentile(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Custom percentile report for numeric columns."""
        col = params.get("column")
        quantiles = params.get("quantiles", [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        num_cols = [col] if col and col in df.columns else df.select_dtypes(include="number").columns.tolist()

        result = df[num_cols].quantile(quantiles).round(4).T.reset_index()
        result.columns = ["column"] + [f"p{int(q*100)}" for q in quantiles]

        return HandlerResult(
            success=True, result_df=result,
            summary=f"Percentile report for {len(num_cols)} numeric columns",
        )

    # ── 3. Normality test ────────────────────────────────────────────────────

    @staticmethod
    def handle_normality_test(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Shapiro-Wilk normality test for numeric columns."""
        from scipy import stats as sp_stats

        col = params.get("column")
        num_cols = [col] if col and col in df.columns else df.select_dtypes(include="number").columns.tolist()
        rows = []

        for c in num_cols:
            data = df[c].dropna()
            sample = data.sample(min(5000, len(data)), random_state=42) if len(data) > 5000 else data
            try:
                stat, p_value = sp_stats.shapiro(sample)
                rows.append({
                    "column": c,
                    "shapiro_stat": round(stat, 4),
                    "p_value": round(p_value, 6),
                    "normal": "Yes" if p_value > 0.05 else "No",
                    "skewness": round(float(data.skew()), 4),
                    "kurtosis": round(float(data.kurtosis()), 4),
                })
            except Exception:
                rows.append({"column": c, "shapiro_stat": None, "p_value": None, "normal": "Error",
                             "skewness": round(float(data.skew()), 4), "kurtosis": round(float(data.kurtosis()), 4)})

        result = pd.DataFrame(rows)
        normal_count = sum(1 for r in rows if r["normal"] == "Yes")
        return HandlerResult(
            success=True, result_df=result,
            summary=f"Normality test: {normal_count}/{len(rows)} columns appear normal (p > 0.05)",
        )

    # ── 4. Class balance ─────────────────────────────────────────────────────

    @staticmethod
    def handle_class_balance(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Analyze target class distribution — important before classification."""
        col = params.get("column")
        if not col or col not in df.columns:
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            num_cols = df.select_dtypes(include="number").columns.tolist()
            # Prefer low-cardinality numeric (likely a label) or first categorical
            for c in num_cols:
                if df[c].nunique() <= 10:
                    col = c
                    break
            if not col:
                col = cat_cols[0] if cat_cols else df.columns[-1]

        vc = df[col].value_counts()
        result = pd.DataFrame({
            "class": vc.index.astype(str),
            "count": vc.values,
            "percentage": (vc.values / len(df) * 100).round(2),
        })

        ratio = vc.max() / vc.min() if vc.min() > 0 else float("inf")
        balanced = "Balanced" if ratio < 2 else ("Moderate imbalance" if ratio < 5 else "Severe imbalance")

        fig = px.bar(result, x="class", y="count", text="percentage")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                          marker_color="#FB8C3C")
        _style(fig, title=f"Class Balance: {col}", bargap=0.3)

        return HandlerResult(
            success=True, result_df=result, charts_plotly=[fig.to_json()],
            summary=f"Class balance '{col}': {len(vc)} classes, ratio {ratio:.1f}:1 — {balanced}",
        )

    # ── 5. Top correlations ──────────────────────────────────────────────────

    @staticmethod
    def handle_top_correlations(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Top N most correlated feature pairs (easier to read than full matrix)."""
        n = params.get("n", 10)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return HandlerResult(success=False, error="Need at least 2 numeric columns")

        corr = df[num_cols].corr().abs()
        # Extract upper triangle pairs
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                pairs.append({
                    "feature_1": corr.columns[i],
                    "feature_2": corr.columns[j],
                    "correlation": round(float(df[num_cols].corr().iloc[i, j]), 4),
                    "abs_correlation": round(float(corr.iloc[i, j]), 4),
                })
        result = pd.DataFrame(pairs).sort_values("abs_correlation", ascending=False).head(n).reset_index(drop=True)

        return HandlerResult(
            success=True, result_df=result,
            summary=f"Top {min(n, len(result))} correlated pairs (highest: {result.iloc[0]['feature_1']} ↔ {result.iloc[0]['feature_2']} = {result.iloc[0]['correlation']})" if len(result) > 0 else "No pairs",
        )

    # ── 6. Kurtosis ──────────────────────────────────────────────────────────

    @staticmethod
    def handle_kurtosis(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Kurtosis per numeric column (complement to skewness)."""
        num_cols = df.select_dtypes(include="number").columns
        kurt = df[num_cols].kurtosis().round(4)
        result = kurt.reset_index()
        result.columns = ["column", "kurtosis"]
        result["shape"] = result["kurtosis"].apply(
            lambda k: "Normal-like" if abs(k) < 1 else ("Heavy-tailed" if k > 1 else "Light-tailed")
        )
        result = result.sort_values("kurtosis", key=abs, ascending=False).reset_index(drop=True)
        return HandlerResult(success=True, result_df=result, summary="Kurtosis per numeric column")

    # ── 7. Zero report ───────────────────────────────────────────────────────

    @staticmethod
    def handle_zero_report(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Count zero values per column (important for sparse data)."""
        rows = []
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                zero_count = int((df[c] == 0).sum())
                zero_pct = round(zero_count / len(df) * 100, 2)
            else:
                zero_count = int((df[c].isin(["0", "", " "])).sum())
                zero_pct = round(zero_count / len(df) * 100, 2)
            rows.append({"column": c, "zero_count": zero_count, "zero_pct": zero_pct})

        result = pd.DataFrame(rows).sort_values("zero_count", ascending=False).reset_index(drop=True)
        total = result["zero_count"].sum()
        return HandlerResult(success=True, result_df=result, summary=f"Total zeros/empty: {total:,}")

    # ── 8. Cardinality report ────────────────────────────────────────────────

    @staticmethod
    def handle_cardinality_report(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Analyze unique ratio per column — identify high/low cardinality."""
        rows = []
        for c in df.columns:
            n_unique = df[c].nunique()
            ratio = round(n_unique / len(df) * 100, 2) if len(df) > 0 else 0
            if ratio > 90:
                label = "ID-like"
            elif n_unique <= 2:
                label = "Binary"
            elif n_unique <= 10:
                label = "Low"
            elif ratio > 50:
                label = "High"
            else:
                label = "Medium"
            rows.append({"column": c, "unique": n_unique, "unique_pct": ratio,
                         "dtype": str(df[c].dtype), "cardinality": label})

        result = pd.DataFrame(rows).sort_values("unique", ascending=False).reset_index(drop=True)
        return HandlerResult(
            success=True, result_df=result,
            summary=f"Cardinality: {sum(1 for r in rows if r['cardinality'] == 'ID-like')} ID-like, "
                    f"{sum(1 for r in rows if r['cardinality'] == 'Binary')} binary, "
                    f"{sum(1 for r in rows if r['cardinality'] == 'High')} high, "
                    f"{sum(1 for r in rows if r['cardinality'] in ('Low', 'Medium'))} low/medium",
        )
