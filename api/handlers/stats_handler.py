"""Statistics handler — 50 handlers for descriptive stats, tests, profiling."""
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
        total = int(df[col].count())
        vc["percentage"] = (vc["count"] / max(total, 1) * 100).round(2)
        return HandlerResult(success=True, result_df=vc, summary=f"Top {n} values in '{col}' (total={total:,})")

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

    # ── 9. Mode report ────────────────────────────────────────────────────────

    @staticmethod
    def handle_mode_report(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Mode values per column."""
        rows = []
        for c in df.columns:
            modes = df[c].mode()
            mode_val = str(modes.iloc[0]) if len(modes) > 0 else "N/A"
            mode_count = int((df[c] == modes.iloc[0]).sum()) if len(modes) > 0 else 0
            rows.append({
                "column": c, "mode": mode_val, "mode_count": mode_count,
                "mode_pct": round(mode_count / len(df) * 100, 2) if len(df) > 0 else 0,
                "n_modes": len(modes),
            })
        result = pd.DataFrame(rows)
        return HandlerResult(success=True, result_df=result, summary="Mode values per column")

    # ── 10. Variance report ───────────────────────────────────────────────────

    @staticmethod
    def handle_variance_report(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Variance per numeric column."""
        num_cols = df.select_dtypes(include="number").columns
        var = df[num_cols].var().round(4)
        result = var.reset_index()
        result.columns = ["column", "variance"]
        result = result.sort_values("variance", ascending=False).reset_index(drop=True)
        return HandlerResult(success=True, result_df=result, summary="Variance per numeric column")

    # ── 11. Range report ──────────────────────────────────────────────────────

    @staticmethod
    def handle_range_report(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Range (max - min) per numeric column."""
        num_cols = df.select_dtypes(include="number").columns
        rows = []
        for c in num_cols:
            mn, mx = float(df[c].min()), float(df[c].max())
            rows.append({"column": c, "min": round(mn, 4), "max": round(mx, 4), "range": round(mx - mn, 4)})
        result = pd.DataFrame(rows).sort_values("range", ascending=False).reset_index(drop=True)
        return HandlerResult(success=True, result_df=result, summary="Range per numeric column")

    # ── 12. IQR report ────────────────────────────────────────────────────────

    @staticmethod
    def handle_iqr_report(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Interquartile range per numeric column."""
        num_cols = df.select_dtypes(include="number").columns
        rows = []
        for c in num_cols:
            q1, q3 = float(df[c].quantile(0.25)), float(df[c].quantile(0.75))
            rows.append({"column": c, "Q1": round(q1, 4), "Q3": round(q3, 4), "IQR": round(q3 - q1, 4)})
        result = pd.DataFrame(rows).sort_values("IQR", ascending=False).reset_index(drop=True)
        return HandlerResult(success=True, result_df=result, summary="IQR per numeric column")

    # ── 13. Z-score report ────────────────────────────────────────────────────

    @staticmethod
    def handle_z_score_report(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Z-score analysis: flag extreme z-scores per column."""
        col = params.get("column")
        num_cols = [col] if col and col in df.columns else df.select_dtypes(include="number").columns.tolist()
        rows = []
        for c in num_cols:
            data = df[c].dropna()
            if data.std() == 0:
                rows.append({"column": c, "max_abs_z": 0, "n_beyond_2": 0, "n_beyond_3": 0})
                continue
            z = ((data - data.mean()) / data.std()).abs()
            rows.append({
                "column": c,
                "max_abs_z": round(float(z.max()), 4),
                "n_beyond_2": int((z > 2).sum()),
                "n_beyond_3": int((z > 3).sum()),
            })
        result = pd.DataFrame(rows).sort_values("max_abs_z", ascending=False).reset_index(drop=True)
        return HandlerResult(success=True, result_df=result, summary=f"Z-score analysis for {len(num_cols)} columns")

    # ── 14. Chi-squared test ──────────────────────────────────────────────────

    @staticmethod
    def handle_chi2_test(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Chi-squared independence test between two categorical columns."""
        from scipy import stats as sp_stats

        cols = params.get("columns", [])
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        col1 = cols[0] if len(cols) > 0 and cols[0] in df.columns else (cat_cols[0] if len(cat_cols) > 0 else None)
        col2 = cols[1] if len(cols) > 1 and cols[1] in df.columns else (cat_cols[1] if len(cat_cols) > 1 else None)
        if not col1 or not col2:
            return HandlerResult(success=False, error="Need 2 categorical columns for chi-squared test")

        ct = pd.crosstab(df[col1], df[col2])
        chi2, p, dof, expected = sp_stats.chi2_contingency(ct)
        result = pd.DataFrame([{
            "column_1": col1, "column_2": col2,
            "chi2": round(chi2, 4), "p_value": round(p, 6),
            "dof": dof, "significant": "Yes" if p < 0.05 else "No",
        }])
        verdict = "significant association" if p < 0.05 else "no significant association"
        return HandlerResult(success=True, result_df=result,
                             summary=f"Chi2 test {col1} vs {col2}: chi2={chi2:.2f}, p={p:.4f} — {verdict}")

    # ── 15. T-test ────────────────────────────────────────────────────────────

    @staticmethod
    def handle_t_test(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Independent t-test: compare numeric column across two groups of a categorical column."""
        from scipy import stats as sp_stats

        col = params.get("column")
        group_col = params.get("group_column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if not col or col not in df.columns:
            col = num_cols[0] if num_cols else None
        if not group_col or group_col not in df.columns:
            group_col = next((c for c in cat_cols if df[c].nunique() == 2), cat_cols[0] if cat_cols else None)
        if not col or not group_col:
            return HandlerResult(success=False, error="Need a numeric column and a categorical group column")

        groups = df[group_col].dropna().unique()[:2]
        if len(groups) < 2:
            return HandlerResult(success=False, error=f"Column '{group_col}' needs at least 2 groups")

        g1 = df.loc[df[group_col] == groups[0], col].dropna()
        g2 = df.loc[df[group_col] == groups[1], col].dropna()
        stat, p = sp_stats.ttest_ind(g1, g2, equal_var=False)
        result = pd.DataFrame([{
            "numeric_col": col, "group_col": group_col,
            "group_1": str(groups[0]), "group_2": str(groups[1]),
            "mean_1": round(float(g1.mean()), 4), "mean_2": round(float(g2.mean()), 4),
            "t_stat": round(stat, 4), "p_value": round(p, 6),
            "significant": "Yes" if p < 0.05 else "No",
        }])
        return HandlerResult(success=True, result_df=result,
                             summary=f"T-test {col} by {group_col}: t={stat:.3f}, p={p:.4f}")

    # ── 16. ANOVA test ────────────────────────────────────────────────────────

    @staticmethod
    def handle_anova_test(df: pd.DataFrame, params: dict) -> HandlerResult:
        """One-way ANOVA: compare numeric column across multiple groups."""
        from scipy import stats as sp_stats

        col = params.get("column")
        group_col = params.get("group_column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if not col or col not in df.columns:
            col = num_cols[0] if num_cols else None
        if not group_col or group_col not in df.columns:
            group_col = cat_cols[0] if cat_cols else None
        if not col or not group_col:
            return HandlerResult(success=False, error="Need a numeric column and a categorical group column")

        groups = [g.dropna().values for _, g in df.groupby(group_col)[col] if len(g.dropna()) > 0]
        if len(groups) < 2:
            return HandlerResult(success=False, error="Need at least 2 groups for ANOVA")

        f_stat, p = sp_stats.f_oneway(*groups)
        result = pd.DataFrame([{
            "numeric_col": col, "group_col": group_col, "n_groups": len(groups),
            "f_stat": round(f_stat, 4), "p_value": round(p, 6),
            "significant": "Yes" if p < 0.05 else "No",
        }])
        return HandlerResult(success=True, result_df=result,
                             summary=f"ANOVA {col} by {group_col}: F={f_stat:.3f}, p={p:.4f} ({len(groups)} groups)")

    # ── 17. Mann-Whitney U test ───────────────────────────────────────────────

    @staticmethod
    def handle_mann_whitney(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Mann-Whitney U test (non-parametric alternative to t-test)."""
        from scipy import stats as sp_stats

        col = params.get("column")
        group_col = params.get("group_column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if not col or col not in df.columns:
            col = num_cols[0] if num_cols else None
        if not group_col or group_col not in df.columns:
            group_col = next((c for c in cat_cols if df[c].nunique() == 2), cat_cols[0] if cat_cols else None)
        if not col or not group_col:
            return HandlerResult(success=False, error="Need a numeric column and a categorical group column")

        groups = df[group_col].dropna().unique()[:2]
        if len(groups) < 2:
            return HandlerResult(success=False, error=f"Column '{group_col}' needs at least 2 groups")

        g1 = df.loc[df[group_col] == groups[0], col].dropna()
        g2 = df.loc[df[group_col] == groups[1], col].dropna()
        stat, p = sp_stats.mannwhitneyu(g1, g2, alternative="two-sided")
        result = pd.DataFrame([{
            "numeric_col": col, "group_col": group_col,
            "group_1": str(groups[0]), "group_2": str(groups[1]),
            "u_stat": round(float(stat), 4), "p_value": round(p, 6),
            "significant": "Yes" if p < 0.05 else "No",
        }])
        return HandlerResult(success=True, result_df=result,
                             summary=f"Mann-Whitney U {col} by {group_col}: U={stat:.1f}, p={p:.4f}")

    # ── 18. Kolmogorov-Smirnov test ───────────────────────────────────────────

    @staticmethod
    def handle_ks_test(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Kolmogorov-Smirnov normality test per numeric column."""
        from scipy import stats as sp_stats

        col = params.get("column")
        num_cols = [col] if col and col in df.columns else df.select_dtypes(include="number").columns.tolist()
        rows = []
        for c in num_cols:
            data = df[c].dropna()
            if len(data) < 3:
                continue
            normed = (data - data.mean()) / data.std() if data.std() > 0 else data
            stat, p = sp_stats.kstest(normed, "norm")
            rows.append({
                "column": c, "ks_stat": round(stat, 4), "p_value": round(p, 6),
                "normal": "Yes" if p > 0.05 else "No",
            })
        result = pd.DataFrame(rows)
        normal_n = sum(1 for r in rows if r["normal"] == "Yes")
        return HandlerResult(success=True, result_df=result,
                             summary=f"KS normality test: {normal_n}/{len(rows)} columns appear normal")

    # ── 19. Frequency table ───────────────────────────────────────────────────

    @staticmethod
    def handle_frequency_table(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Frequency table with cumulative percentage."""
        col = params.get("column")
        if not col or col not in df.columns:
            cats = df.select_dtypes(include=["object", "category"]).columns
            col = cats[0] if len(cats) > 0 else df.columns[0]
        vc = df[col].value_counts().reset_index()
        vc.columns = [col, "count"]
        vc["pct"] = (vc["count"] / vc["count"].sum() * 100).round(2)
        vc["cumulative_pct"] = vc["pct"].cumsum().round(2)
        return HandlerResult(success=True, result_df=vc,
                             summary=f"Frequency table for '{col}' ({len(vc)} unique values)")

    # ── 20. Coefficient of variation ──────────────────────────────────────────

    @staticmethod
    def handle_coefficient_variation(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Coefficient of variation (CV = std/mean) per numeric column."""
        num_cols = df.select_dtypes(include="number").columns
        rows = []
        for c in num_cols:
            mean = float(df[c].mean())
            std = float(df[c].std())
            cv = round(std / mean * 100, 2) if mean != 0 else float("inf")
            rows.append({"column": c, "mean": round(mean, 4), "std": round(std, 4), "cv_pct": cv})
        result = pd.DataFrame(rows).sort_values("cv_pct", ascending=False).reset_index(drop=True)
        return HandlerResult(success=True, result_df=result, summary="Coefficient of variation per column")

    # ── 21. Spearman rank correlation ─────────────────────────────────────────

    @staticmethod
    def handle_correlation_rank(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Spearman rank correlation matrix + heatmap."""
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return HandlerResult(success=False, error="Need at least 2 numeric columns")
        corr = df[num_cols].corr(method="spearman").round(4)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlBu_r", aspect="auto")
        _style(fig, title="Spearman Rank Correlation")
        return HandlerResult(success=True, result_df=corr.reset_index(),
                             charts_plotly=[fig.to_json()],
                             summary=f"Spearman correlation for {len(num_cols)} columns")

    # ── 22. Shannon entropy ───────────────────────────────────────────────────

    @staticmethod
    def handle_entropy_report(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Shannon entropy per column (higher = more diverse)."""
        from scipy import stats as sp_stats

        rows = []
        for c in df.columns:
            vc = df[c].dropna().value_counts(normalize=True)
            ent = round(float(sp_stats.entropy(vc, base=2)), 4)
            rows.append({"column": c, "entropy": ent, "unique": int(df[c].nunique()),
                         "dtype": str(df[c].dtype)})
        result = pd.DataFrame(rows).sort_values("entropy", ascending=False).reset_index(drop=True)
        return HandlerResult(success=True, result_df=result, summary="Shannon entropy per column")

    # ── 23. Gini impurity ─────────────────────────────────────────────────────

    @staticmethod
    def handle_gini_report(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Gini impurity per categorical column (0 = pure, 1 = max diversity)."""
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if not cat_cols:
            cat_cols = [c for c in df.columns if df[c].nunique() <= 20]
        rows = []
        for c in cat_cols:
            probs = df[c].value_counts(normalize=True).values
            gini = round(float(1 - np.sum(probs ** 2)), 4)
            rows.append({"column": c, "gini": gini, "unique": int(df[c].nunique())})
        result = pd.DataFrame(rows).sort_values("gini", ascending=False).reset_index(drop=True)
        return HandlerResult(success=True, result_df=result, summary="Gini impurity per categorical column")

    # ── 24. Missing pattern ───────────────────────────────────────────────────

    @staticmethod
    def handle_missing_pattern(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Analyze which columns tend to be missing together."""
        null_mask = df.isnull()
        null_cols = [c for c in df.columns if null_mask[c].any()]
        if not null_cols:
            return HandlerResult(success=True, result_df=pd.DataFrame([{"pattern": "No missing values"}]),
                                 summary="No missing values found")
        pattern = null_mask[null_cols].value_counts().head(10).reset_index()
        pattern.columns = list(null_cols) + ["count"]
        pattern["pct"] = (pattern["count"] / len(df) * 100).round(2)
        # Co-occurrence: which column pairs are missing together
        co = null_mask[null_cols].T.dot(null_mask[null_cols])
        return HandlerResult(success=True, result_df=pattern,
                             summary=f"Missing patterns across {len(null_cols)} columns (top {len(pattern)} patterns)")

    # ── 25. Quantile detail ───────────────────────────────────────────────────

    @staticmethod
    def handle_quantile_detail(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Detailed quantile report: 1,5,10,25,50,75,90,95,99 percentiles."""
        col = params.get("column")
        qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        num_cols = [col] if col and col in df.columns else df.select_dtypes(include="number").columns.tolist()
        result = df[num_cols].quantile(qs).round(4).T.reset_index()
        result.columns = ["column"] + [f"p{int(q*100)}" for q in qs]
        return HandlerResult(success=True, result_df=result,
                             summary=f"Detailed quantile report for {len(num_cols)} columns")

    # ── 26. Group stats ───────────────────────────────────────────────────────

    @staticmethod
    def handle_group_stats(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Descriptive stats per group (groupby + describe)."""
        col = params.get("column")
        value_col = params.get("value_column")
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not col or col not in df.columns:
            col = cat_cols[0] if cat_cols else None
        if not value_col or value_col not in df.columns:
            value_col = num_cols[0] if num_cols else None
        if not col or not value_col:
            return HandlerResult(success=False, error="Need a categorical group column and a numeric value column")

        grouped = df.groupby(col)[value_col].agg(["count", "mean", "std", "min", "median", "max"]).round(4)
        result = grouped.reset_index()
        return HandlerResult(success=True, result_df=result,
                             summary=f"Stats of '{value_col}' grouped by '{col}' ({len(result)} groups)")

    # ── 27. Column compare ────────────────────────────────────────────────────

    @staticmethod
    def handle_column_compare(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Compare two columns statistically."""
        cols = params.get("columns", [])
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col1 = cols[0] if len(cols) > 0 and cols[0] in df.columns else (num_cols[0] if len(num_cols) > 0 else None)
        col2 = cols[1] if len(cols) > 1 and cols[1] in df.columns else (num_cols[1] if len(num_cols) > 1 else None)
        if not col1 or not col2:
            return HandlerResult(success=False, error="Need 2 numeric columns to compare")

        corr_val = float(df[[col1, col2]].corr().iloc[0, 1])
        result = pd.DataFrame([{
            "stat": s,
            col1: round(float(getattr(df[col1], s)()), 4) if s != "count" else int(df[col1].count()),
            col2: round(float(getattr(df[col2], s)()), 4) if s != "count" else int(df[col2].count()),
        } for s in ["count", "mean", "std", "min", "median", "max"]] + [{
            "stat": "correlation", col1: round(corr_val, 4), col2: round(corr_val, 4),
        }])
        return HandlerResult(success=True, result_df=result,
                             summary=f"Compare {col1} vs {col2}: corr={corr_val:.4f}")

    # ── 28. Mutual info report ────────────────────────────────────────────────

    @staticmethod
    def handle_mutual_info_report(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Mutual information scores between all numeric features."""
        from sklearn.feature_selection import mutual_info_regression

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return HandlerResult(success=False, error="Need at least 2 numeric columns")

        target_col = params.get("column", num_cols[-1])
        if target_col not in num_cols:
            target_col = num_cols[-1]
        feature_cols = [c for c in num_cols if c != target_col]
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        mi = mutual_info_regression(X, y, random_state=42)
        result = pd.DataFrame({"feature": feature_cols, "mutual_info": np.round(mi, 4)})
        result = result.sort_values("mutual_info", ascending=False).reset_index(drop=True)
        return HandlerResult(success=True, result_df=result,
                             summary=f"Mutual information scores vs '{target_col}'")

    # ── 29. Summary extended ──────────────────────────────────────────────────

    @staticmethod
    def handle_summary_extended(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Extended summary: mean, median, mode, std, var, range, IQR, skew, kurtosis."""
        num_cols = df.select_dtypes(include="number").columns
        rows = []
        for c in num_cols:
            data = df[c].dropna()
            modes = data.mode()
            q1, q3 = float(data.quantile(0.25)), float(data.quantile(0.75))
            rows.append({
                "column": c,
                "mean": round(float(data.mean()), 4),
                "median": round(float(data.median()), 4),
                "mode": round(float(modes.iloc[0]), 4) if len(modes) > 0 else None,
                "std": round(float(data.std()), 4),
                "variance": round(float(data.var()), 4),
                "range": round(float(data.max() - data.min()), 4),
                "IQR": round(q3 - q1, 4),
                "skewness": round(float(data.skew()), 4),
                "kurtosis": round(float(data.kurtosis()), 4),
            })
        result = pd.DataFrame(rows)
        return HandlerResult(success=True, result_df=result,
                             summary=f"Extended summary for {len(num_cols)} numeric columns")

    # ── 30. Ratio report ──────────────────────────────────────────────────────

    @staticmethod
    def handle_ratio_report(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Compute key ratios between numeric columns."""
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return HandlerResult(success=False, error="Need at least 2 numeric columns")
        rows = []
        for i in range(min(len(num_cols), 10)):
            for j in range(i + 1, min(len(num_cols), 10)):
                a, b = num_cols[i], num_cols[j]
                mean_a, mean_b = float(df[a].mean()), float(df[b].mean())
                ratio = round(mean_a / mean_b, 4) if mean_b != 0 else float("inf")
                rows.append({"col_a": a, "col_b": b, "mean_a": round(mean_a, 4),
                             "mean_b": round(mean_b, 4), "ratio": ratio})
        result = pd.DataFrame(rows).sort_values("ratio", ascending=False).reset_index(drop=True)
        return HandlerResult(success=True, result_df=result,
                             summary=f"Mean ratios between {min(len(num_cols), 10)} numeric columns")

    # ── 31. Distribution fit ──────────────────────────────────────────────────

    @staticmethod
    def handle_distribution_fit(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Fit best distribution (normal/lognormal/exponential) to a numeric column."""
        from scipy import stats as sp_stats

        col = params.get("column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not col or col not in df.columns:
            col = num_cols[0] if num_cols else None
        if not col:
            return HandlerResult(success=False, error="Need a numeric column")

        data = df[col].dropna().values
        results = []
        for name, dist in [("normal", sp_stats.norm), ("lognormal", sp_stats.lognorm),
                           ("exponential", sp_stats.expon)]:
            try:
                params_fit = dist.fit(data)
                ks_stat, p = sp_stats.kstest(data, dist.cdf, args=params_fit)
                results.append({"distribution": name, "ks_stat": round(ks_stat, 4),
                                "p_value": round(p, 6)})
            except Exception:
                results.append({"distribution": name, "ks_stat": None, "p_value": None})

        result = pd.DataFrame(results).sort_values("p_value", ascending=False).reset_index(drop=True)
        best = result.iloc[0]["distribution"] if len(result) > 0 else "unknown"
        return HandlerResult(success=True, result_df=result,
                             summary=f"Best fit for '{col}': {best}")

    # ── 32. Stability report ──────────────────────────────────────────────────

    @staticmethod
    def handle_stability_report(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Check feature stability: split data in half, compare stats."""
        num_cols = df.select_dtypes(include="number").columns.tolist()
        mid = len(df) // 2
        first_half, second_half = df.iloc[:mid], df.iloc[mid:]
        rows = []
        for c in num_cols:
            m1, m2 = float(first_half[c].mean()), float(second_half[c].mean())
            s1, s2 = float(first_half[c].std()), float(second_half[c].std())
            mean_drift = abs(m1 - m2) / max(abs(m1), 1e-10)
            rows.append({
                "column": c, "mean_first": round(m1, 4), "mean_second": round(m2, 4),
                "std_first": round(s1, 4), "std_second": round(s2, 4),
                "mean_drift_pct": round(mean_drift * 100, 2),
                "stable": "Yes" if mean_drift < 0.1 else "No",
            })
        result = pd.DataFrame(rows).sort_values("mean_drift_pct", ascending=False).reset_index(drop=True)
        stable_n = sum(1 for r in rows if r["stable"] == "Yes")
        return HandlerResult(success=True, result_df=result,
                             summary=f"Stability: {stable_n}/{len(rows)} columns stable (drift < 10%)")

    # ── 33. Pairwise stats ────────────────────────────────────────────────────

    @staticmethod
    def handle_pairwise_stats(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Pairwise statistical comparison between numeric columns."""
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return HandlerResult(success=False, error="Need at least 2 numeric columns")
        rows = []
        for i in range(min(len(num_cols), 10)):
            for j in range(i + 1, min(len(num_cols), 10)):
                a, b = num_cols[i], num_cols[j]
                corr = float(df[[a, b]].corr().iloc[0, 1])
                diff = float(df[a].mean() - df[b].mean())
                rows.append({
                    "col_a": a, "col_b": b,
                    "pearson_corr": round(corr, 4),
                    "mean_diff": round(diff, 4),
                    "std_ratio": round(float(df[a].std()) / max(float(df[b].std()), 1e-10), 4),
                })
        result = pd.DataFrame(rows).sort_values("pearson_corr", key=abs, ascending=False).reset_index(drop=True)
        return HandlerResult(success=True, result_df=result,
                             summary=f"Pairwise stats for {min(len(num_cols), 10)} numeric columns")

    # ── 34. Time stats ────────────────────────────────────────────────────────

    @staticmethod
    def handle_time_stats(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Time-series specific stats (autocorrelation, stationarity hint)."""
        col = params.get("column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not col or col not in df.columns:
            col = num_cols[0] if num_cols else None
        if not col:
            return HandlerResult(success=False, error="Need a numeric column for time-series stats")

        data = df[col].dropna()
        rows = []
        for lag in [1, 2, 5, 10]:
            if lag < len(data):
                ac = round(float(data.autocorr(lag=lag)), 4)
                rows.append({"lag": lag, "autocorrelation": ac})

        # Simple stationarity hint: compare first/second half means
        mid = len(data) // 2
        m1, m2 = float(data.iloc[:mid].mean()), float(data.iloc[mid:].mean())
        drift = abs(m1 - m2) / max(abs(m1), 1e-10)
        stationary = "Likely stationary" if drift < 0.1 else "Non-stationary hint"

        result = pd.DataFrame(rows) if rows else pd.DataFrame([{"info": "Not enough data"}])
        return HandlerResult(success=True, result_df=result,
                             summary=f"Time stats for '{col}': {stationary}, first/second mean drift={drift:.2%}")

    # ── 35. Top/bottom values ─────────────────────────────────────────────────

    @staticmethod
    def handle_top_bottom_values(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Show top N and bottom N values of a column."""
        col = params.get("column")
        n = params.get("n", 5)
        if not col or col not in df.columns:
            num_cols = df.select_dtypes(include="number").columns.tolist()
            col = num_cols[0] if num_cols else df.columns[0]

        top = df.nlargest(n, col) if pd.api.types.is_numeric_dtype(df[col]) else df.head(n)
        bottom = df.nsmallest(n, col) if pd.api.types.is_numeric_dtype(df[col]) else df.tail(n)
        top = top.copy()
        bottom = bottom.copy()
        top["_group"] = "top"
        bottom["_group"] = "bottom"
        result = pd.concat([top, bottom], ignore_index=True)
        return HandlerResult(success=True, result_df=result,
                             summary=f"Top {n} and bottom {n} values by '{col}'")

    # ── 36. Memory report ─────────────────────────────────────────────────────

    @staticmethod
    def handle_memory_report(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Detailed memory usage per column."""
        mem = df.memory_usage(deep=True)
        total = mem.sum()
        rows = []
        for c in df.columns:
            m = int(mem[c])
            rows.append({
                "column": c, "dtype": str(df[c].dtype),
                "memory_bytes": m, "memory_kb": round(m / 1024, 2),
                "pct_of_total": round(m / total * 100, 2),
            })
        result = pd.DataFrame(rows).sort_values("memory_bytes", ascending=False).reset_index(drop=True)
        total_mb = round(total / 1024 ** 2, 2)
        return HandlerResult(success=True, result_df=result,
                             summary=f"Total memory: {total_mb} MB across {len(df.columns)} columns")

    # ── 37. Cluster tendency ──────────────────────────────────────────────────

    @staticmethod
    def handle_cluster_tendency(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Hopkins statistic to check if data is clusterable (< 0.5 = clusterable)."""
        from sklearn.neighbors import NearestNeighbors

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return HandlerResult(success=False, error="Need at least 2 numeric columns")

        X = df[num_cols].dropna()
        if len(X) < 10:
            return HandlerResult(success=False, error="Need at least 10 rows")

        n = min(len(X), 200)
        rng = np.random.RandomState(42)
        sample = X.sample(n, random_state=42).values
        # Uniform random points in feature space
        rand_pts = rng.uniform(X.min().values, X.max().values, size=(n, X.shape[1]))

        nn = NearestNeighbors(n_neighbors=2).fit(X.values)
        u_dist = nn.kneighbors(rand_pts, return_distance=True)[0][:, 1].sum()
        w_dist = nn.kneighbors(sample, return_distance=True)[0][:, 1].sum()
        h = round(float(u_dist / (u_dist + w_dist)), 4)

        label = "Clusterable" if h > 0.5 else "Uniform (not clusterable)"
        result = pd.DataFrame([{"hopkins_stat": h, "interpretation": label, "n_features": len(num_cols)}])
        return HandlerResult(success=True, result_df=result,
                             summary=f"Hopkins statistic: {h} — {label}")

    # ── 38. Sparsity report ───────────────────────────────────────────────────

    @staticmethod
    def handle_sparsity_report(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Sparsity analysis: zeros + nulls per column."""
        rows = []
        total_cells = len(df)
        for c in df.columns:
            null_n = int(df[c].isna().sum())
            zero_n = int((df[c] == 0).sum()) if pd.api.types.is_numeric_dtype(df[c]) else 0
            sparse_n = null_n + zero_n
            rows.append({
                "column": c, "null_count": null_n, "zero_count": zero_n,
                "sparse_count": sparse_n,
                "sparsity_pct": round(sparse_n / total_cells * 100, 2) if total_cells > 0 else 0,
            })
        result = pd.DataFrame(rows).sort_values("sparsity_pct", ascending=False).reset_index(drop=True)
        overall = sum(r["sparse_count"] for r in rows) / (total_cells * len(df.columns)) * 100 if total_cells > 0 else 0
        return HandlerResult(success=True, result_df=result,
                             summary=f"Overall sparsity: {overall:.1f}% (zeros + nulls)")

    # ── 39. Data sample ───────────────────────────────────────────────────────

    @staticmethod
    def handle_data_sample(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Random sample with basic stats summary."""
        n = params.get("n", 10)
        n = min(n, len(df))
        sample = df.sample(n, random_state=42)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        null_total = int(df.isna().sum().sum())
        return HandlerResult(success=True, result_df=sample,
                             summary=f"Random sample of {n} rows | "
                                     f"{df.shape[0]} total rows, {len(num_cols)} numeric cols, "
                                     f"{null_total} nulls")

    # ── 40. Normality comprehensive ───────────────────────────────────────────

    @staticmethod
    def handle_normality_comprehensive(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Comprehensive normality: Shapiro-Wilk + D'Agostino + Anderson-Darling."""
        from scipy import stats as sp_stats

        col = params.get("column")
        num_cols = [col] if col and col in df.columns else df.select_dtypes(include="number").columns.tolist()
        rows = []
        for c in num_cols:
            data = df[c].dropna()
            sample = data.sample(min(5000, len(data)), random_state=42) if len(data) > 5000 else data
            row: dict = {"column": c}
            # Shapiro
            try:
                sw_stat, sw_p = sp_stats.shapiro(sample)
                row["shapiro_stat"] = round(sw_stat, 4)
                row["shapiro_p"] = round(sw_p, 6)
            except Exception:
                row["shapiro_stat"] = None
                row["shapiro_p"] = None
            # D'Agostino
            try:
                da_stat, da_p = sp_stats.normaltest(sample)
                row["dagostino_stat"] = round(float(da_stat), 4)
                row["dagostino_p"] = round(float(da_p), 6)
            except Exception:
                row["dagostino_stat"] = None
                row["dagostino_p"] = None
            # Anderson-Darling
            try:
                ad = sp_stats.anderson(sample, dist="norm")
                row["anderson_stat"] = round(float(ad.statistic), 4)
                row["anderson_cv_5pct"] = round(float(ad.critical_values[2]), 4)
                row["anderson_normal"] = "Yes" if ad.statistic < ad.critical_values[2] else "No"
            except Exception:
                row["anderson_stat"] = None
                row["anderson_cv_5pct"] = None
                row["anderson_normal"] = "Error"

            # Consensus
            votes = sum([
                1 if row.get("shapiro_p") and row["shapiro_p"] > 0.05 else 0,
                1 if row.get("dagostino_p") and row["dagostino_p"] > 0.05 else 0,
                1 if row.get("anderson_normal") == "Yes" else 0,
            ])
            row["consensus"] = "Normal" if votes >= 2 else "Not normal"
            rows.append(row)

        result = pd.DataFrame(rows)
        normal_n = sum(1 for r in rows if r["consensus"] == "Normal")
        return HandlerResult(success=True, result_df=result,
                             summary=f"Comprehensive normality: {normal_n}/{len(rows)} columns normal (2/3 tests agree)")
