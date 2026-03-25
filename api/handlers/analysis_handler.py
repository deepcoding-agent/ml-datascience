"""Analysis handler — 50 smart, high-level analytical handlers.

These go beyond single-column stats: they reason about the data,
combine multiple operations, and return rich, insight-driven results
with charts and formatted summaries.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.viz_handler import _style
from api.logger import get_logger

log = get_logger(__name__)


class AnalysisHandler(BaseHandler):

    # ── 1. Compare extremes (max vs min row) ─────────────────────────────

    @staticmethod
    def handle_compare_extremes(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Compare the rows with the highest and lowest value of a column.
        Shows side-by-side comparison with all columns + a grouped bar chart."""
        col = params.get("column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not col or col not in num_cols:
            col = num_cols[0] if num_cols else None
        if col is None:
            return HandlerResult(success=False, error="No numeric column found for comparison")

        max_idx = df[col].idxmax()
        min_idx = df[col].idxmin()
        max_row = df.loc[max_idx]
        min_row = df.loc[min_idx]

        comp = pd.DataFrame({
            "Column": df.columns.tolist(),
            f"Highest {col}": [max_row[c] for c in df.columns],
            f"Lowest {col}": [min_row[c] for c in df.columns],
        })

        # Build difference column for numerics
        diffs: list[str] = []
        for c in df.columns:
            if c in num_cols:
                high = float(max_row[c]) if pd.notna(max_row[c]) else 0
                low = float(min_row[c]) if pd.notna(min_row[c]) else 0
                diff = high - low
                if low != 0:
                    pct = abs(diff / low) * 100
                    diffs.append(f"{diff:+,.2f} ({pct:.0f}%)")
                else:
                    diffs.append(f"{diff:+,.2f}")
            else:
                diffs.append("—")
        comp["Difference"] = diffs

        # Chart: grouped bar for numeric columns
        chart_cols = [c for c in num_cols if c != col][:8]
        if chart_cols:
            chart_data = pd.DataFrame({
                "Column": chart_cols * 2,
                "Value": [float(max_row[c]) if pd.notna(max_row[c]) else 0 for c in chart_cols]
                       + [float(min_row[c]) if pd.notna(min_row[c]) else 0 for c in chart_cols],
                "Type": [f"Highest {col}"] * len(chart_cols) + [f"Lowest {col}"] * len(chart_cols),
            })
            fig = px.bar(
                chart_data, x="Column", y="Value", color="Type", barmode="group",
                color_discrete_map={f"Highest {col}": "#FB8C3C", f"Lowest {col}": "#2EC4B6"},
                text="Value",
            )
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            _style(fig, title=f"Highest vs Lowest {col} — Side by Side")
            fig.update_layout(xaxis_title="Feature", yaxis_title="Value")
            charts = [fig.to_json()]
        else:
            charts = []

        # Rich summary
        summary_lines = [
            f"**Highest {col}**: {max_row[col]:,.2f}" if isinstance(max_row[col], (int, float)) else f"**Highest {col}**: {max_row[col]}",
            f"**Lowest {col}**: {min_row[col]:,.2f}" if isinstance(min_row[col], (int, float)) else f"**Lowest {col}**: {min_row[col]}",
        ]
        for c in num_cols:
            if c != col:
                high = max_row[c] if pd.notna(max_row[c]) else 0
                low = min_row[c] if pd.notna(min_row[c]) else 0
                summary_lines.append(f"  {c}: {high:,.2f} vs {low:,.2f}")

        return HandlerResult(
            success=True, result_df=comp, output_type="query",
            charts_plotly=charts,
            summary="\n".join(summary_lines),
        )

    # ── 2. Deep profile (single column) ──────────────────────────────────

    @staticmethod
    def handle_deep_profile(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Deep statistical profile of a single column: distribution, outliers,
        patterns, missing values, quartiles, and auto-visualization."""
        col = params.get("column")
        if not col or col not in df.columns:
            num = df.select_dtypes(include="number").columns.tolist()
            col = num[0] if num else df.columns[0]

        s = df[col].dropna()
        total = len(df[col])
        nulls = df[col].isnull().sum()

        profile: dict = {
            "column": col,
            "dtype": str(df[col].dtype),
            "total_rows": total,
            "non_null": len(s),
            "null_count": nulls,
            "null_pct": round(nulls / max(total, 1) * 100, 2),
            "unique_values": int(df[col].nunique()),
            "unique_pct": round(df[col].nunique() / max(total, 1) * 100, 2),
        }

        charts: list[str] = []

        if pd.api.types.is_numeric_dtype(df[col]):
            profile.update({
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4),
                "min": float(s.min()),
                "p25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "p75": float(s.quantile(0.75)),
                "max": float(s.max()),
                "skewness": round(float(s.skew()), 4),
                "kurtosis": round(float(s.kurt()), 4),
                "zeros": int((s == 0).sum()),
                "negatives": int((s < 0).sum()),
            })
            iqr = profile["p75"] - profile["p25"]
            lower = profile["p25"] - 1.5 * iqr
            upper = profile["p75"] + 1.5 * iqr
            outliers = s[(s < lower) | (s > upper)]
            profile["outlier_count"] = len(outliers)
            profile["outlier_pct"] = round(len(outliers) / max(len(s), 1) * 100, 2)
            profile["iqr"] = round(iqr, 4)

            # Distribution chart with box
            fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                                shared_xaxes=True, vertical_spacing=0.05)
            fig.add_trace(go.Histogram(x=s, nbinsx=30, marker_color="#FB8C3C", name="Distribution"), row=1, col=1)
            fig.add_trace(go.Box(x=s, marker_color="#2EC4B6", name="Box plot"), row=2, col=1)
            _style(fig, title=f"Deep Profile — {col} (n={len(s):,}, mean={profile['mean']:,.2f})")
            fig.update_layout(showlegend=False, height=450)
            charts.append(fig.to_json())
        else:
            top_values = s.value_counts().head(10)
            profile["top_10_values"] = {str(k): int(v) for k, v in top_values.items()}
            profile["most_common"] = str(top_values.index[0]) if len(top_values) > 0 else "N/A"
            profile["most_common_count"] = int(top_values.iloc[0]) if len(top_values) > 0 else 0

            fig = px.bar(x=top_values.index.astype(str), y=top_values.values, text=top_values.values)
            fig.update_traces(marker_color="#FB8C3C", textposition="outside")
            _style(fig, title=f"Deep Profile — {col} ({profile['unique_values']} unique values)")
            fig.update_layout(xaxis_title=col, yaxis_title="Count")
            charts.append(fig.to_json())

        result_df = pd.DataFrame([profile])

        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=charts,
            summary=f"Deep profile of '{col}': {profile.get('dtype')}, {profile.get('non_null')} values, "
                    f"{profile.get('null_pct')}% null, {profile.get('unique_values')} unique",
        )

    # ── 3. Group insights (compare groups statistically) ─────────────────

    @staticmethod
    def handle_group_insights(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Compare numeric statistics across groups of a categorical column.
        Shows mean/median/std per group + chart + identifies interesting differences."""
        group_col = params.get("column")
        value_col = params.get("value_column")

        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()

        if not group_col or group_col not in cat_cols:
            group_col = cat_cols[0] if cat_cols else None
        if group_col is None:
            return HandlerResult(success=False, error="No categorical column for grouping")

        if not value_col or value_col not in num_cols:
            value_col = num_cols[0] if num_cols else None
        if value_col is None:
            return HandlerResult(success=False, error="No numeric column for comparison")

        # Limit groups to top 15
        top_groups = df[group_col].value_counts().head(15).index
        subset = df[df[group_col].isin(top_groups)]

        stats = subset.groupby(group_col)[value_col].agg(["count", "mean", "median", "std", "min", "max"])
        stats = stats.round(2).reset_index()
        stats.columns = [group_col, "count", "mean", "median", "std", "min", "max"]
        stats = stats.sort_values("mean", ascending=False)

        # Find most interesting difference
        if len(stats) >= 2:
            best = stats.iloc[0]
            worst = stats.iloc[-1]
            ratio = best["mean"] / worst["mean"] if worst["mean"] != 0 else float("inf")
        else:
            ratio = 1.0

        # Chart: box plot by group
        fig = px.box(subset, x=group_col, y=value_col, color=group_col)
        _style(fig, title=f"{value_col} by {group_col} — Group Comparison (n={len(subset):,})")
        fig.update_layout(showlegend=False, xaxis_title=group_col, yaxis_title=value_col)

        summary = f"Compared {value_col} across {len(stats)} groups of {group_col}. "
        if ratio > 1.5 and len(stats) >= 2:
            summary += f"'{best[group_col]}' has {ratio:.1f}x higher avg than '{worst[group_col]}'. "
        summary += f"Overall range: {stats['mean'].min():,.2f} to {stats['mean'].max():,.2f} (mean)."

        return HandlerResult(
            success=True, result_df=stats, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=summary,
        )

    # ── 4. Anomaly detection ─────────────────────────────────────────────

    @staticmethod
    def handle_anomaly_detect(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Detect anomalies using IQR and Z-score methods across numeric columns.
        Returns flagged rows + anomaly summary + scatter chart."""
        col = params.get("column")
        method = params.get("method", "iqr")  # iqr | zscore
        num_cols = df.select_dtypes(include="number").columns.tolist()

        if col and col in num_cols:
            check_cols = [col]
        else:
            check_cols = num_cols[:5]

        if not check_cols:
            return HandlerResult(success=False, error="No numeric columns for anomaly detection")

        result = df.copy()
        anomaly_flags = pd.Series(False, index=df.index)
        anomaly_cols: dict[str, int] = {}

        for c in check_cols:
            s = result[c].dropna()
            if method == "zscore":
                z = (s - s.mean()) / s.std()
                mask = z.abs() > 3
            else:  # iqr
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)

            col_anomalies = mask.sum()
            if col_anomalies > 0:
                anomaly_cols[c] = int(col_anomalies)
            anomaly_flags |= mask.reindex(df.index, fill_value=False)

        result["_is_anomaly"] = anomaly_flags
        total_anomalies = int(anomaly_flags.sum())
        anomaly_rows = result[result["_is_anomaly"]]

        # Build summary table
        summary_rows = [{"column": c, "anomaly_count": cnt, "anomaly_pct": round(cnt / len(df) * 100, 2)}
                        for c, cnt in anomaly_cols.items()]
        summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame(columns=["column", "anomaly_count", "anomaly_pct"])

        # Scatter chart of first two anomaly columns
        charts: list[str] = []
        if len(check_cols) >= 2:
            fig = px.scatter(
                result, x=check_cols[0], y=check_cols[1],
                color="_is_anomaly",
                color_discrete_map={True: "#E71D36", False: "#86868B"},
            )
            _style(fig, title=f"Anomaly Detection ({method.upper()}) — {total_anomalies} anomalies in {len(df)} rows")
            fig.update_layout(xaxis_title=check_cols[0], yaxis_title=check_cols[1])
            charts.append(fig.to_json())

        return HandlerResult(
            success=True, result_df=summary_df, output_type="query",
            charts_plotly=charts,
            summary=f"Found {total_anomalies} anomalous rows ({total_anomalies/max(len(df),1)*100:.1f}%) using {method.upper()} method across {len(check_cols)} columns",
            metadata={"anomaly_details": anomaly_cols, "total_anomalies": total_anomalies},
        )

    # ── 5. Data quality report ───────────────────────────────────────────

    @staticmethod
    def handle_data_quality(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Comprehensive data quality assessment with scores per column
        and overall quality score. Checks: nulls, duplicates, outliers,
        cardinality, constant columns, mixed types."""
        rows: list[dict] = []
        total_score = 0

        for col in df.columns:
            s = df[col]
            n = len(s)
            null_pct = round(s.isnull().sum() / max(n, 1) * 100, 2)
            unique_pct = round(s.nunique() / max(n, 1) * 100, 2)

            issues: list[str] = []
            col_score = 100.0

            # Null penalty
            if null_pct > 0:
                col_score -= min(null_pct, 40)
                issues.append(f"{null_pct}% null")

            # Constant column
            if s.nunique() <= 1:
                col_score -= 20
                issues.append("constant")

            # High cardinality (potential ID column)
            if s.dtype == "object" and unique_pct > 90:
                col_score -= 10
                issues.append("high cardinality (possible ID)")

            # Outliers for numeric
            if pd.api.types.is_numeric_dtype(s):
                clean = s.dropna()
                if len(clean) > 4:
                    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                    iqr = q3 - q1
                    outlier_pct = ((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).mean() * 100
                    if outlier_pct > 5:
                        col_score -= min(outlier_pct, 15)
                        issues.append(f"{outlier_pct:.1f}% outliers")

            col_score = max(col_score, 0)
            total_score += col_score

            rows.append({
                "column": col,
                "dtype": str(s.dtype),
                "null_pct": null_pct,
                "unique_pct": unique_pct,
                "quality_score": round(col_score, 1),
                "issues": ", ".join(issues) if issues else "clean",
            })

        result_df = pd.DataFrame(rows)
        overall = round(total_score / max(len(df.columns), 1), 1)

        # Duplicate check
        dup_count = df.duplicated().sum()

        # Chart: quality scores
        fig = px.bar(
            result_df.sort_values("quality_score"),
            x="quality_score", y="column", orientation="h",
            color="quality_score",
            color_continuous_scale=["#E71D36", "#FF9F1C", "#2EC4B6"],
            text="quality_score",
        )
        fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        _style(fig, title=f"Data Quality Report — Overall Score: {overall}/100")
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="Quality Score", yaxis_title="Column",
            coloraxis_showscale=False,
        )

        quality_label = "Excellent" if overall >= 90 else "Good" if overall >= 75 else "Fair" if overall >= 50 else "Poor"
        problem_cols = [r for r in rows if r["quality_score"] < 70]

        summary = f"Overall quality: **{overall}/100 ({quality_label})**. "
        summary += f"{len(df):,} rows × {len(df.columns)} columns"
        if dup_count:
            summary += f", {dup_count} duplicates"
        summary += ". "
        if problem_cols:
            summary += f"Columns needing attention: {', '.join(r['column'] for r in problem_cols[:5])}."

        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=summary,
            metadata={"overall_score": overall, "duplicate_count": int(dup_count)},
        )

    # ── 6. Correlation insights ──────────────────────────────────────────

    @staticmethod
    def handle_correlation_insights(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Find and explain the most interesting correlations in the dataset.
        Returns top positive/negative correlations with scatter plots."""
        n = int(params.get("n", 10))
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return HandlerResult(success=False, error="Need at least 2 numeric columns")

        corr = df[num_cols].corr()

        # Extract top correlations (excluding self-correlation)
        pairs: list[tuple[str, str, float]] = []
        for i, c1 in enumerate(num_cols):
            for j, c2 in enumerate(num_cols):
                if i < j:
                    r = corr.loc[c1, c2]
                    if not np.isnan(r):
                        pairs.append((c1, c2, round(float(r), 4)))

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        top_pairs = pairs[:n]

        result_df = pd.DataFrame(top_pairs, columns=["column_1", "column_2", "correlation"])
        result_df["strength"] = result_df["correlation"].abs().apply(
            lambda r: "Very Strong" if r >= 0.8 else "Strong" if r >= 0.6
            else "Moderate" if r >= 0.4 else "Weak"
        )
        result_df["direction"] = result_df["correlation"].apply(lambda r: "Positive" if r > 0 else "Negative")

        # Charts: scatter of top 2 pairs
        charts: list[str] = []
        for c1, c2, r in top_pairs[:2]:
            fig = px.scatter(
                df, x=c1, y=c2, trendline="ols",
                opacity=0.5,
            )
            fig.update_traces(marker_color="#FB8C3C")
            _style(fig, title=f"{c1} vs {c2} (r={r:+.3f})")
            fig.update_layout(xaxis_title=c1, yaxis_title=c2)
            charts.append(fig.to_json())

        # Summary
        strong = [p for p in top_pairs if abs(p[2]) >= 0.6]
        summary = f"Analyzed {len(pairs)} column pairs. "
        if strong:
            summary += f"{len(strong)} strong correlations found. "
            top = strong[0]
            direction = "positively" if top[2] > 0 else "negatively"
            summary += f"Strongest: {top[0]} & {top[1]} (r={top[2]:+.3f}, {direction} correlated)."
        else:
            summary += "No strong correlations (|r| ≥ 0.6) found."

        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=charts,
            summary=summary,
        )

    # ── 7. Compare columns ───────────────────────────────────────────────

    @staticmethod
    def handle_compare_columns(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Side-by-side comparison of two columns: stats, distribution overlap, correlation."""
        columns = params.get("columns", [])
        if len(columns) < 2:
            num_cols = df.select_dtypes(include="number").columns.tolist()
            columns = num_cols[:2] if len(num_cols) >= 2 else []
        if len(columns) < 2:
            return HandlerResult(success=False, error="Need 2 columns to compare")

        c1, c2 = columns[0], columns[1]
        if c1 not in df.columns or c2 not in df.columns:
            return HandlerResult(success=False, error=f"Column(s) not found: {c1}, {c2}")

        s1, s2 = df[c1].dropna(), df[c2].dropna()

        comp: dict = {"metric": [], c1: [], c2: []}
        if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
            for metric, fn in [("count", "count"), ("mean", "mean"), ("std", "std"),
                               ("min", "min"), ("25%", lambda x: x.quantile(0.25)),
                               ("median", "median"), ("75%", lambda x: x.quantile(0.75)),
                               ("max", "max"), ("skew", "skew")]:
                comp["metric"].append(metric)
                v1 = getattr(s1, fn)() if isinstance(fn, str) else fn(s1)
                v2 = getattr(s2, fn)() if isinstance(fn, str) else fn(s2)
                comp[c1].append(round(float(v1), 4))
                comp[c2].append(round(float(v2), 4))

            corr_val = df[[c1, c2]].corr().iloc[0, 1]
            comp["metric"].append("correlation")
            comp[c1].append(round(float(corr_val), 4))
            comp[c2].append(round(float(corr_val), 4))

            # Overlapping histograms
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=s1, name=c1, opacity=0.6, marker_color="#FB8C3C"))
            fig.add_trace(go.Histogram(x=s2, name=c2, opacity=0.6, marker_color="#2EC4B6"))
            fig.update_layout(barmode="overlay")
            _style(fig, title=f"{c1} vs {c2} — Distribution Comparison")
            charts = [fig.to_json()]
        else:
            for metric in ["count", "unique", "most_common"]:
                comp["metric"].append(metric)
                if metric == "count":
                    comp[c1].append(len(s1))
                    comp[c2].append(len(s2))
                elif metric == "unique":
                    comp[c1].append(int(s1.nunique()))
                    comp[c2].append(int(s2.nunique()))
                else:
                    comp[c1].append(str(s1.mode().iloc[0]) if len(s1) > 0 else "N/A")
                    comp[c2].append(str(s2.mode().iloc[0]) if len(s2) > 0 else "N/A")
            charts = []

        result_df = pd.DataFrame(comp)
        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=charts,
            summary=f"Compared '{c1}' vs '{c2}' across {len(result_df)} metrics",
        )

    # ── 8. Trend detection ───────────────────────────────────────────────

    @staticmethod
    def handle_trend_detect(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Detect trends in numeric data: overall direction, rate of change,
        moving average, and turning points."""
        col = params.get("column")
        window = int(params.get("window", 5))
        num_cols = df.select_dtypes(include="number").columns.tolist()

        if not col or col not in num_cols:
            col = num_cols[0] if num_cols else None
        if col is None:
            return HandlerResult(success=False, error="No numeric column found")

        s = df[col].dropna().reset_index(drop=True)
        if len(s) < 5:
            return HandlerResult(success=False, error=f"Need at least 5 data points, got {len(s)}")

        # Linear trend
        x = np.arange(len(s))
        coeffs = np.polyfit(x, s.values, 1)
        slope, intercept = coeffs[0], coeffs[1]
        trend_line = slope * x + intercept

        # Moving average
        ma = s.rolling(window=window, min_periods=1).mean()

        # Trend direction
        if slope > 0:
            pct_change = (trend_line[-1] - trend_line[0]) / abs(trend_line[0]) * 100 if trend_line[0] != 0 else 0
            direction = f"Upward (+{pct_change:.1f}%)"
        elif slope < 0:
            pct_change = (trend_line[-1] - trend_line[0]) / abs(trend_line[0]) * 100 if trend_line[0] != 0 else 0
            direction = f"Downward ({pct_change:.1f}%)"
        else:
            direction = "Flat"

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=s.values, mode="lines", name=col, line=dict(color="#86868B", width=1), opacity=0.5))
        fig.add_trace(go.Scatter(y=ma.values, mode="lines", name=f"MA({window})", line=dict(color="#FB8C3C", width=2)))
        fig.add_trace(go.Scatter(y=trend_line, mode="lines", name="Trend", line=dict(color="#E71D36", width=2, dash="dash")))
        _style(fig, title=f"Trend Analysis — {col} ({direction})")
        fig.update_layout(xaxis_title="Index", yaxis_title=col)

        result_df = pd.DataFrame({
            "metric": ["direction", "slope", "start_value", "end_value", "min", "max", "volatility"],
            "value": [direction, round(slope, 4), round(float(s.iloc[0]), 2), round(float(s.iloc[-1]), 2),
                      round(float(s.min()), 2), round(float(s.max()), 2), round(float(s.std() / s.mean() * 100), 2) if s.mean() != 0 else 0],
        })

        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"Trend in '{col}': {direction}. Slope={slope:+.4f}, range [{s.min():,.2f}, {s.max():,.2f}]",
        )

    # ── 9. Segment analysis ──────────────────────────────────────────────

    @staticmethod
    def handle_segment_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Auto-segment data by a numeric column into quantile-based groups
        and describe each segment with average features."""
        col = params.get("column")
        n_segments = int(params.get("n", 4))
        num_cols = df.select_dtypes(include="number").columns.tolist()

        if not col or col not in num_cols:
            col = num_cols[0] if num_cols else None
        if col is None:
            return HandlerResult(success=False, error="No numeric column for segmentation")

        temp = df.copy()
        try:
            temp["_segment"] = pd.qcut(temp[col], q=n_segments, labels=[f"Q{i+1}" for i in range(n_segments)], duplicates="drop")
        except ValueError:
            temp["_segment"] = pd.cut(temp[col], bins=n_segments, labels=[f"Bin{i+1}" for i in range(n_segments)])

        # Stats per segment
        agg_cols = [c for c in num_cols if c != col][:6]
        agg_dict = {col: ["count", "mean", "min", "max"]}
        for c in agg_cols:
            agg_dict[c] = ["mean"]

        seg_stats = temp.groupby("_segment").agg(agg_dict).round(2)
        seg_stats.columns = ["_".join(c).strip("_") for c in seg_stats.columns]
        seg_stats = seg_stats.reset_index()

        # Chart
        fig = px.bar(
            seg_stats, x="_segment", y=f"{col}_count", text=f"{col}_count",
            color=f"{col}_mean", color_continuous_scale="YlOrRd",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        _style(fig, title=f"Segment Analysis — {col} ({n_segments} segments)")
        fig.update_layout(xaxis_title="Segment", yaxis_title="Count")

        return HandlerResult(
            success=True, result_df=seg_stats, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"Segmented {len(df):,} rows into {n_segments} groups by '{col}'",
        )

    # ── 10. Auto EDA (key findings) ──────────────────────────────────────

    @staticmethod
    def handle_auto_eda(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Automated exploratory data analysis — generates key findings
        about the dataset: shape, quality issues, distributions, correlations,
        and actionable recommendations."""
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        findings: list[str] = []
        recommendations: list[str] = []

        # 1. Shape
        findings.append(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns ({len(num_cols)} numeric, {len(cat_cols)} categorical)")

        # 2. Missing values
        null_summary = df.isnull().sum()
        null_cols = null_summary[null_summary > 0]
        if len(null_cols) > 0:
            worst = null_cols.idxmax()
            worst_pct = null_cols.max() / len(df) * 100
            findings.append(f"Missing values: {len(null_cols)} columns affected, worst is '{worst}' ({worst_pct:.1f}%)")
            if worst_pct > 50:
                recommendations.append(f"Consider dropping '{worst}' (>50% missing)")
            else:
                recommendations.append(f"Fill missing values in {len(null_cols)} columns")

        # 3. Duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            findings.append(f"Duplicates: {dup_count:,} duplicate rows ({dup_count/len(df)*100:.1f}%)")
            recommendations.append("Remove duplicate rows")

        # 4. Correlations
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            high_corr = []
            for i, c1 in enumerate(num_cols):
                for j, c2 in enumerate(num_cols):
                    if i < j and abs(corr.loc[c1, c2]) >= 0.7:
                        high_corr.append((c1, c2, round(float(corr.loc[c1, c2]), 3)))
            if high_corr:
                high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
                top = high_corr[0]
                findings.append(f"Strong correlations: {len(high_corr)} pairs, strongest: {top[0]} & {top[1]} (r={top[2]:+.3f})")

        # 5. Skewed columns
        if num_cols:
            skewed = []
            for c in num_cols:
                sk = df[c].skew()
                if abs(sk) > 1.5:
                    skewed.append((c, round(float(sk), 2)))
            if skewed:
                findings.append(f"Highly skewed: {', '.join(c for c, _ in skewed[:3])} — consider log/power transform")
                recommendations.append("Apply log_transform or power_transform to skewed columns")

        # 6. Constant columns
        constant = [c for c in df.columns if df[c].nunique() <= 1]
        if constant:
            findings.append(f"Constant columns: {', '.join(constant)} — no information, safe to drop")
            recommendations.append(f"Drop constant columns: {', '.join(constant)}")

        # 7. Categorical analysis
        for c in cat_cols[:3]:
            unique = df[c].nunique()
            if unique > 50:
                findings.append(f"'{c}' has high cardinality ({unique} unique) — may need encoding or grouping")
            elif unique == 2:
                findings.append(f"'{c}' is binary — good candidate for label encoding")

        # Build report table
        report_rows = [{"type": "Finding", "detail": f} for f in findings]
        report_rows += [{"type": "Recommendation", "detail": r} for r in recommendations]
        result_df = pd.DataFrame(report_rows)

        # Chart: null heatmap if nulls exist
        charts: list[str] = []
        if len(null_cols) > 0:
            null_pcts = (df.isnull().sum() / len(df) * 100).round(1)
            null_pcts = null_pcts[null_pcts > 0].sort_values(ascending=True)
            fig = px.bar(
                x=null_pcts.values, y=null_pcts.index, orientation="h",
                text=[f"{v:.1f}%" for v in null_pcts.values],
            )
            fig.update_traces(marker_color="#E71D36", textposition="outside")
            _style(fig, title=f"Missing Values — {len(null_cols)} columns affected")
            fig.update_layout(xaxis_title="Null %", yaxis_title="Column")
            charts.append(fig.to_json())

        summary = "\n".join(f"• {f}" for f in findings)
        if recommendations:
            summary += "\n\n**Recommendations:**\n" + "\n".join(f"→ {r}" for r in recommendations)

        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=charts,
            summary=summary,
        )

    # ── 11. K-Means clustering ───────────────────────────────────────────

    @staticmethod
    def handle_cluster_kmeans(df: pd.DataFrame, params: dict) -> HandlerResult:
        """K-Means clustering with auto k selection (silhouette) + 2D scatter."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return HandlerResult(success=False, error="Need at least 2 numeric columns for clustering")

        cols = num_cols[:10]
        X = df[cols].dropna()
        if len(X) < 10:
            return HandlerResult(success=False, error="Need at least 10 non-null rows for clustering")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        max_k = min(int(params.get("max_k", 8)), len(X) - 1, 10)
        min_k = 2
        best_k, best_score = 2, -1.0
        scores: list[dict] = []
        for k in range(min_k, max_k + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
            labels = km.fit_predict(X_scaled)
            s = silhouette_score(X_scaled, labels)
            scores.append({"k": k, "silhouette": round(s, 4)})
            if s > best_score:
                best_k, best_score = k, s

        km = KMeans(n_clusters=best_k, n_init=10, random_state=42, max_iter=300)
        labels = km.fit_predict(X_scaled)
        X = X.copy()
        X["cluster"] = labels

        fig = px.scatter(
            X, x=cols[0], y=cols[1], color="cluster",
            color_continuous_scale="Viridis",
        )
        fig.update_traces(marker_size=5)
        _style(fig, title=f"K-Means Clustering (k={best_k}, silhouette={best_score:.3f})")

        result_df = pd.DataFrame(scores)
        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"K-Means: best k={best_k} (silhouette={best_score:.3f}) on {len(X)} rows, {len(cols)} features",
        )

    # ── 12. PCA 2D projection ────────────────────────────────────────────

    @staticmethod
    def handle_pca_2d(df: pd.DataFrame, params: dict) -> HandlerResult:
        """PCA 2D projection with explained variance + scatter plot."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return HandlerResult(success=False, error="Need at least 2 numeric columns for PCA")

        cols = num_cols[:20]
        X = df[cols].dropna()
        if len(X) < 5:
            return HandlerResult(success=False, error="Need at least 5 non-null rows")

        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1]})
        var = pca.explained_variance_ratio_

        fig = px.scatter(pca_df, x="PC1", y="PC2", opacity=0.5)
        fig.update_traces(marker_color="#FB8C3C", marker_size=4)
        _style(fig, title=f"PCA 2D — Var explained: PC1={var[0]:.1%}, PC2={var[1]:.1%} (total={sum(var):.1%})")

        result_df = pd.DataFrame({
            "component": ["PC1", "PC2"],
            "explained_variance_ratio": [round(float(v), 4) for v in var],
            "cumulative": [round(float(var[0]), 4), round(float(sum(var)), 4)],
        })
        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"PCA 2D on {len(cols)} features: {sum(var):.1%} variance explained",
        )

    # ── 13. Isolation Forest anomaly detection ───────────────────────────

    @staticmethod
    def handle_outlier_isolation_forest(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Isolation Forest anomaly detection + scatter visualization."""
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        contamination = float(params.get("contamination", 0.05))
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return HandlerResult(success=False, error="Need at least 2 numeric columns")

        cols = num_cols[:10]
        X = df[cols].dropna()
        if len(X) < 10:
            return HandlerResult(success=False, error="Need at least 10 non-null rows")

        X_scaled = StandardScaler().fit_transform(X)
        iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        preds = iso.fit_predict(X_scaled)

        X_out = X.copy()
        X_out["anomaly"] = ["Anomaly" if p == -1 else "Normal" for p in preds]
        n_anomalies = int((preds == -1).sum())

        fig = px.scatter(
            X_out, x=cols[0], y=cols[1], color="anomaly",
            color_discrete_map={"Anomaly": "#E71D36", "Normal": "#86868B"},
        )
        fig.update_traces(marker_size=4)
        _style(fig, title=f"Isolation Forest — {n_anomalies} anomalies ({n_anomalies/len(X)*100:.1f}%)")

        result_df = pd.DataFrame({"label": ["Normal", "Anomaly"], "count": [len(X) - n_anomalies, n_anomalies]})
        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"Isolation Forest found {n_anomalies} anomalies ({n_anomalies/len(X)*100:.1f}%) in {len(X)} rows",
        )

    # ── 14. Automatic feature selection ──────────────────────────────────

    @staticmethod
    def handle_feature_selection_auto(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Auto feature selection: variance filter + correlation filter + mutual info."""
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        from sklearn.preprocessing import LabelEncoder

        target_col = params.get("column") or params.get("target")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return HandlerResult(success=False, error="Need at least 2 numeric columns")

        if not target_col or target_col not in df.columns:
            target_col = num_cols[-1]

        feature_cols = [c for c in num_cols if c != target_col]
        clean = df[feature_cols + [target_col]].dropna()
        if len(clean) < 10:
            return HandlerResult(success=False, error="Need at least 10 non-null rows")

        X = clean[feature_cols]
        y = clean[target_col]

        rows: list[dict] = []
        for c in feature_cols:
            var = float(X[c].var())
            corr_val = float(X[c].corr(y)) if pd.api.types.is_numeric_dtype(y) else 0.0
            rows.append({"feature": c, "variance": round(var, 4), "abs_corr_target": round(abs(corr_val), 4)})

        try:
            if y.nunique() <= 20:
                le = LabelEncoder()
                y_enc = le.fit_transform(y.astype(str))
                mi = mutual_info_classif(X, y_enc, random_state=42)
            else:
                mi = mutual_info_regression(X, y, random_state=42)
            for i, c in enumerate(feature_cols):
                rows[i]["mutual_info"] = round(float(mi[i]), 4)
        except Exception:
            for r in rows:
                r["mutual_info"] = 0.0

        result_df = pd.DataFrame(rows).sort_values("mutual_info", ascending=False)
        result_df["rank"] = range(1, len(result_df) + 1)

        fig = px.bar(
            result_df.head(15), x="mutual_info", y="feature", orientation="h",
            text="mutual_info", color="abs_corr_target", color_continuous_scale="YlOrRd",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        _style(fig, title=f"Feature Importance (target={target_col})")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})

        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"Ranked {len(feature_cols)} features by importance for '{target_col}'. Top: {result_df.iloc[0]['feature']}",
        )

    # ── 15. Distribution analysis ────────────────────────────────────────

    @staticmethod
    def handle_distribution_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Analyze distribution shape (skew, kurtosis, normality) per numeric column."""
        from scipy import stats as sp_stats

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return HandlerResult(success=False, error="No numeric columns found")

        rows: list[dict] = []
        for c in num_cols:
            s = df[c].dropna()
            if len(s) < 8:
                continue
            sk = float(s.skew())
            ku = float(s.kurt())
            try:
                _, p_shapiro = sp_stats.shapiro(s.sample(min(len(s), 5000), random_state=42))
            except Exception:
                p_shapiro = 0.0

            shape = "Normal" if abs(sk) < 0.5 and abs(ku) < 1 else "Skewed" if abs(sk) > 1 else "Heavy-tailed" if ku > 3 else "Moderate"
            rows.append({
                "column": c, "skewness": round(sk, 3), "kurtosis": round(ku, 3),
                "shapiro_p": round(float(p_shapiro), 4), "normal": p_shapiro > 0.05, "shape": shape,
            })

        if not rows:
            return HandlerResult(success=False, error="No columns with enough data for distribution analysis")

        result_df = pd.DataFrame(rows)
        non_normal = result_df[~result_df["normal"]]

        fig = px.bar(result_df, x="column", y="skewness", color="shape", text="skewness")
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        _style(fig, title=f"Distribution Shape — {len(result_df)} numeric columns")

        most_skewed = result_df.loc[result_df["skewness"].abs().idxmax(), "column"]
        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"{len(non_normal)}/{len(result_df)} columns are non-normal (Shapiro p<0.05). Most skewed: {most_skewed}",
        )

    # ── 16. Missing value analysis ───────────────────────────────────────

    @staticmethod
    def handle_missing_value_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Deep missing value pattern analysis: co-occurrence, MCAR hint, heatmap."""
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        if len(null_cols) == 0:
            return HandlerResult(
                success=True, result_df=pd.DataFrame({"status": ["No missing values"]}),
                output_type="query", summary="No missing values found in any column.",
            )

        rows: list[dict] = []
        for c in null_cols.index:
            n_miss = int(null_cols[c])
            pct = round(n_miss / len(df) * 100, 2)
            pattern = "Random" if pct < 5 else "Moderate" if pct < 30 else "Systematic"
            rows.append({"column": c, "missing": n_miss, "pct": pct, "pattern": pattern})

        result_df = pd.DataFrame(rows).sort_values("pct", ascending=False)

        # Co-occurrence: which columns tend to be missing together
        co_occ: list[str] = []
        null_matrix = df[null_cols.index].isnull()
        if len(null_cols) >= 2:
            corr_null = null_matrix.corr()
            for i, c1 in enumerate(null_cols.index):
                for j, c2 in enumerate(null_cols.index):
                    if i < j and corr_null.loc[c1, c2] > 0.5:
                        co_occ.append(f"{c1} & {c2} (r={corr_null.loc[c1, c2]:.2f})")

        total_null = int(df.isnull().sum().sum())
        total_cells = len(df) * len(df.columns)

        fig = px.bar(result_df, x="pct", y="column", orientation="h", color="pattern",
                     text="pct", color_discrete_map={"Random": "#2EC4B6", "Moderate": "#FF9F1C", "Systematic": "#E71D36"})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        _style(fig, title=f"Missing Value Analysis — {len(null_cols)} columns, {total_null:,} cells")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})

        summary = f"{len(null_cols)} columns with missing values ({total_null:,}/{total_cells:,} cells = {total_null/total_cells*100:.1f}%)."
        if co_occ:
            summary += f" Co-occurring: {', '.join(co_occ[:3])}."

        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=[fig.to_json()], summary=summary,
        )

    # ── 17. Categorical analysis ─────────────────────────────────────────

    @staticmethod
    def handle_categorical_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Deep analysis of all categorical columns: cardinality, mode, entropy."""
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if not cat_cols:
            return HandlerResult(success=False, error="No categorical columns found")

        rows: list[dict] = []
        for c in cat_cols:
            s = df[c].dropna()
            n_unique = int(s.nunique())
            mode_val = str(s.mode().iloc[0]) if len(s) > 0 else "N/A"
            mode_freq = int(s.value_counts().iloc[0]) if len(s) > 0 else 0
            mode_pct = round(mode_freq / max(len(s), 1) * 100, 1)

            probs = s.value_counts(normalize=True)
            entropy = round(float(-(probs * np.log2(probs.clip(lower=1e-10))).sum()), 3)

            card_type = "Binary" if n_unique == 2 else "Low" if n_unique <= 10 else "Medium" if n_unique <= 50 else "High"
            rows.append({
                "column": c, "unique": n_unique, "cardinality": card_type,
                "mode": mode_val, "mode_pct": mode_pct, "entropy": entropy,
                "null_pct": round(df[c].isnull().sum() / len(df) * 100, 1),
            })

        result_df = pd.DataFrame(rows)

        fig = px.bar(result_df, x="unique", y="column", orientation="h",
                     color="cardinality", text="unique",
                     color_discrete_map={"Binary": "#2EC4B6", "Low": "#FB8C3C", "Medium": "#FF9F1C", "High": "#E71D36"})
        fig.update_traces(textposition="outside")
        _style(fig, title=f"Categorical Column Analysis — {len(cat_cols)} columns")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})

        highest_card = result_df.loc[result_df["unique"].idxmax()]
        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"Analyzed {len(cat_cols)} categorical columns. Highest cardinality: {highest_card['column']} ({highest_card['unique']} unique)",
        )

    # ── 18. Numeric summary ──────────────────────────────────────────────

    @staticmethod
    def handle_numeric_summary(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Comprehensive numeric columns summary in one table."""
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return HandlerResult(success=False, error="No numeric columns found")

        rows: list[dict] = []
        for c in num_cols:
            s = df[c].dropna()
            rows.append({
                "column": c, "count": len(s),
                "null_pct": round(df[c].isnull().sum() / len(df) * 100, 1),
                "mean": round(float(s.mean()), 3), "std": round(float(s.std()), 3),
                "min": round(float(s.min()), 3), "p25": round(float(s.quantile(0.25)), 3),
                "median": round(float(s.median()), 3), "p75": round(float(s.quantile(0.75)), 3),
                "max": round(float(s.max()), 3), "skew": round(float(s.skew()), 3),
                "kurtosis": round(float(s.kurt()), 3),
                "zeros": int((s == 0).sum()), "negatives": int((s < 0).sum()),
            })

        result_df = pd.DataFrame(rows)

        heat_cols = ["mean", "std", "skew", "kurtosis"]
        heat_data = result_df.set_index("column")[heat_cols]
        fig = px.imshow(heat_data.T, text_auto=".2f", aspect="auto", color_continuous_scale="YlOrRd")
        _style(fig, title=f"Numeric Summary Heatmap — {len(num_cols)} columns")

        n_with_nulls = sum(1 for r in rows if r["null_pct"] > 0)
        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"Summarized {len(num_cols)} numeric columns. {n_with_nulls} have nulls.",
        )

    # ── 19. Hypothesis test (auto) ───────────────────────────────────────

    @staticmethod
    def handle_hypothesis_test(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Auto-choose t-test or Mann-Whitney based on normality."""
        from scipy import stats as sp_stats

        col = params.get("column")
        group_col = params.get("group_column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if not col or col not in num_cols:
            col = num_cols[0] if num_cols else None
        if col is None:
            return HandlerResult(success=False, error="No numeric column found")

        if not group_col or group_col not in cat_cols:
            for c in cat_cols:
                if df[c].nunique() == 2:
                    group_col = c
                    break
            if not group_col:
                group_col = cat_cols[0] if cat_cols else None
        if group_col is None:
            return HandlerResult(success=False, error="No categorical column found for grouping")

        groups = df[group_col].dropna().unique()[:2]
        if len(groups) < 2:
            return HandlerResult(success=False, error=f"Need at least 2 groups in '{group_col}', found {len(groups)}")

        g1 = df[df[group_col] == groups[0]][col].dropna()
        g2 = df[df[group_col] == groups[1]][col].dropna()

        normal_1 = sp_stats.shapiro(g1.sample(min(len(g1), 5000), random_state=42))[1] > 0.05 if len(g1) >= 8 else False
        normal_2 = sp_stats.shapiro(g2.sample(min(len(g2), 5000), random_state=42))[1] > 0.05 if len(g2) >= 8 else False

        if normal_1 and normal_2:
            stat, p = sp_stats.ttest_ind(g1, g2)
            test_name = "Independent t-test"
        else:
            stat, p = sp_stats.mannwhitneyu(g1, g2, alternative="two-sided")
            test_name = "Mann-Whitney U"

        effect_size = abs(g1.mean() - g2.mean()) / max(float(pd.concat([g1, g2]).std()), 1e-10)
        sig = "Significant" if p < 0.05 else "Not significant"

        result_df = pd.DataFrame({
            "metric": ["test", "statistic", "p_value", "significant", "effect_size_d",
                       f"mean_{groups[0]}", f"mean_{groups[1]}", "n_group_1", "n_group_2"],
            "value": [test_name, round(float(stat), 4), round(float(p), 6), sig,
                      round(float(effect_size), 3), round(float(g1.mean()), 3),
                      round(float(g2.mean()), 3), len(g1), len(g2)],
        })

        fig = px.box(df[df[group_col].isin(groups)], x=group_col, y=col, color=group_col)
        _style(fig, title=f"{test_name}: {col} by {group_col} (p={float(p):.4f}, {sig})")

        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"{test_name}: {sig} (p={float(p):.4f}). {groups[0]} mean={g1.mean():.3f} vs {groups[1]} mean={g2.mean():.3f}, Cohen's d={effect_size:.3f}",
        )

    # ── 20. Quick OLS regression ─────────────────────────────────────────

    @staticmethod
    def handle_regression_quick(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Quick OLS linear regression + scatter + R-squared + coefficients."""
        from sklearn.linear_model import LinearRegression

        target = params.get("column") or params.get("target")
        feature = params.get("feature")
        num_cols = df.select_dtypes(include="number").columns.tolist()

        if not target or target not in num_cols:
            target = num_cols[-1] if num_cols else None
        if target is None:
            return HandlerResult(success=False, error="No numeric target column found")

        features = [c for c in num_cols if c != target]
        if not features:
            return HandlerResult(success=False, error="Need at least 1 feature column")

        clean = df[features + [target]].dropna()
        if len(clean) < 5:
            return HandlerResult(success=False, error="Need at least 5 non-null rows")

        X = clean[features].values
        y = clean[target].values
        model = LinearRegression().fit(X, y)
        r2 = round(float(model.score(X, y)), 4)
        y_pred = model.predict(X)

        coef_df = pd.DataFrame({"feature": features, "coefficient": [round(float(c), 4) for c in model.coef_]})
        coef_df["abs_coeff"] = coef_df["coefficient"].abs()
        coef_df = coef_df.sort_values("abs_coeff", ascending=False).drop(columns=["abs_coeff"])
        coef_df.loc[len(coef_df)] = {"feature": "(intercept)", "coefficient": round(float(model.intercept_), 4)}

        fig = px.scatter(x=y, y=y_pred, opacity=0.5, labels={"x": "Actual", "y": "Predicted"})
        fig.update_traces(marker_color="#FB8C3C", marker_size=4)
        fig.add_trace(go.Scatter(x=[float(y.min()), float(y.max())], y=[float(y.min()), float(y.max())],
                                 mode="lines", line=dict(dash="dash", color="#E71D36"), name="Perfect fit"))
        _style(fig, title=f"OLS Regression — {target} (R\u00b2={r2})")

        return HandlerResult(
            success=True, result_df=coef_df, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"OLS on {len(features)} features \u2192 R\u00b2={r2}. Top predictor: {coef_df.iloc[0]['feature']} (coeff={coef_df.iloc[0]['coefficient']:.4f})",
        )

    # ── 21. Top N analysis ───────────────────────────────────────────────

    @staticmethod
    def handle_top_n_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Analyze top N rows by a metric with details."""
        col = params.get("column")
        n = int(params.get("n", 10))
        num_cols = df.select_dtypes(include="number").columns.tolist()

        if not col or col not in num_cols:
            col = num_cols[0] if num_cols else None
        if col is None:
            return HandlerResult(success=False, error="No numeric column found")

        top = df.nlargest(n, col)
        overall_mean = float(df[col].mean())
        top_mean = float(top[col].mean())

        fig = px.bar(top.reset_index(drop=True), y=col, text=col)
        fig.update_traces(marker_color="#FB8C3C", texttemplate="%{text:,.2f}", textposition="outside")
        _style(fig, title=f"Top {n} by {col}")

        ratio = top_mean / overall_mean if overall_mean != 0 else 0
        return HandlerResult(
            success=True, result_df=top, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"Top {n} by '{col}': range [{top[col].min():,.2f}, {top[col].max():,.2f}], mean={top_mean:,.2f} ({ratio:.1f}x overall avg)",
        )

    # ── 22. Bottom N analysis ────────────────────────────────────────────

    @staticmethod
    def handle_bottom_n_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Analyze bottom N rows by a metric."""
        col = params.get("column")
        n = int(params.get("n", 10))
        num_cols = df.select_dtypes(include="number").columns.tolist()

        if not col or col not in num_cols:
            col = num_cols[0] if num_cols else None
        if col is None:
            return HandlerResult(success=False, error="No numeric column found")

        bottom = df.nsmallest(n, col)
        overall_mean = float(df[col].mean())

        fig = px.bar(bottom.reset_index(drop=True), y=col, text=col)
        fig.update_traces(marker_color="#2EC4B6", texttemplate="%{text:,.2f}", textposition="outside")
        _style(fig, title=f"Bottom {n} by {col}")

        return HandlerResult(
            success=True, result_df=bottom, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"Bottom {n} by '{col}': range [{bottom[col].min():,.2f}, {bottom[col].max():,.2f}], mean={bottom[col].mean():,.2f} (overall avg={overall_mean:,.2f})",
        )

    # ── 23. Percentile analysis ──────────────────────────────────────────

    @staticmethod
    def handle_percentile_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Compare stats across percentile bands (Q1-Q4)."""
        col = params.get("column")
        num_cols = df.select_dtypes(include="number").columns.tolist()

        if not col or col not in num_cols:
            col = num_cols[0] if num_cols else None
        if col is None:
            return HandlerResult(success=False, error="No numeric column found")

        temp = df[[col]].dropna().copy()
        try:
            temp["quartile"] = pd.qcut(temp[col], q=4, labels=["Q1 (0-25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (75-100%)"], duplicates="drop")
        except ValueError:
            return HandlerResult(success=False, error=f"Cannot create quartiles for '{col}' — too few unique values")

        stats = temp.groupby("quartile")[col].agg(["count", "mean", "min", "max", "std"]).round(3).reset_index()

        fig = px.box(temp, x="quartile", y=col, color="quartile")
        _style(fig, title=f"Percentile Analysis — {col}")
        fig.update_layout(showlegend=False)

        return HandlerResult(
            success=True, result_df=stats, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"Percentile breakdown for '{col}': Q1 mean={stats.iloc[0]['mean']:.3f}, Q4 mean={stats.iloc[-1]['mean']:.3f}",
        )

    # ── 24. Variance analysis ────────────────────────────────────────────

    @staticmethod
    def handle_variance_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Variance contribution per feature (% of total variance)."""
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return HandlerResult(success=False, error="No numeric columns found")

        variances: dict[str, float] = {}
        for c in num_cols:
            v = df[c].var(skipna=True)
            if pd.notna(v):
                variances[c] = float(v)

        total_var = sum(variances.values())
        rows = [{"feature": k, "variance": round(v, 4), "pct_of_total": round(v / max(total_var, 1e-10) * 100, 2)}
                for k, v in sorted(variances.items(), key=lambda x: x[1], reverse=True)]

        result_df = pd.DataFrame(rows)

        fig = px.bar(result_df.head(15), x="pct_of_total", y="feature", orientation="h",
                     text="pct_of_total", color="pct_of_total", color_continuous_scale="YlOrRd")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        _style(fig, title=f"Variance Contribution — {len(num_cols)} features")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)

        top3_pct = sum(r["pct_of_total"] for r in rows[:3])
        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"Top variance contributor: {rows[0]['feature']} ({rows[0]['pct_of_total']:.1f}%). Top 3 account for {top3_pct:.1f}%",
        )

    # ── 25. Change point detection ───────────────────────────────────────

    @staticmethod
    def handle_change_point_detect(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Detect change points in a numeric series using sliding window mean diff."""
        col = params.get("column")
        window = int(params.get("window", 10))
        num_cols = df.select_dtypes(include="number").columns.tolist()

        if not col or col not in num_cols:
            col = num_cols[0] if num_cols else None
        if col is None:
            return HandlerResult(success=False, error="No numeric column found")

        s = df[col].dropna().reset_index(drop=True)
        if len(s) < window * 2:
            return HandlerResult(success=False, error=f"Need at least {window * 2} data points, got {len(s)}")

        diffs: list[float] = []
        for i in range(window, len(s) - window):
            left_mean = float(s.iloc[i - window:i].mean())
            right_mean = float(s.iloc[i:i + window].mean())
            diffs.append(abs(right_mean - left_mean))

        diffs_s = pd.Series(diffs)
        threshold = float(diffs_s.mean() + 2 * diffs_s.std())
        change_points = [i + window for i, d in enumerate(diffs) if d > threshold]

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=s.values, mode="lines", name=col, line=dict(color="#86868B", width=1)))
        for cp in change_points[:10]:
            fig.add_vline(x=cp, line_dash="dash", line_color="#E71D36", annotation_text=f"CP@{cp}")
        _style(fig, title=f"Change Point Detection — {col} ({len(change_points)} points)")

        cp_limited = change_points[:20]
        result_df = pd.DataFrame({
            "change_point_index": cp_limited,
            "value": [round(float(s.iloc[cp]), 3) for cp in cp_limited],
        })
        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"Found {len(change_points)} change points in '{col}' (window={window}, threshold=mean+2*std)",
        )

    # ── 26. Target analysis ──────────────────────────────────────────────

    @staticmethod
    def handle_target_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Analyze relationship between each feature and a target column."""
        target = params.get("column") or params.get("target")
        num_cols = df.select_dtypes(include="number").columns.tolist()

        if not target or target not in num_cols:
            target = num_cols[-1] if num_cols else None
        if target is None:
            return HandlerResult(success=False, error="No numeric target column found")

        features = [c for c in num_cols if c != target]
        if not features:
            return HandlerResult(success=False, error="No feature columns found besides target")

        rows: list[dict] = []
        for f in features:
            clean = df[[f, target]].dropna()
            if len(clean) < 5:
                continue
            corr_val = float(clean[f].corr(clean[target]))
            rows.append({"feature": f, "correlation": round(corr_val, 4),
                         "abs_corr": round(abs(corr_val), 4),
                         "direction": "Positive" if corr_val > 0 else "Negative"})

        if not rows:
            return HandlerResult(success=False, error=f"No valid features for target '{target}'")

        result_df = pd.DataFrame(rows).sort_values("abs_corr", ascending=False)

        fig = px.bar(result_df.head(15), x="correlation", y="feature", orientation="h",
                     color="correlation", color_continuous_scale="RdBu_r", text="correlation")
        fig.update_traces(texttemplate="%{text:+.3f}", textposition="outside")
        _style(fig, title=f"Feature-Target Correlation (target={target})")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})

        top_feat = result_df.iloc[0]["feature"]
        top_corr = result_df.iloc[0]["correlation"]
        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"Target '{target}': strongest predictor is '{top_feat}' (r={top_corr:+.3f})",
        )

    # ── 27. Multicollinearity check (VIF) ────────────────────────────────

    @staticmethod
    def handle_multicollinearity_check(df: pd.DataFrame, params: dict) -> HandlerResult:
        """VIF-based multicollinearity detection."""
        from sklearn.linear_model import LinearRegression

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return HandlerResult(success=False, error="Need at least 2 numeric columns")

        cols = num_cols[:15]
        clean = df[cols].dropna()
        if len(clean) < 10:
            return HandlerResult(success=False, error="Need at least 10 non-null rows")

        rows: list[dict] = []
        X = clean[cols].values
        for i, c in enumerate(cols):
            y_i = X[:, i]
            X_i = np.delete(X, i, axis=1)
            if X_i.shape[1] == 0:
                continue
            r2 = float(LinearRegression().fit(X_i, y_i).score(X_i, y_i))
            vif = 1.0 / (1.0 - r2) if r2 < 1 else float("inf")
            concern = "High" if vif > 10 else "Moderate" if vif > 5 else "Low"
            rows.append({"feature": c, "VIF": round(vif, 2), "R2_other": round(r2, 4), "concern": concern})

        result_df = pd.DataFrame(rows).sort_values("VIF", ascending=False)

        fig = px.bar(result_df, x="VIF", y="feature", orientation="h", color="concern",
                     text="VIF", color_discrete_map={"Low": "#2EC4B6", "Moderate": "#FF9F1C", "High": "#E71D36"})
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        _style(fig, title=f"Multicollinearity Check (VIF) — {len(cols)} features")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})

        high_vif = [r for r in rows if r["VIF"] > 10]
        worst = rows[0] if rows else {"feature": "N/A", "VIF": 0}
        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"VIF analysis: {len(high_vif)} features with high multicollinearity (VIF>10). Worst: {worst['feature']} (VIF={worst['VIF']:.1f})",
        )

    # ── 28. A/B test ─────────────────────────────────────────────────────

    @staticmethod
    def handle_ab_test(df: pd.DataFrame, params: dict) -> HandlerResult:
        """A/B test with significance: p-value, effect size, confidence interval."""
        from scipy import stats as sp_stats

        metric_col = params.get("column") or params.get("metric")
        group_col = params.get("group_column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if not metric_col or metric_col not in num_cols:
            metric_col = num_cols[0] if num_cols else None
        if metric_col is None:
            return HandlerResult(success=False, error="No numeric metric column found")

        if not group_col or group_col not in cat_cols:
            for c in cat_cols:
                if df[c].nunique() == 2:
                    group_col = c
                    break
            if not group_col:
                group_col = cat_cols[0] if cat_cols else None
        if group_col is None:
            return HandlerResult(success=False, error="No categorical group column found")

        groups = df[group_col].dropna().unique()[:2]
        if len(groups) < 2:
            return HandlerResult(success=False, error="Need exactly 2 groups for A/B test")

        a = df[df[group_col] == groups[0]][metric_col].dropna()
        b = df[df[group_col] == groups[1]][metric_col].dropna()

        stat, p = sp_stats.ttest_ind(a, b)
        pooled_std = float(np.sqrt((a.var() * (len(a) - 1) + b.var() * (len(b) - 1)) / (len(a) + len(b) - 2)))
        effect_d = abs(float(a.mean()) - float(b.mean())) / max(pooled_std, 1e-10)
        se_diff = float(np.sqrt(a.var() / len(a) + b.var() / len(b)))
        ci_low = float(a.mean() - b.mean()) - 1.96 * se_diff
        ci_high = float(a.mean() - b.mean()) + 1.96 * se_diff

        winner = str(groups[0]) if float(a.mean()) > float(b.mean()) else str(groups[1])
        sig = "Significant" if p < 0.05 else "Not significant"

        result_df = pd.DataFrame({
            "metric": ["test", "p_value", "significant", "effect_size_d", f"mean_{groups[0]}", f"mean_{groups[1]}",
                       "diff", "ci_lower", "ci_upper", "winner", "n_A", "n_B"],
            "value": ["Welch t-test", round(float(p), 6), sig, round(effect_d, 3),
                      round(float(a.mean()), 3), round(float(b.mean()), 3),
                      round(float(a.mean() - b.mean()), 3), round(ci_low, 3), round(ci_high, 3),
                      winner, len(a), len(b)],
        })

        fig = px.box(df[df[group_col].isin(groups)], x=group_col, y=metric_col, color=group_col)
        _style(fig, title=f"A/B Test: {metric_col} — {sig} (p={float(p):.4f})")
        fig.update_layout(showlegend=False)

        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"A/B test ({metric_col}): {sig} (p={float(p):.4f}). {winner} wins. Effect size d={effect_d:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]",
        )

    # ── 29. Pareto analysis ──────────────────────────────────────────────

    @staticmethod
    def handle_pareto_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Pareto 80/20 rule analysis on a column."""
        col = params.get("column")
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()

        if not col or col not in df.columns:
            col = cat_cols[0] if cat_cols else (num_cols[0] if num_cols else None)
        if col is None:
            return HandlerResult(success=False, error="No column found for Pareto analysis")

        counts = df[col].value_counts()
        total = counts.sum()
        pareto = pd.DataFrame({"category": counts.index.astype(str), "count": counts.values})
        pareto["pct"] = round(pareto["count"] / total * 100, 2)
        pareto["cumulative_pct"] = round(pareto["pct"].cumsum(), 2)

        cutoff_idx = int((pareto["cumulative_pct"] >= 80).idxmax())
        n_for_80 = cutoff_idx + 1
        pct_categories = round(n_for_80 / len(pareto) * 100, 1)

        top = pareto.head(20)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=top["category"], y=top["count"], name="Count", marker_color="#FB8C3C"), secondary_y=False)
        fig.add_trace(go.Scatter(x=top["category"], y=top["cumulative_pct"], name="Cumulative %",
                                 line=dict(color="#E71D36", width=2), mode="lines+markers"), secondary_y=True)
        fig.add_hline(y=80, line_dash="dash", line_color="#86868B", secondary_y=True, annotation_text="80%")
        _style(fig, title=f"Pareto Analysis — {col}")
        fig.update_layout(yaxis_title="Count", yaxis2_title="Cumulative %")

        return HandlerResult(
            success=True, result_df=pareto, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"Pareto: top {n_for_80} of {len(pareto)} categories ({pct_categories:.0f}%) account for 80% of values in '{col}'",
        )

    # ── 30. Data completeness scorecard ──────────────────────────────────

    @staticmethod
    def handle_data_completeness(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Data completeness scorecard per column + overall score."""
        rows: list[dict] = []
        for c in df.columns:
            n_total = len(df)
            n_present = int(df[c].count())
            n_missing = n_total - n_present
            completeness = round(n_present / max(n_total, 1) * 100, 2)
            grade = "A" if completeness >= 95 else "B" if completeness >= 80 else "C" if completeness >= 50 else "F"
            rows.append({
                "column": c, "total": n_total, "present": n_present,
                "missing": n_missing, "completeness_pct": completeness, "grade": grade,
            })

        result_df = pd.DataFrame(rows).sort_values("completeness_pct", ascending=True)
        overall = round(float(result_df["completeness_pct"].mean()), 1)
        overall_grade = "A" if overall >= 95 else "B" if overall >= 80 else "C" if overall >= 50 else "F"

        fig = px.bar(result_df, x="completeness_pct", y="column", orientation="h",
                     color="grade", text="completeness_pct",
                     color_discrete_map={"A": "#2EC4B6", "B": "#FB8C3C", "C": "#FF9F1C", "F": "#E71D36"})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        _style(fig, title=f"Data Completeness Scorecard — Overall: {overall}% ({overall_grade})")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_range=[0, 105])

        incomplete = [r for r in rows if r["completeness_pct"] < 100]
        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"Data completeness: {overall}% ({overall_grade}). {len(incomplete)}/{len(rows)} columns have missing values.",
        )

    # ── 31. Seasonality detect ───────────────────────────────────────────

    @staticmethod
    def handle_seasonality_detect(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col = col if col and col in num_cols else (num_cols[0] if num_cols else None)
        if col is None:
            return HandlerResult(success=False, error="No numeric column found")
        s = df[col].dropna().reset_index(drop=True)
        if len(s) < 10:
            return HandlerResult(success=False, error="Need ≥10 data points")
        autocorrs = [round(float(s.autocorr(lag=i)), 4) for i in range(1, min(len(s) // 2, 30))]
        peak_lag = int(np.argmax(autocorrs[1:]) + 2) if len(autocorrs) > 1 else 0
        peak_val = max(autocorrs[1:]) if len(autocorrs) > 1 else 0
        result_df = pd.DataFrame({"lag": list(range(1, len(autocorrs) + 1)), "autocorrelation": autocorrs})
        fig = px.bar(result_df, x="lag", y="autocorrelation", text="autocorrelation")
        fig.update_traces(marker_color="#FB8C3C", texttemplate="%{text:.3f}", textposition="outside")
        _style(fig, title=f"Autocorrelation — {col} (peak lag={peak_lag}, r={peak_val:.3f})")
        seasonal = "Likely seasonal" if peak_val > 0.3 else "No clear seasonality"
        return HandlerResult(success=True, result_df=result_df, output_type="query", charts_plotly=[fig.to_json()],
                             summary=f"{seasonal} in '{col}'. Peak autocorrelation at lag {peak_lag} (r={peak_val:.3f})")

    # ── 32. Cluster profile ──────────────────────────────────────────────

    @staticmethod
    def handle_cluster_profile(df: pd.DataFrame, params: dict) -> HandlerResult:
        n_clusters = int(params.get("n", 3))
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return HandlerResult(success=False, error="Need ≥2 numeric columns")
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            X = df[num_cols].dropna()
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(Xs)
            X_copy = X.copy()
            X_copy["_cluster"] = labels
            profile = X_copy.groupby("_cluster")[num_cols].mean().round(3).reset_index()
            counts = X_copy["_cluster"].value_counts().sort_index()
            profile["_count"] = counts.values
            fig = px.bar(profile.melt(id_vars=["_cluster", "_count"], value_vars=num_cols[:6]),
                         x="variable", y="value", color="_cluster", barmode="group", text="value")
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            _style(fig, title=f"Cluster Profiles — {n_clusters} clusters")
            return HandlerResult(success=True, result_df=profile, output_type="query", charts_plotly=[fig.to_json()],
                                 summary=f"Profiled {n_clusters} clusters across {len(num_cols)} features. Sizes: {dict(counts)}")
        except Exception as e:
            return HandlerResult(success=False, error=f"Cluster profile error: {e}")

    # ── 33. Feature interaction detect ───────────────────────────────────

    @staticmethod
    def handle_feature_interaction(df: pd.DataFrame, params: dict) -> HandlerResult:
        target = params.get("column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not target or target not in num_cols:
            target = num_cols[-1] if num_cols else None
        if target is None or len(num_cols) < 3:
            return HandlerResult(success=False, error="Need target + ≥2 features")
        feats = [c for c in num_cols if c != target][:8]
        rows = []
        y = df[target].dropna()
        for i, f1 in enumerate(feats):
            for f2 in feats[i + 1:]:
                clean = df[[f1, f2, target]].dropna()
                interaction = clean[f1] * clean[f2]
                r_individual = max(abs(clean[f1].corr(clean[target])), abs(clean[f2].corr(clean[target])))
                r_interaction = abs(interaction.corr(clean[target]))
                if r_interaction > r_individual:
                    rows.append({"feature_1": f1, "feature_2": f2, "individual_max_r": round(r_individual, 4),
                                 "interaction_r": round(r_interaction, 4), "lift": round(r_interaction - r_individual, 4)})
        rows.sort(key=lambda x: x["lift"], reverse=True)
        result_df = pd.DataFrame(rows[:15]) if rows else pd.DataFrame(columns=["feature_1", "feature_2", "interaction_r", "lift"])
        return HandlerResult(success=True, result_df=result_df, output_type="query",
                             summary=f"Found {len(rows)} feature interactions improving correlation with '{target}'")

    # ── 34. Cohort analysis ──────────────────────────────────────────────

    @staticmethod
    def handle_cohort_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        value_col = params.get("value_column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        col = col if col and col in df.columns else (cat_cols[0] if cat_cols else None)
        value_col = value_col if value_col and value_col in num_cols else (num_cols[0] if num_cols else None)
        if not col or not value_col:
            return HandlerResult(success=False, error="Need categorical column + numeric value column")
        cohorts = df.groupby(col)[value_col].agg(["count", "mean", "median", "sum", "std"]).round(2).reset_index()
        cohorts = cohorts.sort_values("mean", ascending=False)
        fig = px.bar(cohorts, x=col, y="mean", text="count", color="mean", color_continuous_scale="YlOrRd")
        fig.update_traces(texttemplate="n=%{text}", textposition="outside")
        _style(fig, title=f"Cohort Analysis — {value_col} by {col}")
        return HandlerResult(success=True, result_df=cohorts, output_type="query", charts_plotly=[fig.to_json()],
                             summary=f"Analyzed {len(cohorts)} cohorts by '{col}'. Mean '{value_col}' range: {cohorts['mean'].min():.2f}–{cohorts['mean'].max():.2f}")

    # ── 35. RFM analysis ────────────────────────────────────────────────

    @staticmethod
    def handle_rfm_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 3:
            return HandlerResult(success=False, error="Need ≥3 numeric columns for RFM scoring")
        cols = num_cols[:3]
        result = df.copy()
        for c in cols:
            result[f"{c}_score"] = pd.qcut(result[c].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        score_cols = [f"{c}_score" for c in cols]
        result["rfm_total"] = result[score_cols].sum(axis=1)
        summary = result[score_cols + ["rfm_total"]].describe().round(2).reset_index()
        fig = px.histogram(result, x="rfm_total", nbins=13, text_auto=True)
        fig.update_traces(marker_color="#FB8C3C")
        _style(fig, title=f"RFM Score Distribution (columns: {', '.join(cols)})")
        return HandlerResult(success=True, result_df=result, output_type="query", charts_plotly=[fig.to_json()],
                             summary=f"RFM scoring on {cols}. Score range: {int(result['rfm_total'].min())}–{int(result['rfm_total'].max())}")

    # ── 36. Gap analysis ─────────────────────────────────────────────────

    @staticmethod
    def handle_gap_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col = col if col and col in num_cols else (num_cols[0] if num_cols else None)
        if col is None:
            return HandlerResult(success=False, error="No numeric column found")
        s = df[col].dropna().sort_values().reset_index(drop=True)
        diffs = s.diff().dropna()
        mean_gap = float(diffs.mean())
        std_gap = float(diffs.std()) if len(diffs) > 1 else 0
        threshold = mean_gap + 2 * std_gap
        large_gaps = diffs[diffs > threshold]
        rows = [{"position": int(i), "value_before": round(float(s.iloc[i - 1]), 4), "value_after": round(float(s.iloc[i]), 4),
                 "gap_size": round(float(diffs.iloc[i - 1]), 4)} for i in large_gaps.index[:20]]
        result_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["position", "value_before", "value_after", "gap_size"])
        return HandlerResult(success=True, result_df=result_df, output_type="query",
                             summary=f"Found {len(large_gaps)} significant gaps in '{col}' (threshold={threshold:.2f}, mean_gap={mean_gap:.2f})")

    # ── 37. Benchmark compare ────────────────────────────────────────────

    @staticmethod
    def handle_benchmark_compare(df: pd.DataFrame, params: dict) -> HandlerResult:
        benchmarks = params.get("benchmarks")  # dict {col: target_value}
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not benchmarks or not isinstance(benchmarks, dict):
            benchmarks = {c: float(df[c].mean()) for c in num_cols[:5]}
        rows = []
        for col, target in benchmarks.items():
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                actual = float(df[col].mean())
                diff = actual - float(target)
                pct = diff / abs(float(target)) * 100 if float(target) != 0 else 0
                rows.append({"column": col, "actual": round(actual, 2), "benchmark": float(target),
                             "difference": round(diff, 2), "pct_diff": round(pct, 1),
                             "status": "Above" if diff > 0 else "Below" if diff < 0 else "Equal"})
        result_df = pd.DataFrame(rows)
        fig = px.bar(result_df, x="column", y="pct_diff", color="status", text="pct_diff",
                     color_discrete_map={"Above": "#2EC4B6", "Below": "#E71D36", "Equal": "#86868B"})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        _style(fig, title="Benchmark Comparison — % Difference from Target")
        return HandlerResult(success=True, result_df=result_df, output_type="query", charts_plotly=[fig.to_json()],
                             summary=f"Compared {len(rows)} metrics against benchmarks")

    # ── 38. Sensitivity analysis ─────────────────────────────────────────

    @staticmethod
    def handle_sensitivity_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
        target = params.get("column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        target = target if target and target in num_cols else (num_cols[-1] if num_cols else None)
        if target is None or len(num_cols) < 2:
            return HandlerResult(success=False, error="Need target + features")
        feats = [c for c in num_cols if c != target][:10]
        rows = []
        baseline = float(df[target].mean())
        for f in feats:
            std = float(df[f].std())
            if std == 0: continue
            high = df.copy(); high[f] = high[f] + std
            low = df.copy(); low[f] = low[f] - std
            corr = abs(float(df[f].corr(df[target])))
            rows.append({"feature": f, "baseline": round(baseline, 2), "correlation": round(corr, 4),
                          "sensitivity": round(corr * std, 4), "std": round(std, 4)})
        rows.sort(key=lambda x: x["sensitivity"], reverse=True)
        result_df = pd.DataFrame(rows)
        fig = px.bar(result_df, x="sensitivity", y="feature", orientation="h", text="sensitivity")
        fig.update_traces(marker_color="#FB8C3C", texttemplate="%{text:.4f}", textposition="outside")
        _style(fig, title=f"Sensitivity Analysis — {target}")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        return HandlerResult(success=True, result_df=result_df, output_type="query", charts_plotly=[fig.to_json()],
                             summary=f"Sensitivity of '{target}' to {len(rows)} features")

    # ── 39. Correlation network ──────────────────────────────────────────

    @staticmethod
    def handle_correlation_network(df: pd.DataFrame, params: dict) -> HandlerResult:
        threshold = float(params.get("threshold", 0.5))
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return HandlerResult(success=False, error="Need ≥2 numeric columns")
        corr = df[num_cols].corr()
        edges = []
        for i, c1 in enumerate(num_cols):
            for j, c2 in enumerate(num_cols):
                if i < j and abs(corr.loc[c1, c2]) >= threshold:
                    edges.append({"source": c1, "target": c2, "correlation": round(float(corr.loc[c1, c2]), 4),
                                  "strength": "strong" if abs(corr.loc[c1, c2]) >= 0.7 else "moderate"})
        result_df = pd.DataFrame(edges) if edges else pd.DataFrame(columns=["source", "target", "correlation"])
        return HandlerResult(success=True, result_df=result_df, output_type="query",
                             summary=f"Correlation network: {len(edges)} edges above |r|≥{threshold} among {len(num_cols)} features")

    # ── 40. Feature drift ────────────────────────────────────────────────

    @staticmethod
    def handle_feature_drift(df: pd.DataFrame, params: dict) -> HandlerResult:
        num_cols = df.select_dtypes(include="number").columns.tolist()[:10]
        if not num_cols:
            return HandlerResult(success=False, error="No numeric columns")
        mid = len(df) // 2
        first, second = df.iloc[:mid], df.iloc[mid:]
        rows = []
        for c in num_cols:
            m1, m2 = float(first[c].mean()), float(second[c].mean())
            s1, s2 = float(first[c].std()), float(second[c].std())
            drift_pct = abs(m1 - m2) / max(abs(m1), 1e-10) * 100
            rows.append({"column": c, "first_half_mean": round(m1, 4), "second_half_mean": round(m2, 4),
                          "drift_pct": round(drift_pct, 2), "drifted": drift_pct > 10})
        result_df = pd.DataFrame(rows)
        drifted = sum(1 for r in rows if r["drifted"])
        fig = px.bar(result_df, x="column", y="drift_pct", color="drifted", text="drift_pct",
                     color_discrete_map={True: "#E71D36", False: "#2EC4B6"})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        _style(fig, title=f"Feature Drift Analysis — {drifted}/{len(rows)} features drifted")
        return HandlerResult(success=True, result_df=result_df, output_type="query", charts_plotly=[fig.to_json()],
                             summary=f"Feature drift: {drifted}/{len(rows)} features show >10% mean drift between halves")

    # ── 41. Sample bias check ────────────────────────────────────────────

    @staticmethod
    def handle_sample_bias_check(df: pd.DataFrame, params: dict) -> HandlerResult:
        sample_frac = float(params.get("sample_frac", 0.3))
        num_cols = df.select_dtypes(include="number").columns.tolist()[:8]
        if not num_cols:
            return HandlerResult(success=False, error="No numeric columns")
        sample = df.sample(frac=sample_frac, random_state=42)
        rows = []
        for c in num_cols:
            pop_mean, samp_mean = float(df[c].mean()), float(sample[c].mean())
            bias = abs(pop_mean - samp_mean) / max(abs(pop_mean), 1e-10) * 100
            rows.append({"column": c, "population_mean": round(pop_mean, 4), "sample_mean": round(samp_mean, 4),
                          "bias_pct": round(bias, 2), "acceptable": bias < 5})
        result_df = pd.DataFrame(rows)
        issues = sum(1 for r in rows if not r["acceptable"])
        return HandlerResult(success=True, result_df=result_df, output_type="query",
                             summary=f"Sample bias check ({sample_frac*100:.0f}% sample): {issues}/{len(rows)} features show >5% bias")

    # ── 42. Effect size ──────────────────────────────────────────────────

    @staticmethod
    def handle_effect_size(df: pd.DataFrame, params: dict) -> HandlerResult:
        group_col = params.get("column")
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()
        group_col = group_col if group_col and group_col in cat_cols else (cat_cols[0] if cat_cols else None)
        if not group_col or not num_cols:
            return HandlerResult(success=False, error="Need categorical group column + numeric features")
        groups = df[group_col].dropna().unique()[:2]
        if len(groups) < 2:
            return HandlerResult(success=False, error=f"Need ≥2 groups in '{group_col}'")
        g1, g2 = df[df[group_col] == groups[0]], df[df[group_col] == groups[1]]
        rows = []
        for c in num_cols[:10]:
            m1, m2 = float(g1[c].mean()), float(g2[c].mean())
            s_pooled = np.sqrt((float(g1[c].std()) ** 2 + float(g2[c].std()) ** 2) / 2)
            d = (m1 - m2) / max(s_pooled, 1e-10)
            size = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small"
            rows.append({"feature": c, "mean_1": round(m1, 4), "mean_2": round(m2, 4),
                          "cohens_d": round(d, 4), "effect_size": size})
        result_df = pd.DataFrame(rows)
        return HandlerResult(success=True, result_df=result_df, output_type="query",
                             summary=f"Effect sizes ({groups[0]} vs {groups[1]}): {sum(1 for r in rows if r['effect_size']!='small')}/{len(rows)} features have medium/large effect")

    # ── 43. Bootstrap CI ─────────────────────────────────────────────────

    @staticmethod
    def handle_bootstrap_ci(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        n_boot = int(params.get("n_bootstrap", 1000))
        ci = float(params.get("confidence", 0.95))
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col = col if col and col in num_cols else (num_cols[0] if num_cols else None)
        if col is None:
            return HandlerResult(success=False, error="No numeric column found")
        s = df[col].dropna().values
        rng = np.random.RandomState(42)
        means = [float(np.mean(rng.choice(s, size=len(s), replace=True))) for _ in range(n_boot)]
        alpha = (1 - ci) / 2
        lo, hi = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
        result_df = pd.DataFrame({"metric": ["mean", "ci_lower", "ci_upper", "ci_width", "n_bootstrap"],
                                   "value": [round(float(np.mean(s)), 4), round(float(lo), 4), round(float(hi), 4),
                                             round(float(hi - lo), 4), n_boot]})
        fig = px.histogram(x=means, nbins=40)
        fig.update_traces(marker_color="#FB8C3C")
        fig.add_vline(x=lo, line_dash="dash", line_color="#E71D36")
        fig.add_vline(x=hi, line_dash="dash", line_color="#E71D36")
        _style(fig, title=f"Bootstrap {ci*100:.0f}% CI — {col}: [{lo:.4f}, {hi:.4f}]")
        return HandlerResult(success=True, result_df=result_df, output_type="query", charts_plotly=[fig.to_json()],
                             summary=f"Bootstrap {ci*100:.0f}% CI for '{col}' mean: [{lo:.4f}, {hi:.4f}] (n={n_boot})")

    # ── 44. Cross-correlation ────────────────────────────────────────────

    @staticmethod
    def handle_cross_correlation(df: pd.DataFrame, params: dict) -> HandlerResult:
        columns = params.get("columns", [])
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(columns) >= 2 and all(c in num_cols for c in columns[:2]):
            c1, c2 = columns[0], columns[1]
        elif len(num_cols) >= 2:
            c1, c2 = num_cols[0], num_cols[1]
        else:
            return HandlerResult(success=False, error="Need 2 numeric columns")
        max_lag = min(len(df) // 3, 20)
        rows = []
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                r = float(df[c1].iloc[lag:].reset_index(drop=True).corr(df[c2].iloc[:len(df) - lag].reset_index(drop=True)))
            else:
                r = float(df[c2].iloc[-lag:].reset_index(drop=True).corr(df[c1].iloc[:len(df) + lag].reset_index(drop=True)))
            rows.append({"lag": lag, "correlation": round(r, 4) if not np.isnan(r) else 0})
        result_df = pd.DataFrame(rows)
        peak = result_df.loc[result_df["correlation"].abs().idxmax()]
        fig = px.bar(result_df, x="lag", y="correlation")
        fig.update_traces(marker_color="#2EC4B6")
        _style(fig, title=f"Cross-Correlation: {c1} vs {c2} (peak lag={int(peak['lag'])}, r={peak['correlation']:.3f})")
        return HandlerResult(success=True, result_df=result_df, output_type="query", charts_plotly=[fig.to_json()],
                             summary=f"Cross-correlation {c1}↔{c2}: peak at lag {int(peak['lag'])} (r={peak['correlation']:.3f})")

    # ── 45. Survival curve ───────────────────────────────────────────────

    @staticmethod
    def handle_survival_curve(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col = col if col and col in num_cols else (num_cols[0] if num_cols else None)
        if col is None:
            return HandlerResult(success=False, error="No numeric column found")
        s = df[col].dropna().sort_values().reset_index(drop=True)
        n = len(s)
        survival = [(1 - (i + 1) / n) for i in range(n)]
        result_df = pd.DataFrame({col: s.values, "survival_prob": [round(x, 4) for x in survival]})
        fig = go.Figure(go.Scatter(x=s.values, y=survival, mode="lines", line=dict(color="#FB8C3C", width=2)))
        fig.add_hline(y=0.5, line_dash="dash", line_color="#86868B")
        _style(fig, title=f"Survival Curve — {col}")
        fig.update_layout(xaxis_title=col, yaxis_title="Survival Probability")
        median_val = float(s.median())
        return HandlerResult(success=True, result_df=result_df, output_type="query", charts_plotly=[fig.to_json()],
                             summary=f"Survival curve for '{col}': median={median_val:.2f}, range [{s.min():.2f}, {s.max():.2f}]")

    # ── 46. Concentration analysis ───────────────────────────────────────

    @staticmethod
    def handle_concentration_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col = col if col and col in num_cols else (num_cols[0] if num_cols else None)
        if col is None:
            return HandlerResult(success=False, error="No numeric column found")
        s = df[col].dropna().sort_values().values
        n = len(s)
        cum = np.cumsum(s) / s.sum()
        pct = np.arange(1, n + 1) / n
        gini = float(1 - 2 * np.trapz(cum, pct))
        top10_share = float(s[int(n * 0.9):].sum() / s.sum() * 100)
        top20_share = float(s[int(n * 0.8):].sum() / s.sum() * 100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pct * 100, y=cum * 100, mode="lines", name="Lorenz", line=dict(color="#FB8C3C", width=2)))
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Equality", line=dict(color="#86868B", dash="dash")))
        _style(fig, title=f"Lorenz Curve — {col} (Gini={gini:.3f})")
        fig.update_layout(xaxis_title="Population %", yaxis_title="Cumulative %")
        result_df = pd.DataFrame({"metric": ["gini", "top_10%_share", "top_20%_share"], "value": [round(gini, 4), round(top10_share, 2), round(top20_share, 2)]})
        return HandlerResult(success=True, result_df=result_df, output_type="query", charts_plotly=[fig.to_json()],
                             summary=f"Concentration of '{col}': Gini={gini:.3f}, top 20% holds {top20_share:.1f}%")

    # ── 47. Diminishing returns ──────────────────────────────────────────

    @staticmethod
    def handle_diminishing_returns(df: pd.DataFrame, params: dict) -> HandlerResult:
        columns = params.get("columns", [])
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(columns) >= 2 and all(c in num_cols for c in columns[:2]):
            x_col, y_col = columns[0], columns[1]
        elif len(num_cols) >= 2:
            x_col, y_col = num_cols[0], num_cols[1]
        else:
            return HandlerResult(success=False, error="Need 2 numeric columns")
        clean = df[[x_col, y_col]].dropna().sort_values(x_col).reset_index(drop=True)
        n = len(clean)
        mid = n // 2
        r_first = float(clean.iloc[:mid][[x_col, y_col]].corr().iloc[0, 1])
        r_second = float(clean.iloc[mid:][[x_col, y_col]].corr().iloc[0, 1])
        diminishing = r_first > r_second and r_first > 0
        fig = px.scatter(clean, x=x_col, y=y_col, trendline="lowess", opacity=0.5)
        fig.update_traces(marker_color="#FB8C3C")
        _style(fig, title=f"Diminishing Returns? {x_col} → {y_col} ({'Yes' if diminishing else 'No'})")
        result_df = pd.DataFrame({"metric": ["r_first_half", "r_second_half", "diminishing"], "value": [round(r_first, 4), round(r_second, 4), diminishing]})
        return HandlerResult(success=True, result_df=result_df, output_type="query", charts_plotly=[fig.to_json()],
                             summary=f"{'Diminishing returns detected' if diminishing else 'No diminishing returns'}: r drops from {r_first:.3f} to {r_second:.3f}")

    # ── 48. Categorical target crosstab ──────────────────────────────────

    @staticmethod
    def handle_categorical_target_crosstab(df: pd.DataFrame, params: dict) -> HandlerResult:
        target = params.get("column")
        feature = params.get("feature_column")
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if not target or target not in cat_cols:
            target = cat_cols[0] if cat_cols else None
        if not feature or feature not in cat_cols:
            feature = cat_cols[1] if len(cat_cols) > 1 else None
        if not target or not feature:
            return HandlerResult(success=False, error="Need 2 categorical columns")
        ct = pd.crosstab(df[feature], df[target], margins=True, margins_name="Total")
        ct_pct = pd.crosstab(df[feature], df[target], normalize="index").round(4) * 100
        fig = px.imshow(ct_pct.values[:-1] if "Total" in ct_pct.index else ct_pct.values,
                        x=ct_pct.columns.tolist(), y=ct_pct.index.tolist(),
                        text_auto=".1f", color_continuous_scale="YlOrRd")
        _style(fig, title=f"Crosstab: {feature} × {target} (%)")
        return HandlerResult(success=True, result_df=ct.reset_index(), output_type="query", charts_plotly=[fig.to_json()],
                             summary=f"Crosstab: {feature} ({df[feature].nunique()} levels) × {target} ({df[target].nunique()} levels)")

    # ── 49. Prediction baseline ──────────────────────────────────────────

    @staticmethod
    def handle_prediction_baseline(df: pd.DataFrame, params: dict) -> HandlerResult:
        target = params.get("column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if target and target in df.columns:
            pass
        elif num_cols:
            target = num_cols[-1]
        elif cat_cols:
            target = cat_cols[0]
        else:
            return HandlerResult(success=False, error="No columns found")
        s = df[target].dropna()
        rows = []
        if pd.api.types.is_numeric_dtype(s):
            rows.append({"baseline": "mean", "value": round(float(s.mean()), 4), "metric": "MAE", "score": round(float((s - s.mean()).abs().mean()), 4)})
            rows.append({"baseline": "median", "value": round(float(s.median()), 4), "metric": "MAE", "score": round(float((s - s.median()).abs().mean()), 4)})
            rows.append({"baseline": "zero", "value": 0, "metric": "MAE", "score": round(float(s.abs().mean()), 4)})
        else:
            mode = s.mode().iloc[0] if len(s.mode()) > 0 else "N/A"
            acc = float((s == mode).mean())
            rows.append({"baseline": "most_frequent", "value": str(mode), "metric": "accuracy", "score": round(acc, 4)})
            rows.append({"baseline": "random", "value": "uniform", "metric": "accuracy", "score": round(1.0 / max(s.nunique(), 1), 4)})
        result_df = pd.DataFrame(rows)
        return HandlerResult(success=True, result_df=result_df, output_type="query",
                             summary=f"Baselines for '{target}': best naive = {rows[0]['baseline']} ({rows[0]['metric']}={rows[0]['score']:.4f})")

    # ── 50. Data readiness score ─────────────────────────────────────────

    @staticmethod
    def handle_data_readiness_score(df: pd.DataFrame, params: dict) -> HandlerResult:
        scores = {}
        # Completeness
        null_pct = df.isnull().mean().mean() * 100
        scores["completeness"] = round(max(0, 100 - null_pct * 2), 1)
        # Duplicates
        dup_pct = df.duplicated().mean() * 100
        scores["uniqueness"] = round(max(0, 100 - dup_pct * 2), 1)
        # Consistency (no constant cols, no mixed types)
        constant_pct = sum(1 for c in df.columns if df[c].nunique() <= 1) / max(len(df.columns), 1) * 100
        scores["consistency"] = round(max(0, 100 - constant_pct * 5), 1)
        # Balance (for categoricals)
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            imb = max(df[c].value_counts().max() / max(df[c].value_counts().min(), 1) for c in cat_cols[:3])
            scores["balance"] = round(max(0, 100 - (imb - 1) * 10), 1)
        else:
            scores["balance"] = 100.0
        # Size
        scores["size"] = min(100.0, round(len(df) / 10, 1))  # 1000 rows = 100%
        overall = round(sum(scores.values()) / len(scores), 1)
        grade = "A" if overall >= 90 else "B" if overall >= 75 else "C" if overall >= 50 else "D"
        result_df = pd.DataFrame([{"dimension": k, "score": v} for k, v in scores.items()])
        result_df = pd.concat([result_df, pd.DataFrame([{"dimension": "OVERALL", "score": overall}])], ignore_index=True)
        fig = px.bar(result_df, x="score", y="dimension", orientation="h", text="score",
                     color="score", color_continuous_scale=["#E71D36", "#FF9F1C", "#2EC4B6"])
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        _style(fig, title=f"ML Data Readiness — {overall}/100 ({grade})")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_range=[0, 110], coloraxis_showscale=False)
        return HandlerResult(success=True, result_df=result_df, output_type="query", charts_plotly=[fig.to_json()],
                             summary=f"Data readiness score: {overall}/100 ({grade}). {', '.join(f'{k}={v}' for k, v in scores.items())}")
