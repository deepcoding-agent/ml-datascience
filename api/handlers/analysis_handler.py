"""Analysis handler — 10 smart, high-level analytical handlers.

These go beyond single-column stats: they reason about the data,
combine multiple operations, and return rich, insight-driven results
with charts and formatted summaries.
"""
from __future__ import annotations

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
