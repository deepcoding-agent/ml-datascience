"""Document Agent — generates comprehensive EDA report documents.

Flow:
  1. Compute deep analysis (profiles, distributions, correlations, quality)
  2. Generate Plotly charts (distribution, correlation heatmap, missing values, outlier box)
  3. AI writes narrative per section
  4. Return structured document with sections + embedded charts
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from langchain_core.messages import HumanMessage

from api.agents.context_analyzer import analyze_context
from api.llm import get_llm
from api.logger import get_logger

log = get_logger(__name__)

# ── Theme (matches PrepPilot brand) ──────────────────────────────────────────

_COLORS = ["#FB8C3C", "#2EC4B6", "#457B9D", "#E71D36", "#FF9F1C", "#A8DADC", "#1D3557"]

_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Noto Sans Thai, Tahoma, sans-serif", size=12),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=40, b=40),
    colorway=_COLORS,
)


def _style(fig: go.Figure, **kw) -> go.Figure:
    fig.update_layout(**{**_LAYOUT, **kw})
    fig.update_xaxes(showgrid=False, tickfont=dict(size=10))
    fig.update_yaxes(showgrid=True, gridcolor="#F0F0F0", tickfont=dict(size=10))
    return fig


# ── AI prompt ────────────────────────────────────────────────────────────────

DOCUMENT_PROMPT = """\
You are a senior data scientist writing a technical EDA report for a fellow
analyst. The audience knows statistics. Your job is to characterize the data
deeply — distributions, relationships, quality, modeling potential — NOT to
give business advice (a separate /biz-report covers that).

## Dataset Analysis
{analysis}

## Instructions
Write a JSON EDA report. Every claim must cite specific column names, numbers,
or percentages from the analysis above. Be precise and technical.

IMPORTANT: Respond in the same language as the column names suggest.
If columns look Thai (ราคา, ชื่อ, etc.) → write in Thai.
If columns look English (price, name, etc.) → write in English.
Mixed → write in the dominant language.

Return EXACTLY this JSON (no markdown fences):
{{
  "title": "EDA Report: <dataset name>",
  "executive_summary": "4-5 sentences: shape, dominant data types, the 2-3 most important statistical findings, and the headline ML readiness verdict. Pure data characterization, no business framing.",
  "data_overview": "3-4 sentences: row/column counts, memory footprint, dtype mix (numeric/categorical/datetime), and what each major column likely represents based on its name and stats.",
  "quality_assessment": "4-5 sentences: completeness %, duplicate %, constant columns, high-null columns. Classify each as MCAR/MAR/MNAR if hints exist. Concrete: should we drop, impute, or flag each problem column?",
  "distribution_analysis": "5-7 sentences: For each notable numeric, classify its shape (normal/right-skew/left-skew/bimodal/uniform/heavy-tail). Cite skew values. Flag log-transform candidates explicitly. Mention scale heterogeneity if numeric ranges differ by orders of magnitude.",
  "correlation_analysis": "5-7 sentences: Top 3-5 correlated pairs by name + value. Explicitly call out |r|>0.9 multicollinearity (which feature in each pair to drop). Note if a likely target shows strong signal correlations with features. Spurious correlation warnings if any (e.g. ID-like columns).",
  "missing_value_analysis": "3-4 sentences: column-by-column null % for problem columns. Pattern detection: are nulls concentrated in specific rows or random? Which columns are imputable vs which should be dropped.",
  "categorical_analysis": "4-5 sentences: cardinality per categorical. Flag dominant-value imbalance (>80% one value). Flag high-cardinality (>50 unique → don't one-hot). Suggest encoding strategy per column (one-hot, ordinal, target, frequency).",
  "outlier_analysis": "4-5 sentences: Columns with most IQR outliers by % and count. Distinguish true outliers (heavy tails) from data errors. Suggest treatment per column: clip/winsorize/keep-as-feature/log-transform.",
  "ml_readiness": "5-7 sentences: ML readiness score with breakdown of each factor (sample size, feature-to-sample ratio, completeness, scale heterogeneity, multicollinearity). List the top probable target columns with task type. State which ML approaches are immediately viable vs which need more preprocessing.",
  "analytical_directions": "5-7 sentences: What analyses can this dataset support? Be specific — list 4-6 concrete analyses by name (e.g. 'churn classification using churn as target with logistic regression as baseline', 'cohort analysis on hire_date', 'price-feature regression with log-transformed price'). For each, name the columns and method.",
  "recommendations": [
    "Data-prep step 1 — specific cleaning/encoding action with column names",
    "Data-prep step 2 — feature engineering with rationale",
    "Modeling experiment 1 — algorithm, target, expected metric range",
    "Modeling experiment 2 — ...",
    "Further data to collect (if obvious gaps exist)"
  ],
  "conclusion": "3-4 sentences: technical bottom line. Is this dataset analysis-ready or modeling-ready? What's the highest-value analysis to run first, and why?"
}}

Rules:
- Cite ACTUAL column names, numbers, percentages from the analysis
- Use statistical vocabulary (skewness, kurtosis, IQR, multicollinearity, MCAR/MAR, stratification)
- NO business framing (no revenue, ROI, customer segments, marketing) — /biz-report handles that
- Every sentence must say something concrete about THIS dataset, not generic EDA advice
- If a section truly has nothing notable to report, write a single sentence saying so
"""


# ── Chart generators ─────────────────────────────────────────────────────────

def _chart_distribution(df: pd.DataFrame, nums: list[str]) -> str | None:
    """Distribution histograms for top numeric columns."""
    cols = nums[:6]
    if not cols:
        return None
    n = len(cols)
    rows_count = (n + 2) // 3
    fig = make_subplots(rows=rows_count, cols=min(n, 3),
                        subplot_titles=cols)
    for i, col in enumerate(cols):
        r, c = i // 3 + 1, i % 3 + 1
        data = df[col].dropna()
        fig.add_trace(go.Histogram(x=data, nbinsx=30, marker_color=_COLORS[i % len(_COLORS)],
                                   name=col, showlegend=False), row=r, col=c)
    _style(fig, title="Numeric Distributions", height=250 * rows_count)
    return fig.to_json()


def _chart_correlation(df: pd.DataFrame, nums: list[str]) -> str | None:
    """Correlation heatmap."""
    cols = nums[:15]
    if len(cols) < 2:
        return None
    corr = df[cols].corr().round(3)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale="RdBu_r", zmin=-1, zmax=1, text=corr.values.round(2),
        texttemplate="%{text}", textfont=dict(size=9),
    ))
    _style(fig, title="Correlation Matrix", height=max(350, len(cols) * 30))
    return fig.to_json()


def _chart_missing(df: pd.DataFrame) -> str | None:
    """Missing values bar chart."""
    null_pcts = (df.isnull().mean() * 100).round(1)
    null_pcts = null_pcts[null_pcts > 0].sort_values(ascending=True)
    if len(null_pcts) == 0:
        return None
    fig = go.Figure(go.Bar(
        x=null_pcts.values, y=null_pcts.index.tolist(),
        orientation="h", marker_color="#E71D36",
        text=[f"{v:.1f}%" for v in null_pcts.values], textposition="outside",
    ))
    _style(fig, title="Missing Values (%)", height=max(250, len(null_pcts) * 25),
           xaxis_title="Missing %")
    return fig.to_json()


def _chart_outlier_box(df: pd.DataFrame, nums: list[str]) -> str | None:
    """Box plots for outlier visualization."""
    cols = nums[:8]
    if not cols:
        return None
    # Normalize for comparison
    normalized = df[cols].apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x)
    melted = normalized.melt(var_name="column", value_name="z_score")
    fig = px.box(melted, x="column", y="z_score", color="column",
                 color_discrete_sequence=_COLORS)
    _style(fig, title="Outlier Overview (Z-Score Normalized)", showlegend=False)
    return fig.to_json()


def _chart_categorical(df: pd.DataFrame, cats: list[str]) -> str | None:
    """Top categorical columns distribution."""
    cols = [c for c in cats if df[c].nunique() <= 15][:4]
    if not cols:
        return None
    n = len(cols)
    fig = make_subplots(rows=1, cols=n, subplot_titles=cols)
    for i, col in enumerate(cols):
        vc = df[col].value_counts().head(10)
        fig.add_trace(go.Bar(x=vc.index.tolist(), y=vc.values.tolist(),
                             marker_color=_COLORS[i % len(_COLORS)],
                             name=col, showlegend=False), row=1, col=i + 1)
    _style(fig, title="Categorical Distributions", height=300)
    return fig.to_json()


def _chart_dtype_pie(df: pd.DataFrame) -> str | None:
    """Pie chart of column data types."""
    dtype_counts = df.dtypes.astype(str).value_counts()
    fig = go.Figure(go.Pie(
        labels=dtype_counts.index.tolist(), values=dtype_counts.values.tolist(),
        marker=dict(colors=_COLORS), textinfo="label+value+percent",
        textfont=dict(size=11),
    ))
    _style(fig, title="Column Data Types", height=300, showlegend=False)
    return fig.to_json()


# ── Column profiling ─────────────────────────────────────────────────────────

def _build_column_profiles(df: pd.DataFrame) -> list[dict]:
    """Build per-column profile data."""
    profiles = []
    for col in df.columns:
        s = df[col]
        profile: dict = {
            "name": col,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "null_count": int(s.isna().sum()),
            "null_pct": round(s.isna().mean() * 100, 1),
            "unique": int(s.nunique()),
            "unique_pct": round(s.nunique() / max(len(s), 1) * 100, 1),
        }
        if pd.api.types.is_numeric_dtype(s):
            profile["type"] = "numeric"
            desc = s.describe()
            profile["mean"] = round(float(desc.get("mean", 0)), 4)
            profile["std"] = round(float(desc.get("std", 0)), 4)
            profile["min"] = round(float(desc.get("min", 0)), 4)
            profile["q25"] = round(float(desc.get("25%", 0)), 4)
            profile["median"] = round(float(desc.get("50%", 0)), 4)
            profile["q75"] = round(float(desc.get("75%", 0)), 4)
            profile["max"] = round(float(desc.get("max", 0)), 4)
            profile["skewness"] = round(float(s.skew()), 4) if len(s.dropna()) > 2 else 0
            profile["kurtosis"] = round(float(s.kurtosis()), 4) if len(s.dropna()) > 3 else 0
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
            profile["outliers"] = outliers
            profile["zeros"] = int((s == 0).sum())
        else:
            profile["type"] = "categorical"
            vc = s.value_counts()
            profile["top_value"] = str(vc.index[0]) if len(vc) > 0 else ""
            profile["top_freq"] = int(vc.iloc[0]) if len(vc) > 0 else 0
            profile["avg_length"] = round(s.astype(str).str.len().mean(), 1)
        profiles.append(profile)
    return profiles


# ── Main entry ───────────────────────────────────────────────────────────────

def run_document(
    data: list[dict],
    dataset_name: str,
    model_id: str | None = None,
) -> dict:
    """Generate a comprehensive EDA document report.

    Returns dict with: success, dataset_name, document (sections + charts),
    column_profiles, analysis, duration_seconds.
    """
    t0 = time.perf_counter()
    df = pd.DataFrame(data)
    log.info("Document: analyzing '%s' — %d x %d", dataset_name, *df.shape)

    ctx = analyze_context(df)

    # ── 1. Compute analysis ──────────────────────────────────────────
    total_cells = df.shape[0] * df.shape[1]
    null_cells = int(df.isnull().sum().sum())
    completeness = round((1 - null_cells / total_cells) * 100, 1) if total_cells > 0 else 100

    overview = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "memory_mb": round(ctx.memory_mb, 2),
        "numeric_count": len(ctx.numeric_cols),
        "categorical_count": len(ctx.categorical_cols),
        "datetime_count": len(ctx.datetime_cols),
        "completeness_pct": completeness,
        "duplicate_rows": ctx.duplicate_count,
        "duplicate_pct": round(ctx.duplicate_count / max(df.shape[0], 1) * 100, 1),
        "total_nulls": null_cells,
    }

    # Correlations
    top_corr = []
    if len(ctx.numeric_cols) >= 2:
        try:
            corr = df[ctx.numeric_cols[:30]].corr()
            seen = set()
            for i, c1 in enumerate(corr.columns):
                for j, c2 in enumerate(corr.columns):
                    if i >= j:
                        continue
                    v = corr.iloc[i, j]
                    if pd.notna(v) and abs(v) > 0.3:
                        pair = tuple(sorted([c1, c2]))
                        if pair not in seen:
                            seen.add(pair)
                            top_corr.append({"col1": c1, "col2": c2,
                                             "value": round(float(v), 4),
                                             "strength": "strong" if abs(v) > 0.7 else "moderate"})
            top_corr.sort(key=lambda x: abs(x["value"]), reverse=True)
            top_corr = top_corr[:15]
        except Exception:
            pass

    # Skewed columns
    skewed = []
    for col in ctx.numeric_cols[:20]:
        try:
            sk = float(df[col].skew())
            if abs(sk) > 1:
                skewed.append({"column": col, "skewness": round(sk, 2),
                               "direction": "right" if sk > 0 else "left"})
        except Exception:
            pass

    # Outlier summary
    outlier_summary = []
    for col in ctx.numeric_cols[:20]:
        try:
            s = df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
                if n_out > 0:
                    outlier_summary.append({"column": col, "count": n_out,
                                            "pct": round(n_out / len(s) * 100, 1)})
        except Exception:
            pass
    outlier_summary.sort(key=lambda x: x["pct"], reverse=True)

    analysis = {
        "overview": overview,
        "correlations": top_corr,
        "skewed_columns": skewed,
        "outlier_summary": outlier_summary[:10],
        "null_columns": ctx.null_cols,
        "high_null_columns": ctx.high_null_cols,
        "constant_columns": ctx.constant_cols,
        "warnings": ctx.warnings,
    }

    # ── Deep EDA: target detection, balance, ML readiness, datetime, cardinality ──
    # These are surfaced to the LLM narrator so each section can speak to them.
    # Build profiles once here; the same list is reused for column_profiles below.
    column_profiles = _build_column_profiles(df)
    analysis["target_candidates"]      = _detect_target_candidates(df, column_profiles)
    analysis["categorical_balance"]    = _categorical_balance(df, ctx.categorical_cols)
    analysis["high_cardinality_cats"]  = _high_cardinality_cats(df, ctx.categorical_cols)
    analysis["datetime_summary"]       = _datetime_summary(df, ctx.datetime_cols)
    analysis["multicollinear_pairs"]   = [c for c in top_corr if abs(c["value"]) > 0.9]
    analysis["ml_readiness"]           = _ml_readiness(df, analysis, ctx.numeric_cols)

    # ── 2. Generate charts ───────────────────────────────────────────
    charts = {}
    charts["distribution"] = _chart_distribution(df, ctx.numeric_cols)
    charts["correlation"] = _chart_correlation(df, ctx.numeric_cols)
    charts["missing"] = _chart_missing(df)
    charts["outlier_box"] = _chart_outlier_box(df, ctx.numeric_cols)
    charts["categorical"] = _chart_categorical(df, ctx.categorical_cols)
    charts["dtype_pie"] = _chart_dtype_pie(df)
    # Remove None entries
    charts = {k: v for k, v in charts.items() if v is not None}

    # ── 3. Column profiles already built above for target detection ──────

    # ── 4. AI narrative ──────────────────────────────────────────────
    analysis_text = _format_for_llm(df, analysis, column_profiles)
    sections = _ai_narrative(analysis_text, dataset_name, model_id)

    # ── 5. Quality score ─────────────────────────────────────────────
    score = 100
    score -= min(30, int((100 - completeness) * 0.5))
    score -= min(20, int(overview["duplicate_pct"] * 2))
    score -= min(10, len(analysis.get("constant_columns", [])) * 3)
    score -= min(10, len(analysis.get("high_null_columns", [])) * 5)
    score = max(0, score)
    quality_label = "Excellent" if score >= 85 else "Good" if score >= 70 else "Fair" if score >= 50 else "Poor"

    elapsed = round(time.perf_counter() - t0, 2)
    log.info("Document done: '%s' quality=%d/%s in %.1fs", dataset_name, score, quality_label, elapsed)

    return {
        "success": True,
        "dataset_name": dataset_name,
        "document": {
            "sections": sections,
            "charts": charts,
            "quality_score": score,
            "quality_label": quality_label,
        },
        "column_profiles": column_profiles,
        "analysis": analysis,
        "duration_seconds": elapsed,
    }


def _detect_target_candidates(df: pd.DataFrame, profiles: list[dict]) -> list[dict]:
    """Heuristically pick likely target columns + suggest task type.

    Three signals — last column position, name keywords, class-balance shape.
    Score is informal (0-100); the LLM uses it to suggest modeling directions.
    """
    candidates: list[dict] = []
    target_keywords = ("target", "label", "class", "y", "outcome", "price", "churn",
                       "fraud", "default", "rating", "score")
    last_col = df.columns[-1] if len(df.columns) else None

    for p in profiles:
        col = p["name"]
        score = 0
        reasons: list[str] = []

        if col == last_col:
            score += 30
            reasons.append("last column")
        if any(kw in col.lower() for kw in target_keywords):
            score += 40
            reasons.append("name suggests target")

        # Task type inference
        task_type = "regression"
        if p.get("type") == "numeric":
            uniq = p.get("unique", 0)
            if uniq <= 10:
                task_type = "classification"
                score += 15
                reasons.append(f"{uniq} discrete values")
            else:
                score += 10
                reasons.append("continuous numeric")
        else:
            uniq = p.get("unique", 0)
            if 2 <= uniq <= 20:
                task_type = "classification"
                score += 25
                reasons.append(f"{uniq} categories")
            else:
                # Free-text or high-cardinality categorical → probably not target
                continue

        if score > 0:
            candidates.append({
                "column": col,
                "score": score,
                "task_type": task_type,
                "reason": "; ".join(reasons),
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:5]


def _categorical_balance(df: pd.DataFrame, cat_cols: list[str]) -> list[dict]:
    """Flag categoricals whose dominant value covers > 80% — bad for classification."""
    results: list[dict] = []
    for col in cat_cols[:20]:
        try:
            counts = df[col].value_counts(dropna=False)
            if counts.empty:
                continue
            n_classes = len(counts)
            dominant_val = str(counts.index[0])
            dominant_pct = round(float(counts.iloc[0]) / max(len(df), 1) * 100, 1)
            results.append({
                "column": col,
                "n_classes": int(n_classes),
                "dominant_value": dominant_val[:50],
                "dominant_pct": dominant_pct,
                "imbalanced": dominant_pct > 80,
            })
        except Exception:
            continue
    return results


def _ml_readiness(df: pd.DataFrame, analysis: dict, num_cols: list[str]) -> dict:
    """Score how ready the dataset is for ML modeling (0-100) with breakdown."""
    factors: list[dict] = []
    score = 100

    rows, cols = df.shape

    # 1. Sample size adequacy
    if rows < 50:
        penalty, status = 40, "too small for modeling"
    elif rows < 500:
        penalty, status = 15, "small — use cross-validation aggressively"
    elif rows < 5000:
        penalty, status = 0, "adequate"
    else:
        penalty, status = 0, "good"
    score -= penalty
    factors.append({"factor": "sample_size", "value": rows, "penalty": penalty, "status": status})

    # 2. Feature/sample ratio (curse of dimensionality risk)
    ratio = cols / max(rows, 1)
    if ratio > 0.5:
        penalty, status = 25, f"{cols}/{rows} features-to-rows → overfitting risk"
    elif ratio > 0.1:
        penalty, status = 10, "moderate dimensionality"
    else:
        penalty, status = 0, "healthy"
    score -= penalty
    factors.append({"factor": "feature_ratio", "value": round(ratio, 3), "penalty": penalty, "status": status})

    # 3. Completeness
    completeness = analysis["overview"]["completeness_pct"]
    if completeness < 70:
        penalty, status = 25, "many missing values"
    elif completeness < 90:
        penalty, status = 10, "moderate missingness"
    else:
        penalty, status = 0, "clean"
    score -= penalty
    factors.append({"factor": "completeness", "value": completeness, "penalty": penalty, "status": status})

    # 4. Scale heterogeneity — flag if numeric features span > 6 orders of magnitude
    if len(num_cols) >= 2:
        try:
            ranges = [float(df[c].max() - df[c].min()) for c in num_cols
                      if df[c].notna().any() and df[c].max() != df[c].min()]
            if ranges:
                ratio = max(ranges) / max(min(ranges), 1e-9)
                if ratio > 1e6:
                    penalty, status = 5, "wide scale range — scaling required"
                else:
                    penalty, status = 0, "comparable scales"
                score -= penalty
                factors.append({"factor": "scale_heterogeneity", "value": f"{ratio:.0e}", "penalty": penalty, "status": status})
        except Exception:
            pass

    # 5. Multicollinearity
    multi_pairs = [c for c in analysis.get("correlations", []) if abs(c["value"]) > 0.9]
    if multi_pairs:
        penalty, status = min(15, len(multi_pairs) * 3), f"{len(multi_pairs)} pairs with |r|>0.9"
        score -= penalty
        factors.append({"factor": "multicollinearity", "value": len(multi_pairs), "penalty": penalty, "status": status})

    score = max(0, score)
    label = "Ready" if score >= 80 else "Mostly ready" if score >= 60 else "Needs work" if score >= 40 else "Not ready"
    return {"score": score, "label": label, "factors": factors}


def _datetime_summary(df: pd.DataFrame, dt_cols: list[str]) -> list[dict]:
    """Range + frequency analysis for datetime columns — flags potential time series."""
    results: list[dict] = []
    for col in dt_cols[:5]:
        try:
            s = pd.to_datetime(df[col], errors="coerce").dropna()
            if s.empty:
                continue
            results.append({
                "column": col,
                "min": str(s.min()),
                "max": str(s.max()),
                "range_days": int((s.max() - s.min()).days),
                "n_unique": int(s.nunique()),
                "looks_like_timeseries": s.nunique() == len(s),
            })
        except Exception:
            continue
    return results


def _high_cardinality_cats(df: pd.DataFrame, cat_cols: list[str]) -> list[dict]:
    """Flag categoricals with > 50 unique values — one-hot would explode."""
    results: list[dict] = []
    for col in cat_cols[:20]:
        try:
            uniq = int(df[col].nunique())
            if uniq > 50:
                results.append({
                    "column": col,
                    "unique": uniq,
                    "pct_of_rows": round(uniq / max(len(df), 1) * 100, 1),
                })
        except Exception:
            continue
    results.sort(key=lambda x: x["unique"], reverse=True)
    return results


def _format_for_llm(df: pd.DataFrame, analysis: dict, profiles: list[dict]) -> str:
    ov = analysis["overview"]
    lines = [
        f"Shape: {ov['rows']:,} rows x {ov['columns']} columns",
        f"Memory: {ov['memory_mb']} MB",
        f"Numeric: {ov['numeric_count']}, Categorical: {ov['categorical_count']}, Datetime: {ov['datetime_count']}",
        f"Completeness: {ov['completeness_pct']}%, Duplicates: {ov['duplicate_rows']} ({ov['duplicate_pct']}%)",
    ]
    if analysis["null_columns"]:
        lines.append(f"Null columns: {dict(list(analysis['null_columns'].items())[:10])}")
    if analysis["high_null_columns"]:
        lines.append(f"High null (>40%): {analysis['high_null_columns']}")
    if analysis["constant_columns"]:
        lines.append(f"Constant columns: {analysis['constant_columns']}")

    lines.append("\nColumn profiles:")
    for p in profiles[:20]:
        parts = [f"{p['name']} ({p['dtype']})"]
        if p.get("type") == "numeric":
            parts.append(f"mean={p.get('mean')}, std={p.get('std')}, range=[{p.get('min')}, {p.get('max')}]")
            if abs(p.get("skewness", 0)) > 1:
                parts.append(f"skew={p['skewness']}")
            if p.get("outliers", 0) > 0:
                parts.append(f"outliers={p['outliers']}")
        else:
            parts.append(f"unique={p.get('unique')}, top='{p.get('top_value')}'")
        if p.get("null_pct", 0) > 0:
            parts.append(f"null={p['null_pct']}%")
        lines.append(f"  {', '.join(parts)}")

    if analysis["correlations"]:
        lines.append("\nTop correlations:")
        for c in analysis["correlations"][:8]:
            lines.append(f"  {c['col1']} ↔ {c['col2']}: {c['value']} ({c['strength']})")

    if analysis["skewed_columns"]:
        lines.append(f"\nSkewed: {[(s['column'], s['skewness']) for s in analysis['skewed_columns'][:8]]}")

    if analysis["outlier_summary"]:
        lines.append("\nOutliers (IQR):")
        for o in analysis["outlier_summary"][:8]:
            lines.append(f"  {o['column']}: {o['count']} ({o['pct']}%)")

    # Deep EDA blocks for the LLM narrator
    if analysis.get("target_candidates"):
        lines.append("\nProbable target columns (use these to frame ML direction):")
        for t in analysis["target_candidates"][:3]:
            lines.append(f"  {t['column']} → {t['task_type']} (score={t['score']}, {t['reason']})")

    if analysis.get("multicollinear_pairs"):
        lines.append("\nMulticollinearity (|r|>0.9, drop one of each pair before linear models):")
        for c in analysis["multicollinear_pairs"][:8]:
            lines.append(f"  {c['col1']} ↔ {c['col2']}: {c['value']}")

    if analysis.get("categorical_balance"):
        imbalanced = [c for c in analysis["categorical_balance"] if c["imbalanced"]]
        if imbalanced:
            lines.append("\nImbalanced categoricals (dominant value > 80%):")
            for c in imbalanced[:5]:
                lines.append(f"  {c['column']}: '{c['dominant_value']}' = {c['dominant_pct']}% ({c['n_classes']} classes)")

    if analysis.get("high_cardinality_cats"):
        lines.append("\nHigh-cardinality categoricals (>50 unique — one-hot would explode):")
        for c in analysis["high_cardinality_cats"][:5]:
            lines.append(f"  {c['column']}: {c['unique']} unique values")

    if analysis.get("datetime_summary"):
        lines.append("\nDatetime columns:")
        for d in analysis["datetime_summary"]:
            ts = " (time-series candidate)" if d["looks_like_timeseries"] else ""
            lines.append(f"  {d['column']}: {d['min']} → {d['max']} ({d['range_days']} days{ts})")

    if analysis.get("ml_readiness"):
        mr = analysis["ml_readiness"]
        lines.append(f"\nML Readiness: {mr['score']}/100 — {mr['label']}")
        for f in mr["factors"]:
            lines.append(f"  - {f['factor']}: {f['status']} ({f['value']})")

    return "\n".join(lines)


def _ai_narrative(analysis_text: str, dataset_name: str, model_id: str | None) -> dict:
    """Use LLM to write narrative sections, fallback on failure."""
    try:
        llm = get_llm(temperature=0.2, max_tokens=8192, model_id=model_id)
        prompt = DOCUMENT_PROMPT.format(analysis=analysis_text)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        sections = json.loads(raw)
        log.info("Document: AI narrative complete")
        return sections
    except Exception as e:
        log.error("Document: AI narrative failed (%s) — using fallback", e)
        return {
            "title": f"Data Analysis Report: {dataset_name}",
            "executive_summary": f"This report provides a comprehensive analysis of the '{dataset_name}' dataset, covering data quality, distributions, correlations, and actionable business insights.",
            "data_overview": "See the overview statistics and data type distribution chart below for a complete picture of the dataset structure.",
            "quality_assessment": "Review the completeness and duplicate metrics in the overview section to assess data reliability.",
            "distribution_analysis": "Numeric column distributions are shown in the histogram charts below, revealing patterns in the data.",
            "correlation_analysis": "See the correlation heatmap for relationships between features that may indicate business drivers.",
            "missing_value_analysis": "Missing value patterns are visualized in the chart below.",
            "categorical_analysis": "Categorical column distributions reveal the key segments and groups in the data.",
            "outlier_analysis": "Box plots below show outlier distribution — these may represent premium segments or data errors.",
            "business_insights": "This dataset contains patterns that can inform business strategy. Review the distribution and correlation sections to identify key drivers and segments for targeted action.",
            "use_cases": [
                {"title": "Descriptive Analytics", "description": "Use this data to build dashboards and track key metrics over time.", "category": "operations"},
                {"title": "Customer Segmentation", "description": "Segment records by categorical and numeric features to identify high-value groups.", "category": "marketing"},
                {"title": "Predictive Modeling", "description": "Build ML models to predict key outcomes based on the available features.", "category": "research"},
            ],
            "market_analysis": "Review the categorical distributions and numeric patterns to understand market segments and demand drivers.",
            "promotion_strategies": [
                "Target the most frequent category segments with tailored promotions.",
                "Use outlier analysis to identify premium or underserved segments for special offers.",
                "Leverage correlation insights to bundle related products or features.",
            ],
            "recommendations": [
                "Review columns with high missing values before analysis.",
                "Check for and remove duplicate rows if appropriate.",
                "Consider transformations for highly skewed columns.",
                "Investigate strong correlations for business driver analysis.",
                "Build segmentation models based on categorical features.",
            ],
            "next_steps": [
                "Perform deeper segmentation analysis on key categorical columns.",
                "Collect additional time-series data if trend analysis is needed.",
                "Build a predictive model targeting the most relevant outcome column.",
            ],
            "conclusion": f"The '{dataset_name}' dataset has been profiled. Review the sections above for detailed findings.",
        }
