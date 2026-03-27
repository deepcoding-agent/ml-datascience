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
You are a business intelligence consultant presenting findings to a CEO.
Your job: turn raw data analysis into ACTIONABLE BUSINESS INTELLIGENCE.

## Dataset Analysis
{analysis}

## Instructions
Write a JSON report that a non-technical executive can read and immediately act on.
Focus on: revenue impact, cost savings, customer insights, market opportunities, and risk areas.
Use actual numbers from the analysis. Every insight must suggest a concrete action.

IMPORTANT: Respond in the same language as the column names suggest.
If columns look Thai (ราคา, ชื่อ, etc.) → write in Thai.
If columns look English (price, name, etc.) → write in English.
Mixed → write in the dominant language.

Return EXACTLY this JSON (no markdown fences):
{{
  "title": "Business Intelligence Report: <dataset name>",
  "executive_summary": "5-6 sentences: What is this data? What are the 3 most important findings? What is the #1 action item? Frame everything in business value (revenue, cost, customers, growth).",
  "data_overview": "2-3 sentences: What kind of business data is this? How many records? What business dimensions does it cover?",
  "quality_assessment": "3-4 sentences: Is this data trustworthy for decision-making? What data gaps could lead to wrong decisions? Rate: Ready/Needs Cleaning/Unreliable.",
  "distribution_analysis": "4-5 sentences: What do the numbers tell us about the business? Are there concentrations (e.g., 80% of revenue from 20% of products)? Price ranges? Volume patterns? Translate every stat into a business implication.",
  "correlation_analysis": "4-5 sentences: What drives what? If X goes up, does Y go up? Which factors most influence the key business metrics? What levers can the business pull?",
  "missing_value_analysis": "2-3 sentences: Are there data collection gaps? Could missing data cause blind spots in decision-making?",
  "categorical_analysis": "3-4 sentences: What segments exist? Which segments are largest/most profitable? Are there underserved segments with growth potential?",
  "outlier_analysis": "3-4 sentences: Are outliers premium opportunities (VIP customers, luxury products) or problems (errors, fraud)? What special handling do they need?",
  "business_insights": "6-8 sentences: The KEY section. Write like a consultant briefing the CEO. What story does the data tell? What are the top 3 strategic insights? What competitive advantages can be derived? What risks should leadership be aware of?",
  "use_cases": [
    {{"title": "specific use case", "description": "2-3 sentences: exactly how to use this data to generate revenue, reduce cost, or improve operations. Include which columns and what analysis.", "category": "marketing|sales|operations|product|finance|research"}},
    {{"title": "...", "description": "...", "category": "..."}}
  ],
  "market_analysis": "4-5 sentences: What does the data reveal about the market? Customer segments? Price sensitivity? Demand patterns? Competitive positioning? Geographic or demographic trends?",
  "promotion_strategies": [
    "Strategy 1: Reference specific segments/values from the data. e.g., 'Target the 95 properties with 4+ bedrooms (17% of inventory) with premium staging services — they average 45% higher prices'",
    "Strategy 2: Another data-driven strategy with specific numbers",
    "Strategy 3: ..."
  ],
  "recommendations": [
    "Priority 1 (Quick Win): Something achievable in 1-2 weeks with clear ROI",
    "Priority 2 (Short-term): 1-3 month initiative",
    "Priority 3 (Strategic): Longer-term data infrastructure or capability",
    "Priority 4: ...",
    "Priority 5: ..."
  ],
  "next_steps": [
    "Immediate: What analysis should be done next to validate these findings",
    "Data gap: What additional data would unlock more value",
    "ML opportunity: What prediction or classification could automate decisions"
  ],
  "conclusion": "3-4 sentences: Bottom line for the CEO. What is the strategic value of this data? What decision should be made today?"
}}

Rules:
- Use ACTUAL column names, numbers, and percentages from the analysis
- Frame everything as business impact: revenue, cost, customers, growth, risk
- Every recommendation must be actionable — who does what by when
- use_cases: 4-6 realistic use cases with ROI potential
- promotion_strategies: 3-5 strategies with SPECIFIC numbers from the data
- Think like McKinsey presenting to a Fortune 500 CEO
- NO vague platitudes — every sentence must reference specific data
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

    # ── 3. Column profiles ───────────────────────────────────────────
    column_profiles = _build_column_profiles(df)

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

    return "\n".join(lines)


def _ai_narrative(analysis_text: str, dataset_name: str, model_id: str | None) -> dict:
    """Use LLM to write narrative sections, fallback on failure."""
    try:
        llm = get_llm(temperature=0.2, max_tokens=4096, model_id=model_id)
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
