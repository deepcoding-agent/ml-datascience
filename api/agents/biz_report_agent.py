"""BizReportAgent — business-focused recommendation report.

Pipeline:
  1. Reuse the deep analysis + charts computed by document_agent (DRY).
  2. Feed the same analysis to a BUSINESS-focused LLM prompt that writes
     executive-style sections (insights, use cases, market reads, strategies,
     prioritized recommendations).
  3. Return the same document/artifact shape so the existing DocumentReportCard
     can render it — the difference is which keys the LLM populates.

Split from /eda so analysts and stakeholders each get a report tuned to them.
"""
from __future__ import annotations

import time
from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage

from api.agents.context_analyzer import analyze_context
from api.agents.document_agent import (
    _build_column_profiles,
    _categorical_balance,
    _chart_categorical,
    _chart_correlation,
    _chart_distribution,
    _chart_dtype_pie,
    _chart_missing,
    _chart_outlier_box,
    _datetime_summary,
    _detect_target_candidates,
    _format_for_llm,
    _high_cardinality_cats,
    _ml_readiness,
    _normalize_string_lists,
    _parse_llm_json,
)
from api.llm import get_llm
from api.logger import get_logger

log = get_logger(__name__)


BIZ_PROMPT = """\
You are a senior business consultant (think McKinsey / BCG partner) briefing
a CEO on what this dataset means for the business and what they should DO.
Technical EDA lives in a separate /eda report — you do NOT need to repeat
distributions, skewness, or modeling-readiness here. Focus on business value,
strategy, and action.

## Dataset Analysis
{analysis}

## Instructions
Write a JSON business report. Every claim must reference SPECIFIC numbers from
the analysis — segment counts, percentages, value ranges, dominant categories.
Generic advice is worthless; concrete strategy tied to this dataset is the goal.

IMPORTANT: Respond in the same language as the column names suggest.
If columns look Thai (ราคา, ชื่อ, etc.) → write in Thai.
If columns look English (price, name, etc.) → write in English.
Mixed → write in the dominant language.

Return EXACTLY this JSON (no markdown fences):
{{
  "title": "Business Strategy Report: <dataset name>",
  "executive_summary": "5-6 sentences: What this data represents in business terms. The 3 highest-leverage findings. The single #1 action item with expected impact. CEO-level only.",
  "business_context": "4-5 sentences: What industry/domain does this data reflect? What part of the value chain? What business decisions does it inform? Cite columns that hint at the domain.",
  "business_insights": "8-10 sentences: The KEY section. What does the data REVEAL about the business? Concentrations (Pareto-style), unusual segments, hidden risks, untapped opportunities. Each insight must cite real numbers (e.g. '17% of records carry 60% of the value'). Tell the story.",
  "key_segments": [
    {{"name": "segment name", "size": "X records (Y%)", "characteristic": "what makes them distinct", "value_signal": "why they matter to the business"}},
    {{"name": "...", "size": "...", "characteristic": "...", "value_signal": "..."}}
  ],
  "market_analysis": "5-7 sentences: What does the data say about market structure? Price/value tiers, customer or product segments, demand patterns, competitive whitespace. Reference exact value ranges and counts.",
  "use_cases": [
    {{"title": "specific business use case", "description": "2-3 sentences: HOW the business uses this data — which columns drive the decision, expected outcome", "category": "marketing|sales|operations|product|finance|customer"}},
    {{"title": "...", "description": "...", "category": "..."}}
  ],
  "opportunities": [
    "Opportunity 1 — specific revenue/cost lever with size estimate (cite the data)",
    "Opportunity 2 — ...",
    "Opportunity 3 — ..."
  ],
  "risks": [
    "Risk 1 — business risk visible in the data (e.g. customer concentration, quality decline). Cite the metric.",
    "Risk 2 — ...",
    "Risk 3 — ..."
  ],
  "promotion_strategies": [
    "Strategy 1: Target the X records (Y% of inventory) that share Z — proposed action with expected lift",
    "Strategy 2: ...",
    "Strategy 3: ..."
  ],
  "recommendations": [
    "Quick Win (1-2 weeks): clear ROI action, owner = function",
    "Short-term (1-3 months): initiative with measurable outcome",
    "Strategic (6-12 months): capability or infrastructure investment",
    "Quick Win 2: ...",
    "Short-term 2: ..."
  ],
  "suggested_kpis": [
    "KPI 1 — metric definition + why it matters + baseline from this data if computable",
    "KPI 2 — ...",
    "KPI 3 — ..."
  ],
  "roadmap": {{
    "next_30_days": "What to do in the first month — usually validation, pilot, or quick win",
    "next_90_days": "What to do in the first quarter — usually deploy strategy or build capability",
    "next_12_months": "What strategic shift this data enables"
  }},
  "conclusion": "3-4 sentences: Bottom line for the CEO. What is the strategic value of this data? What decision should be made this week?"
}}

Rules:
- Cite REAL column names, segment sizes, percentages, value ranges
- Every recommendation has a who, what, and rough timeline
- key_segments: 3-5 segments tied to specific data slices
- use_cases: 4-6 with category tags
- opportunities/risks: 3-5 each, concrete and quantified where possible
- promotion_strategies: 3-5 with specific data references
- NO technical jargon (skewness, multicollinearity, etc.) — /eda handles that
- NO generic advice — every sentence must be derivable from the analysis above
"""


def run_biz_report(
    data: list[dict],
    dataset_name: str,
    model_id: str | None = None,
) -> dict:
    """Generate a business strategy report.

    Returns the same shape as run_document so the existing DocumentReportCard
    renders it unchanged — only the LLM-populated section keys differ.
    """
    t0 = time.perf_counter()
    df = pd.DataFrame(data)
    log.info("Biz report: analyzing '%s' — %d x %d", dataset_name, *df.shape)

    ctx = analyze_context(df)

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

    # Reuse the same correlation + outlier + skew compute as document_agent so
    # both reports speak from identical underlying numbers.
    top_corr: list[dict] = []
    if len(ctx.numeric_cols) >= 2:
        try:
            corr = df[ctx.numeric_cols[:30]].corr()
            seen: set[tuple[str, str]] = set()
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

    outlier_summary: list[dict] = []
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

    analysis: dict[str, Any] = {
        "overview": overview,
        "correlations": top_corr,
        "skewed_columns": [],
        "outlier_summary": outlier_summary[:10],
        "null_columns": ctx.null_cols,
        "high_null_columns": ctx.high_null_cols,
        "constant_columns": ctx.constant_cols,
        "warnings": ctx.warnings,
    }
    column_profiles = _build_column_profiles(df)
    analysis["target_candidates"]      = _detect_target_candidates(df, column_profiles)
    analysis["categorical_balance"]    = _categorical_balance(df, ctx.categorical_cols)
    analysis["high_cardinality_cats"]  = _high_cardinality_cats(df, ctx.categorical_cols)
    analysis["datetime_summary"]       = _datetime_summary(df, ctx.datetime_cols)
    analysis["multicollinear_pairs"]   = [c for c in top_corr if abs(c["value"]) > 0.9]
    analysis["ml_readiness"]           = _ml_readiness(df, analysis, ctx.numeric_cols)

    charts: dict[str, str | None] = {
        "distribution": _chart_distribution(df, ctx.numeric_cols),
        "correlation":  _chart_correlation(df, ctx.numeric_cols),
        "missing":      _chart_missing(df),
        "outlier_box":  _chart_outlier_box(df, ctx.numeric_cols),
        "categorical":  _chart_categorical(df, ctx.categorical_cols),
        "dtype_pie":    _chart_dtype_pie(df),
    }
    charts = {k: v for k, v in charts.items() if v is not None}

    analysis_text = _format_for_llm(analysis, column_profiles)
    sections = _biz_narrative(analysis_text, dataset_name, model_id)

    # Reuse the same quality score calc — biz audiences benefit from knowing it too.
    score = 100
    score -= min(30, int((100 - completeness) * 0.5))
    score -= min(20, int(overview["duplicate_pct"] * 2))
    score -= min(10, len(analysis.get("constant_columns", [])) * 3)
    score -= min(10, len(analysis.get("high_null_columns", [])) * 5)
    score = max(0, score)
    quality_label = "Excellent" if score >= 85 else "Good" if score >= 70 else "Fair" if score >= 50 else "Poor"

    elapsed = round(time.perf_counter() - t0, 2)
    log.info("Biz report done: '%s' quality=%d/%s in %.1fs", dataset_name, score, quality_label, elapsed)

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


def _biz_narrative(analysis_text: str, dataset_name: str, model_id: str | None) -> dict:
    """LLM narrator for the business report. Falls back to a stub on failure."""
    try:
        llm = get_llm(temperature=0.3, max_tokens=8192, model_id=model_id)
        prompt = BIZ_PROMPT.format(analysis=analysis_text)
        response = llm.invoke([HumanMessage(content=prompt)])
        sections = _normalize_string_lists(_parse_llm_json(response.content))
        log.info("Biz narrative parsed: %d sections", len(sections))
        return sections
    except Exception as exc:
        log.warning("Biz narrative LLM failed (%s) — using fallback", exc)
        return {
            "title": f"Business Strategy Report: {dataset_name}",
            "executive_summary": (
                "Business narrative could not be generated automatically. "
                "Please retry or use the technical /eda report for raw findings."
            ),
            "business_context": "",
            "business_insights": "",
            "key_segments": [],
            "market_analysis": "",
            "use_cases": [],
            "opportunities": [],
            "risks": [],
            "promotion_strategies": [],
            "recommendations": [],
            "suggested_kpis": [],
            "roadmap": {"next_30_days": "", "next_90_days": "", "next_12_months": ""},
            "conclusion": "",
        }
