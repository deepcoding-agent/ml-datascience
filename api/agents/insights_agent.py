"""Insights Agent — AI-powered dataset analysis report.

Flow:
  1. Deterministic analysis — compute stats, distributions, correlations, quality
  2. AI interpretation — LLM reads the analysis and produces narrative insights
  3. Structured output — sections with key findings, recommendations, highlights
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from api.agents.context_analyzer import analyze_context
from api.llm import get_llm
from api.logger import get_logger

log = get_logger(__name__)

# ── AI interpretation prompt ─────────────────────────────────────────────────

INSIGHTS_PROMPT = """\
You are an expert data analyst writing a dataset insights report.
Given the analysis below, produce a structured JSON insights report.

## Dataset Analysis
{analysis}

## Output — valid JSON only, no markdown fences

Return EXACTLY this structure:
{{
  "summary": "2-3 sentence executive summary of the dataset — what it appears to contain, its quality, and its readiness for analysis.",
  "data_quality_score": <integer 0-100>,
  "data_quality_label": "Excellent|Good|Fair|Poor",
  "highlights": [
    {{"icon": "star|warning|trend|link|target|bulb", "title": "short title", "detail": "1-2 sentence finding"}}
  ],
  "recommendations": [
    "actionable recommendation 1",
    "actionable recommendation 2"
  ],
  "interesting_columns": [
    {{"column": "col_name", "reason": "why this column is interesting"}}
  ],
  "potential_targets": [
    {{"column": "col_name", "reason": "why this could be a target variable", "task_type": "classification|regression"}}
  ]
}}

Rules:
- highlights: 4-8 items, mix of positive findings and warnings
- recommendations: 3-6 actionable items
- interesting_columns: 3-5 most interesting columns
- potential_targets: 1-3 columns that could be prediction targets
- data_quality_score: 0-100 based on completeness, duplicates, type consistency, outliers
- Be specific: use actual column names, actual numbers, actual percentages
- Write concisely — no filler words
"""


def _compute_analysis(df: pd.DataFrame) -> dict:
    """Compute comprehensive deterministic analysis of the dataset."""
    ctx = analyze_context(df)

    # ── Basic info ─────────────────────────────────────────────────
    info = {
        "shape": {"rows": df.shape[0], "cols": df.shape[1]},
        "memory_mb": ctx.memory_mb,
        "dtypes": {},
        "column_types": {
            "numeric": ctx.numeric_cols,
            "categorical": ctx.categorical_cols,
            "datetime": ctx.datetime_cols,
        },
    }

    # Dtype breakdown
    for dtype, count in df.dtypes.value_counts().items():
        info["dtypes"][str(dtype)] = int(count)

    # ── Quality metrics ────────────────────────────────────────────
    total_cells = df.shape[0] * df.shape[1]
    null_cells = int(df.isnull().sum().sum())
    completeness = round((1 - null_cells / total_cells) * 100, 1) if total_cells > 0 else 100
    dupe_pct = round(ctx.duplicate_count / df.shape[0] * 100, 1) if df.shape[0] > 0 else 0

    quality = {
        "completeness_pct": completeness,
        "total_nulls": null_cells,
        "null_columns": ctx.null_cols,
        "high_null_columns": ctx.high_null_cols,
        "duplicate_rows": ctx.duplicate_count,
        "duplicate_pct": dupe_pct,
        "constant_columns": ctx.constant_cols,
    }

    # ── Numeric statistics ─────────────────────────────────────────
    numeric_stats = {}
    for col in ctx.numeric_cols[:20]:
        try:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            stats = {
                "mean": round(float(s.mean()), 4),
                "median": round(float(s.median()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
                "skewness": round(float(s.skew()), 4),
                "null_pct": round(float(df[col].isnull().mean() * 100), 1),
                "unique_count": int(s.nunique()),
            }
            # Outlier count (IQR)
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
                stats["outlier_count"] = outliers
                stats["outlier_pct"] = round(outliers / len(s) * 100, 1)
            numeric_stats[col] = stats
        except Exception:
            pass

    # ── Categorical statistics ─────────────────────────────────────
    cat_stats = {}
    for col in ctx.categorical_cols[:20]:
        try:
            vc = df[col].value_counts()
            uniq = int(df[col].nunique())
            cat_stats[col] = {
                "unique_count": uniq,
                "top_value": str(vc.index[0]) if len(vc) > 0 else None,
                "top_count": int(vc.iloc[0]) if len(vc) > 0 else 0,
                "top_pct": round(float(vc.iloc[0] / len(df) * 100), 1) if len(vc) > 0 else 0,
                "null_pct": round(float(df[col].isnull().mean() * 100), 1),
                "is_binary": uniq == 2,
                "is_high_cardinality": uniq > 50,
            }
        except Exception:
            pass

    # ── Correlations ───────────────────────────────────────────────
    top_correlations = []
    if len(ctx.numeric_cols) >= 2:
        try:
            corr = df[ctx.numeric_cols[:30]].corr()
            pairs_seen = set()
            for i, c1 in enumerate(corr.columns):
                for j, c2 in enumerate(corr.columns):
                    if i >= j:
                        continue
                    val = corr.iloc[i, j]
                    if pd.notna(val) and abs(val) > 0.3:
                        pair = tuple(sorted([c1, c2]))
                        if pair not in pairs_seen:
                            pairs_seen.add(pair)
                            top_correlations.append({
                                "col1": c1,
                                "col2": c2,
                                "value": round(float(val), 4),
                                "strength": "strong" if abs(val) > 0.7 else "moderate",
                            })
            top_correlations.sort(key=lambda x: abs(x["value"]), reverse=True)
            top_correlations = top_correlations[:15]
        except Exception:
            pass

    # ── Distribution overview ──────────────────────────────────────
    skewed_details = []
    for col in ctx.skewed_cols[:10]:
        try:
            skew_val = float(df[col].skew())
            direction = "right-skewed" if skew_val > 0 else "left-skewed"
            skewed_details.append({
                "column": col,
                "skewness": round(skew_val, 2),
                "direction": direction,
            })
        except Exception:
            pass

    return {
        "info": info,
        "quality": quality,
        "numeric_stats": numeric_stats,
        "categorical_stats": cat_stats,
        "top_correlations": top_correlations,
        "skewed_columns": skewed_details,
        "warnings": ctx.warnings,
        "suggested_target": ctx.suggested_target,
    }


def _format_analysis_for_llm(analysis: dict) -> str:
    """Format the analysis dict as a readable string for the LLM."""
    lines = []

    info = analysis["info"]
    lines.append(f"Shape: {info['shape']['rows']:,} rows x {info['shape']['cols']} columns")
    lines.append(f"Memory: {info['memory_mb']:.2f} MB")
    lines.append(f"Column types: {info['dtypes']}")
    lines.append(f"Numeric columns ({len(info['column_types']['numeric'])}): {info['column_types']['numeric'][:15]}")
    lines.append(f"Categorical columns ({len(info['column_types']['categorical'])}): {info['column_types']['categorical'][:15]}")
    if info["column_types"]["datetime"]:
        lines.append(f"Datetime columns: {info['column_types']['datetime']}")

    q = analysis["quality"]
    lines.append(f"\nData completeness: {q['completeness_pct']}%")
    lines.append(f"Total nulls: {q['total_nulls']:,}")
    if q["null_columns"]:
        lines.append(f"Null columns: { {k: f'{v}%' for k, v in sorted(q['null_columns'].items(), key=lambda x: -x[1])[:10]} }")
    if q["high_null_columns"]:
        lines.append(f"High null (>40%): {q['high_null_columns']}")
    lines.append(f"Duplicate rows: {q['duplicate_rows']:,} ({q['duplicate_pct']}%)")
    if q["constant_columns"]:
        lines.append(f"Constant columns: {q['constant_columns']}")

    if analysis["numeric_stats"]:
        lines.append("\nNumeric column statistics:")
        for col, stats in list(analysis["numeric_stats"].items())[:15]:
            parts = [f"mean={stats['mean']}", f"std={stats['std']}", f"range=[{stats['min']}, {stats['max']}]"]
            if abs(stats["skewness"]) > 1:
                parts.append(f"skew={stats['skewness']}")
            if stats.get("outlier_pct", 0) > 1:
                parts.append(f"outliers={stats['outlier_count']}({stats['outlier_pct']}%)")
            if stats["null_pct"] > 0:
                parts.append(f"null={stats['null_pct']}%")
            lines.append(f"  {col}: {', '.join(parts)}")

    if analysis["categorical_stats"]:
        lines.append("\nCategorical column statistics:")
        for col, stats in list(analysis["categorical_stats"].items())[:15]:
            parts = [f"unique={stats['unique_count']}"]
            if stats["top_value"]:
                parts.append(f"top='{stats['top_value']}'({stats['top_pct']}%)")
            if stats["is_binary"]:
                parts.append("binary")
            if stats["is_high_cardinality"]:
                parts.append("high-cardinality")
            if stats["null_pct"] > 0:
                parts.append(f"null={stats['null_pct']}%")
            lines.append(f"  {col}: {', '.join(parts)}")

    if analysis["top_correlations"]:
        lines.append("\nTop correlations:")
        for c in analysis["top_correlations"][:10]:
            lines.append(f"  {c['col1']} <-> {c['col2']}: {c['value']} ({c['strength']})")

    if analysis["skewed_columns"]:
        lines.append(f"\nSkewed columns: {[(s['column'], s['skewness']) for s in analysis['skewed_columns']]}")

    if analysis["warnings"]:
        lines.append(f"\nWarnings: {analysis['warnings']}")

    if analysis["suggested_target"]:
        lines.append(f"\nSuggested target column: {analysis['suggested_target']}")

    return "\n".join(lines)


# ── Main entry point ─────────────────────────────────────────────────────────

def run_insights(
    data: list[dict],
    dataset_name: str,
    model_id: str | None = None,
) -> dict:
    """Generate a comprehensive insights report for a dataset.

    Returns dict with:
      success, dataset_name, analysis (deterministic stats),
      ai_insights (LLM interpretation), report (markdown)
    """
    t0 = time.perf_counter()
    df = pd.DataFrame(data)
    log.info("Insights: analyzing '%s' — %d x %d", dataset_name, *df.shape)

    # ── 1. Deterministic analysis ──────────────────────────────────
    analysis = _compute_analysis(df)

    # ── 2. AI interpretation ───────────────────────────────────────
    analysis_text = _format_analysis_for_llm(analysis)
    ai_insights = None

    try:
        llm = get_llm(temperature=0.1, max_tokens=2048, model_id=model_id)
        prompt = INSIGHTS_PROMPT.format(analysis=analysis_text)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Strip markdown fencing
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        ai_insights = json.loads(raw)
        log.info("Insights: AI interpretation complete — quality=%s",
                 ai_insights.get("data_quality_score"))
    except Exception as e:
        log.error("Insights: AI interpretation failed (%s) — using fallback", e)
        ai_insights = _fallback_insights(analysis, dataset_name)

    # ── 3. Build markdown report ───────────────────────────────────
    elapsed = time.perf_counter() - t0
    report = _build_report(dataset_name, analysis, ai_insights, elapsed)

    log.info("Insights done: '%s' quality=%s in %.1fs",
             dataset_name, ai_insights.get("data_quality_score"), elapsed)

    return {
        "success": True,
        "dataset_name": dataset_name,
        "analysis": analysis,
        "ai_insights": ai_insights,
        "report": report,
        "duration_seconds": round(elapsed, 2),
    }


def _fallback_insights(analysis: dict, dataset_name: str) -> dict:
    """Deterministic fallback if AI fails."""
    q = analysis["quality"]
    completeness = q["completeness_pct"]

    # Score: 100 base, subtract for issues
    score = 100
    score -= min(30, int((100 - completeness) * 0.5))
    score -= min(20, int(q["duplicate_pct"] * 2))
    score -= min(10, len(q["constant_columns"]) * 3)
    score -= min(10, len(q["high_null_columns"]) * 5)
    score = max(0, score)

    label = "Excellent" if score >= 85 else "Good" if score >= 70 else "Fair" if score >= 50 else "Poor"

    highlights = []
    info = analysis["info"]
    highlights.append({
        "icon": "star",
        "title": f"{info['shape']['rows']:,} rows, {info['shape']['cols']} columns",
        "detail": f"Dataset contains {info['shape']['rows']:,} records across {info['shape']['cols']} features.",
    })
    if completeness >= 95:
        highlights.append({"icon": "star", "title": "High completeness", "detail": f"{completeness}% of all cells have values."})
    if q["duplicate_rows"] > 0:
        highlights.append({"icon": "warning", "title": f"{q['duplicate_rows']:,} duplicates", "detail": f"{q['duplicate_pct']}% of rows are duplicated."})
    if q["high_null_columns"]:
        highlights.append({"icon": "warning", "title": "High-null columns", "detail": f"{len(q['high_null_columns'])} columns have >40% missing values."})
    if analysis["top_correlations"]:
        top = analysis["top_correlations"][0]
        highlights.append({"icon": "link", "title": "Strong correlation found", "detail": f"{top['col1']} and {top['col2']} have correlation {top['value']}."})

    return {
        "summary": f"Dataset '{dataset_name}' has {info['shape']['rows']:,} rows and {info['shape']['cols']} columns with {completeness}% completeness.",
        "data_quality_score": score,
        "data_quality_label": label,
        "highlights": highlights,
        "recommendations": ["Review columns with missing values.", "Check for duplicate rows.", "Consider feature engineering on skewed columns."],
        "interesting_columns": [],
        "potential_targets": [],
    }


def _build_report(dataset_name: str, analysis: dict, ai: dict, elapsed: float) -> str:
    """Build the markdown report from analysis + AI insights."""
    lines = [
        f"## Dataset Insights — {dataset_name}",
        "",
        ai.get("summary", ""),
        "",
        f"**Data Quality:** {ai.get('data_quality_score', '?')}/100 ({ai.get('data_quality_label', 'N/A')})",
        "",
    ]

    # Overview table
    info = analysis["info"]
    q = analysis["quality"]
    lines.extend([
        "| Metric | Value |",
        "|--------|-------|",
        f"| Rows | {info['shape']['rows']:,} |",
        f"| Columns | {info['shape']['cols']} |",
        f"| Completeness | {q['completeness_pct']}% |",
        f"| Duplicates | {q['duplicate_rows']:,} ({q['duplicate_pct']}%) |",
        f"| Memory | {info['memory_mb']:.2f} MB |",
        "",
    ])

    # Highlights
    highlights = ai.get("highlights", [])
    if highlights:
        lines.append("### Key Findings")
        lines.append("")
        for h in highlights:
            lines.append(f"- **{h['title']}** — {h['detail']}")
        lines.append("")

    # Interesting columns
    interesting = ai.get("interesting_columns", [])
    if interesting:
        lines.append("### Interesting Columns")
        lines.append("")
        for ic in interesting:
            lines.append(f"- **{ic['column']}** — {ic['reason']}")
        lines.append("")

    # Potential targets
    targets = ai.get("potential_targets", [])
    if targets:
        lines.append("### Potential Target Variables")
        lines.append("")
        for t in targets:
            lines.append(f"- **{t['column']}** ({t.get('task_type', 'unknown')}) — {t['reason']}")
        lines.append("")

    # Top correlations
    if analysis["top_correlations"]:
        lines.append("### Top Correlations")
        lines.append("")
        lines.append("| Column A | Column B | Correlation | Strength |")
        lines.append("|----------|----------|------------|----------|")
        for c in analysis["top_correlations"][:8]:
            lines.append(f"| {c['col1']} | {c['col2']} | {c['value']:+.3f} | {c['strength']} |")
        lines.append("")

    # Recommendations
    recs = ai.get("recommendations", [])
    if recs:
        lines.append("### Recommendations")
        lines.append("")
        for i, r in enumerate(recs, 1):
            lines.append(f"{i}. {r}")
        lines.append("")

    lines.append(f"_Analysis completed in {elapsed:.1f}s_")
    return "\n".join(lines)
