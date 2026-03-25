"""Auto-Prepare Agent — AI-driven ML data preparation.

Flow:
  1. Analyze dataset + target column
  2. AI decides optimal PrepConfig (missing strategy, scaling, encoding, etc.)
  3. Execute the existing 10-step pipeline with AI-chosen config
  4. Return X_train, X_test, y_train, y_test + metadata
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from api.agents.context_analyzer import analyze_context
from api.data_preparation_agent import run_data_preparation
from api.llm import get_llm
from api.logger import get_logger
from api.models import PrepConfig

log = get_logger(__name__)

# ── AI planning prompt ────────────────────────────────────────────────────────

AUTO_PREPARE_PROMPT = """\
You are an ML data preparation specialist. Given a dataset analysis and target
column, decide the optimal preprocessing configuration.

## Dataset Analysis
{analysis}

## Target Column
{target_info}

## Task Classification
{task_type_hint}

## Configuration Options
| field | options | default | when to change |
|-------|---------|---------|----------------|
| missing_strategy | auto/mean/median/mode/drop | auto | "drop" if <5% missing rows; "median" if heavy skew; "mode" for categorical-heavy |
| scaling_method | standard/minmax/robust/none | standard | "robust" if many outliers; "minmax" for tree models; "none" if data already normalized |
| encoding_method | auto/onehot/label/ordinal | auto | "label" if high-cardinality cats; "onehot" if <10 categories; "auto" picks per column |
| outlier_treatment | iqr/zscore/none | iqr | "none" for tree-based models; "zscore" for normally distributed data |
| outlier_threshold | 0.5-5.0 | 1.5 | lower=more aggressive clipping; higher=more permissive |
| correlation_threshold | 0.5-1.0 | 0.95 | lower=remove more correlated features; 0.8 for strict selection |
| test_size | 0.1-0.3 | 0.2 | 0.1 for small datasets (<500 rows); 0.2-0.3 for larger |
| drop_threshold | 0.1-0.9 | 0.4 | lower=stricter about missing cols; 0.3 for important features |

## Decision rules
1. For classification tasks with imbalanced classes: use test_size=0.2, stratified split handles balance
2. For regression with skewed features: use missing_strategy="median", scaling_method="robust"
3. For small datasets (<500 rows): use test_size=0.15, outlier_treatment="none" (don't lose rows)
4. For high-cardinality categoricals: use encoding_method="label" or "auto"
5. For datasets with many outliers: use scaling_method="robust", outlier_treatment="iqr"
6. For already clean data (no nulls, no outliers): simplify — focus on encoding and scaling

## Output — ONLY valid JSON, no markdown, no explanation
{{
  "missing_strategy": "auto",
  "scaling_method": "standard",
  "encoding_method": "auto",
  "outlier_treatment": "iqr",
  "outlier_threshold": 1.5,
  "correlation_threshold": 0.95,
  "test_size": 0.2,
  "drop_threshold": 0.4,
  "reasoning": "one sentence explaining the choices"
}}
"""


def _build_prepare_analysis(df: pd.DataFrame, ctx, target: str) -> str:
    """Build analysis text for the AI planner."""
    lines = [
        f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns",
        f"Memory: {ctx.memory_mb:.2f} MB",
        f"Duplicate rows: {ctx.duplicate_count:,}",
        f"Numeric columns ({len(ctx.numeric_cols)}): {ctx.numeric_cols[:10]}",
        f"Categorical columns ({len(ctx.categorical_cols)}): {ctx.categorical_cols[:10]}",
    ]

    # Null summary
    if ctx.null_cols:
        total_null_pct = sum(ctx.null_cols.values()) / len(ctx.null_cols)
        lines.append(f"Null columns: {len(ctx.null_cols)}, avg {total_null_pct:.1f}%")
        high = [(c, p) for c, p in ctx.null_cols.items() if p > 20]
        if high:
            lines.append(f"  High-null: {', '.join(f'{c}({p:.0f}%)' for c, p in high[:5])}")
    else:
        lines.append("No null values")

    # Skew
    if ctx.skewed_cols:
        lines.append(f"Highly skewed ({len(ctx.skewed_cols)}): {ctx.skewed_cols[:5]}")

    # Cardinality
    cat_card = {}
    for c in ctx.categorical_cols[:10]:
        cat_card[c] = int(df[c].nunique())
    if cat_card:
        lines.append(f"Categorical cardinality: {cat_card}")

    # Outlier count
    outlier_cols = []
    for c in ctx.numeric_cols[:15]:
        if c == target:
            continue
        try:
            q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                n = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
                if n > 0:
                    outlier_cols.append(f"{c}({n})")
        except Exception:
            pass
    if outlier_cols:
        lines.append(f"Outlier columns: {', '.join(outlier_cols[:8])}")

    return "\n".join(lines)


def _build_target_info(df: pd.DataFrame, target: str) -> tuple[str, str]:
    """Build target column analysis and task type hint."""
    col = df[target]
    nunique = int(col.nunique())
    null_pct = col.isnull().mean() * 100

    if col.dtype in ("object", "category") or nunique <= 20:
        task_type = "classification"
        top_vals = col.value_counts().head(5).to_dict()
        target_info = (
            f"Column: {target}\n"
            f"Type: categorical ({nunique} classes)\n"
            f"Null: {null_pct:.1f}%\n"
            f"Class distribution: {top_vals}"
        )
        # Check balance
        freqs = col.value_counts(normalize=True)
        imbalance = freqs.max() / freqs.min() if freqs.min() > 0 else float("inf")
        hint = f"Classification task with {nunique} classes"
        if imbalance > 5:
            hint += " (imbalanced)"
        else:
            hint += " (balanced)"
    else:
        task_type = "regression"
        target_info = (
            f"Column: {target}\n"
            f"Type: numeric (continuous)\n"
            f"Null: {null_pct:.1f}%\n"
            f"Stats: mean={col.mean():.2f}, std={col.std():.2f}, "
            f"min={col.min():.2f}, max={col.max():.2f}, skew={col.skew():.2f}"
        )
        hint = f"Regression task, target range [{col.min():.1f}, {col.max():.1f}]"
        if abs(col.skew()) > 1:
            hint += " (skewed target)"

    return target_info, hint


def _default_config(df: pd.DataFrame, ctx, target: str) -> PrepConfig:
    """Deterministic fallback config if AI planning fails."""
    n_rows = df.shape[0]
    config = PrepConfig()

    # Small dataset — don't lose data
    if n_rows < 500:
        config.test_size = 0.15
        config.outlier_treatment = "none"

    # Heavy skew
    if len(ctx.skewed_cols) > len(ctx.numeric_cols) * 0.5:
        config.missing_strategy = "median"
        config.scaling_method = "robust"

    # High cardinality
    high_card = sum(1 for c in ctx.categorical_cols if df[c].nunique() > 10)
    if high_card > 2:
        config.encoding_method = "label"

    return config


def run_auto_prepare(
    data: list[dict],
    target_column: str | None = None,
    model_id: str | None = None,
) -> dict:
    """AI-driven ML data preparation.

    1. Analyze dataset + target
    2. AI picks optimal PrepConfig
    3. Run existing 10-step pipeline
    4. Return results with metadata
    """
    df = pd.DataFrame(data)

    # Resolve target
    if not target_column or target_column not in df.columns:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        target_column = num_cols[-1] if num_cols else df.columns[-1]

    ctx = analyze_context(df)
    log.info("Auto-prepare: %dx%d, target='%s'", *df.shape, target_column)

    # ── AI decides config ─────────────────────────────────────────────────────
    analysis = _build_prepare_analysis(df, ctx, target_column)
    target_info, task_hint = _build_target_info(df, target_column)

    try:
        llm = get_llm(temperature=0.0, max_tokens=512, model_id=model_id)
        prompt = AUTO_PREPARE_PROMPT.format(
            analysis=analysis,
            target_info=target_info,
            task_type_hint=task_hint,
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        ai_config = json.loads(raw)
        reasoning = ai_config.pop("reasoning", "")
        config = PrepConfig(**{k: v for k, v in ai_config.items() if k in PrepConfig.model_fields})
        log.info("Auto-prepare AI config: %s — %s", config.model_dump(), reasoning)
    except Exception as e:
        log.error("Auto-prepare AI planning failed (%s) — using fallback", e)
        config = _default_config(df, ctx, target_column)
        reasoning = "Fallback: deterministic config based on data analysis"

    # ── Run existing pipeline ─────────────────────────────────────────────────
    result = run_data_preparation(
        data=data,
        target_column=target_column,
        mode="full",
        config=config,
    )

    # Attach AI reasoning and config to result
    result["ai_config"] = config.model_dump()
    result["ai_reasoning"] = reasoning
    result["task_type"] = task_hint.split(",")[0] if task_hint else "unknown"

    return result
