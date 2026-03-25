"""Auto-Cleaner Agent — AI-driven dataset cleaning.

Flow:
  1. Deterministic analysis — identify all data issues
  2. AI planning — LLM decides which cleaning operations to apply
  3. Execute — run clean handlers sequentially on the DataFrame
  4. Report — build before/after comparison with detailed operation log
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from api.agents.context_analyzer import analyze_context
from api.handlers import HANDLER_REGISTRY
from api.llm import get_llm
from api.logger import get_logger

log = get_logger(__name__)

# ── Cleaning planner prompt ───────────────────────────────────────────────────

AUTO_CLEAN_PROMPT = """\
You are a data cleaning specialist. Analyze the dataset issues below and decide
which cleaning operations to apply. Be thorough but conservative — only include
operations that are actually needed based on the analysis.

## Dataset Analysis
{analysis}

## Available Operations (use ONLY these IDs)
| id | what it does | params |
|----|-------------|--------|
| clean.fix_dtypes | auto-convert strings that look numeric/datetime | (none) |
| clean.strip_whitespace | trim leading/trailing whitespace from all strings | (none) |
| clean.drop_constant | drop columns where all values are identical (zero info) | (none) |
| clean.drop_id_columns | auto-detect and drop ID-like columns (sequential, UUIDs, generic names) | (none) |
| clean.remove_duplicates | remove duplicate rows | (none) |
| clean.drop_nulls | drop columns exceeding null threshold | threshold (float 0-1, e.g. 0.4 = drop cols >40% null) |
| clean.fill_nulls | fill remaining missing values with smart strategy | strategy (auto/median/mean/mode/zero) |
| clean.clip_outliers | cap extreme outlier values using IQR bounds | method (iqr) |
| clean.lowercase_columns | normalize column names to snake_case | (none) |

## Decision rules — READ CAREFULLY
1. ALWAYS start with clean.fix_dtypes if any string columns might be numeric
2. Include clean.strip_whitespace if there are string/object columns
3. Include clean.drop_constant ONLY if constant columns exist in analysis
4. Include clean.drop_id_columns ONLY if ID-like columns are mentioned
5. Include clean.remove_duplicates ONLY if duplicate_count > 0
6. Include clean.drop_nulls ONLY if analysis shows columns with >40% null
7. Include clean.fill_nulls with strategy="auto" if nulls remain after dropping
8. Include clean.clip_outliers ONLY if outlier columns are mentioned and significant
9. Include clean.lowercase_columns ONLY if column names use CamelCase/mixed case
10. SKIP any operation where the issue does not exist in the analysis

## Correct ordering (follow this sequence, skip irrelevant ones)
fix_dtypes → strip_whitespace → drop_constant → drop_id_columns →
remove_duplicates → drop_nulls → fill_nulls → clip_outliers → lowercase_columns

## Output — ONLY valid JSON array, no markdown fences, no explanation
[{{"id":"clean.fix_dtypes","params":{{}},"reason":"3 string cols contain numeric values"}}]
"""


def _build_analysis_text(df: pd.DataFrame, ctx) -> str:
    """Build a human-readable analysis of dataset issues for the LLM."""
    lines = [
        f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns",
        f"Memory: {ctx.memory_mb:.2f} MB",
        f"Duplicate rows: {ctx.duplicate_count:,}",
        f"Numeric columns ({len(ctx.numeric_cols)}): {ctx.numeric_cols[:10]}",
        f"Categorical/string columns ({len(ctx.categorical_cols)}): {ctx.categorical_cols[:10]}",
        f"Datetime columns ({len(ctx.datetime_cols)}): {ctx.datetime_cols}",
    ]

    # Null analysis
    if ctx.null_cols:
        lines.append(f"\nNull columns ({len(ctx.null_cols)}):")
        for col, pct in sorted(ctx.null_cols.items(), key=lambda x: -x[1])[:15]:
            lines.append(f"  {col}: {pct:.1f}% null")
    else:
        lines.append("\nNo null values found.")

    if ctx.high_null_cols:
        lines.append(f"\nHigh-null columns (>40%): {ctx.high_null_cols}")

    # Constant columns
    if ctx.constant_cols:
        lines.append(f"\nConstant columns (1 unique value): {ctx.constant_cols}")
    else:
        lines.append("\nNo constant columns.")

    # Skewed columns
    if ctx.skewed_cols:
        lines.append(f"\nHighly skewed columns (|skew|>1): {ctx.skewed_cols[:10]}")

    # ID-like columns
    id_like = _detect_id_columns(df)
    if id_like:
        lines.append(f"\nPotential ID columns: {id_like}")
    else:
        lines.append("\nNo ID-like columns detected.")

    # Column name style
    has_mixed = any(c != c.lower() and "_" not in c for c in df.columns)
    if has_mixed:
        lines.append("\nColumn names include CamelCase or mixed case.")
    else:
        lines.append("\nColumn names are already lowercase/snake_case.")

    # String columns that might be numeric
    suspect = _detect_suspect_numeric(df, ctx.categorical_cols)
    if suspect:
        lines.append(f"\nString columns that look numeric: {suspect}")

    # Outlier overview
    outlier_info = _detect_outliers(df, ctx.numeric_cols)
    if outlier_info:
        lines.append(f"\nOutlier columns (IQR): {', '.join(outlier_info[:10])}")
    else:
        lines.append("\nNo significant outliers detected.")

    return "\n".join(lines)


def _detect_id_columns(df: pd.DataFrame) -> list[str]:
    """Detect columns that look like row IDs."""
    result = []
    for col in df.columns:
        name_lower = col.lower().strip()
        if name_lower in ("id", "index", "row_id", "record_id", "uuid", "uid", "_id"):
            result.append(col)
            continue
        if df[col].dtype in ("int64", "int32") and df[col].is_unique and len(df) > 5:
            diff = df[col].diff().dropna()
            if len(diff) > 0 and (diff == 1).mean() > 0.9:
                result.append(col)
    return result


def _detect_suspect_numeric(df: pd.DataFrame, cat_cols: list[str]) -> list[str]:
    """Find string columns that contain mostly numeric values."""
    suspect = []
    for col in cat_cols[:20]:
        try:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().mean() > 0.8:
                suspect.append(col)
        except Exception:
            pass
    return suspect


def _detect_outliers(df: pd.DataFrame, num_cols: list[str]) -> list[str]:
    """Find numeric columns with significant outliers (IQR method)."""
    info = []
    for col in num_cols[:20]:
        try:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                n = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
                pct = n / len(df) * 100
                if n > 0 and pct > 1:
                    info.append(f"{col}({n} rows, {pct:.1f}%)")
        except Exception:
            pass
    return info


# ── Fallback plan ─────────────────────────────────────────────────────────────

def _default_plan(ctx) -> list[dict]:
    """Deterministic fallback if AI planning fails."""
    plan = []

    if ctx.categorical_cols:
        plan.append({"id": "clean.fix_dtypes", "params": {},
                      "reason": "Auto-detect data types"})
        plan.append({"id": "clean.strip_whitespace", "params": {},
                      "reason": "Clean string values"})

    if ctx.constant_cols:
        plan.append({"id": "clean.drop_constant", "params": {},
                      "reason": f"Drop {len(ctx.constant_cols)} constant columns"})

    if ctx.duplicate_count > 0:
        plan.append({"id": "clean.remove_duplicates", "params": {},
                      "reason": f"{ctx.duplicate_count} duplicate rows found"})

    if ctx.high_null_cols:
        plan.append({"id": "clean.drop_nulls", "params": {"threshold": 0.4},
                      "reason": f"Drop high-null columns: {ctx.high_null_cols}"})

    if ctx.null_cols:
        plan.append({"id": "clean.fill_nulls", "params": {"strategy": "auto"},
                      "reason": f"Fill remaining nulls in {len(ctx.null_cols)} columns"})

    return plan


# ── Main entry point ──────────────────────────────────────────────────────────

def run_auto_clean(
    data: list[dict],
    model_id: str | None = None,
) -> dict:
    """AI-driven auto-cleaning pipeline.

    Returns dict with:
      success, cleaned_data, columns, report, operations,
      original_shape, cleaned_shape
    """
    t0 = time.perf_counter()
    df = pd.DataFrame(data)
    original_shape = df.shape
    original_cols = df.columns.tolist()
    original_nulls = int(df.isnull().sum().sum())
    original_dupes = int(df.duplicated().sum())

    log.info("Auto-clean: analyzing %d x %d dataset", *original_shape)

    # ── 1. Analyze ────────────────────────────────────────────────────────────
    ctx = analyze_context(df)
    analysis = _build_analysis_text(df, ctx)

    # ── 2. AI planning ────────────────────────────────────────────────────────
    try:
        llm = get_llm(temperature=0.0, max_tokens=1024, model_id=model_id)
        prompt = AUTO_CLEAN_PROMPT.format(analysis=analysis)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Strip markdown fencing
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        plan = json.loads(raw)
        if not isinstance(plan, list):
            plan = [plan]
        log.info("Auto-clean: AI planned %d operations", len(plan))
    except Exception as e:
        log.error("Auto-clean: AI planning failed (%s) — using fallback", e)
        plan = _default_plan(ctx)

    # ── 3. Execute ────────────────────────────────────────────────────────────
    current_df = df.copy()
    operations: list[dict] = []

    for op in plan:
        handler_id = op.get("id", "")
        params = op.get("params") or {}
        reason = op.get("reason", "")

        if "." not in handler_id:
            continue

        cat, sub = handler_id.split(".", 1)
        handler_fn = HANDLER_REGISTRY.get((cat, sub))

        if handler_fn is None:
            log.info("  skip unknown handler: %s", handler_id)
            continue

        try:
            before_shape = current_df.shape
            hr = handler_fn(current_df, params)

            if hr.success and hr.result_df is not None:
                current_df = hr.result_df
                after_shape = current_df.shape
                operations.append({
                    "operation": handler_id,
                    "reason": reason,
                    "summary": hr.summary or "",
                    "success": True,
                    "rows_before": before_shape[0],
                    "cols_before": before_shape[1],
                    "rows_after": after_shape[0],
                    "cols_after": after_shape[1],
                })
                log.info("  ok %s: %s (%dx%d -> %dx%d)",
                         handler_id, (hr.summary or "")[:60],
                         *before_shape, *after_shape)
            elif hr.success:
                # Handler succeeded but returned no df (e.g. stats-only)
                operations.append({
                    "operation": handler_id,
                    "reason": reason,
                    "summary": hr.summary or "No changes needed",
                    "success": True,
                    "rows_before": before_shape[0],
                    "cols_before": before_shape[1],
                    "rows_after": before_shape[0],
                    "cols_after": before_shape[1],
                })
            else:
                log.info("  skip %s: %s", handler_id, hr.error)
                operations.append({
                    "operation": handler_id,
                    "reason": reason,
                    "summary": f"Skipped: {hr.error}",
                    "success": False,
                })
        except Exception as e:
            log.error("  error %s: %s", handler_id, e)
            operations.append({
                "operation": handler_id,
                "reason": reason,
                "summary": f"Error: {e}",
                "success": False,
            })

    # ── 4. Build report ───────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    cleaned_shape = current_df.shape
    cleaned_nulls = int(current_df.isnull().sum().sum())
    cleaned_dupes = int(current_df.duplicated().sum())
    cols_removed = sorted(set(original_cols) - set(current_df.columns.tolist()))
    rows_removed = original_shape[0] - cleaned_shape[0]

    success_ops = [o for o in operations if o["success"]]

    report = _build_report(
        original_shape=original_shape,
        cleaned_shape=cleaned_shape,
        original_nulls=original_nulls,
        cleaned_nulls=cleaned_nulls,
        original_dupes=original_dupes,
        cleaned_dupes=cleaned_dupes,
        operations=operations,
        cols_removed=cols_removed,
        rows_removed=rows_removed,
        elapsed=elapsed,
    )

    log.info("Auto-clean done: %dx%d -> %dx%d, %d ops, %.1fs",
             original_shape[0], original_shape[1],
             cleaned_shape[0], cleaned_shape[1],
             len(success_ops), elapsed)

    return {
        "success": True,
        "cleaned_data": current_df.to_dict(orient="records"),
        "columns": current_df.columns.tolist(),
        "report": report,
        "operations": operations,
        "original_shape": {"rows": original_shape[0], "cols": original_shape[1]},
        "cleaned_shape": {"rows": cleaned_shape[0], "cols": cleaned_shape[1]},
    }


def _build_report(
    *,
    original_shape: tuple[int, int],
    cleaned_shape: tuple[int, int],
    original_nulls: int,
    cleaned_nulls: int,
    original_dupes: int,
    cleaned_dupes: int,
    operations: list[dict],
    cols_removed: list[str],
    rows_removed: int,
    elapsed: float,
) -> str:
    """Build a markdown cleaning report."""
    success_ops = [o for o in operations if o["success"]]

    lines = [
        "## Auto-Clean Complete",
        "",
        "| Metric | Before | After |",
        "|--------|--------|-------|",
        f"| Rows | {original_shape[0]:,} | {cleaned_shape[0]:,} |",
        f"| Columns | {original_shape[1]} | {cleaned_shape[1]} |",
        f"| Null values | {original_nulls:,} | {cleaned_nulls:,} |",
        f"| Duplicate rows | {original_dupes:,} | {cleaned_dupes:,} |",
        "",
        f"### Operations ({len(success_ops)} applied)",
        "",
    ]

    for i, op in enumerate(operations, 1):
        status = "+" if op["success"] else "-"
        lines.append(f"{i}. **{op['operation']}** — {op['summary']}")
        if op.get("reason"):
            lines.append(f"   _{op['reason']}_")

    if cols_removed:
        lines.extend(["", f"**Columns removed:** {', '.join(cols_removed)}"])
    if rows_removed > 0:
        lines.extend(["", f"**Rows removed:** {rows_removed:,}"])

    lines.extend(["", f"_Completed in {elapsed:.1f}s_"])

    return "\n".join(lines)
