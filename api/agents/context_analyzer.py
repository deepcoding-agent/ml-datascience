"""Analyzes a DataFrame to produce structured context for intent classification and handlers."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DataContext:
    """Rich structural analysis of a DataFrame."""
    shape: tuple[int, int] = (0, 0)
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    datetime_cols: list[str] = field(default_factory=list)
    null_cols: dict[str, float] = field(default_factory=dict)      # col → null_pct
    high_null_cols: list[str] = field(default_factory=list)        # >40% null
    duplicate_count: int = 0
    skewed_cols: list[str] = field(default_factory=list)           # |skew| > 1
    high_cardinality_cols: list[str] = field(default_factory=list) # unique > 50
    constant_cols: list[str] = field(default_factory=list)         # 1 unique value
    memory_mb: float = 0.0
    suggested_target: str | None = None
    warnings: list[str] = field(default_factory=list)
    column_list: list[str] = field(default_factory=list)


def analyze_context(df: pd.DataFrame) -> DataContext:
    """Build a DataContext from a DataFrame."""
    ctx = DataContext()
    ctx.shape = df.shape
    ctx.column_list = df.columns.tolist()
    ctx.memory_mb = round(df.memory_usage(deep=True).sum() / 1024**2, 2)
    ctx.duplicate_count = int(df.duplicated().sum())

    ctx.numeric_cols = df.select_dtypes(include="number").columns.tolist()
    ctx.categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    ctx.datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

    # Null analysis
    for col in df.columns:
        pct = df[col].isnull().mean() * 100
        if pct > 0:
            ctx.null_cols[col] = round(pct, 1)
        if pct > 40:
            ctx.high_null_cols.append(col)

    # Skewness (numeric only)
    for col in ctx.numeric_cols:
        try:
            skew = abs(df[col].skew())
            if skew > 1.0:
                ctx.skewed_cols.append(col)
        except Exception:
            pass

    # Cardinality + constants
    for col in df.columns:
        uniq = df[col].nunique()
        if uniq <= 1:
            ctx.constant_cols.append(col)
        elif col in ctx.categorical_cols and uniq > 50:
            ctx.high_cardinality_cols.append(col)

    # Suggest target: highest-variance numeric column (heuristic)
    if ctx.numeric_cols:
        try:
            variances = df[ctx.numeric_cols].var()
            ctx.suggested_target = str(variances.idxmax())
        except Exception:
            pass

    # Warnings
    if ctx.duplicate_count > 0:
        ctx.warnings.append(f"{ctx.duplicate_count:,} duplicate rows")
    for col in ctx.high_null_cols:
        ctx.warnings.append(f"{col} has {ctx.null_cols[col]:.0f}% null values")
    for col in ctx.constant_cols:
        ctx.warnings.append(f"{col} is constant (useless feature)")

    return ctx
