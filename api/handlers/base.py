"""Base handler class and HandlerResult dataclass for all DS-Agent handlers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class HandlerResult:
    """Standardized result from any handler function."""
    success: bool
    result_df: Optional[pd.DataFrame] = None
    charts_plotly: list[str] = field(default_factory=list)
    stdout: str = ""
    summary: str = ""
    output_type: str = "query"   # "query" | "generate"
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class BaseHandler:
    """Shared utilities inherited by all handler classes."""

    @staticmethod
    def validate_columns(df: pd.DataFrame, required_cols: list[str]) -> list[str]:
        """Return list of columns that are missing from the DataFrame."""
        return [c for c in required_cols if c not in df.columns]

    @staticmethod
    def get_numeric_cols(df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include="number").columns.tolist()

    @staticmethod
    def get_categorical_cols(df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include=["object", "category"]).columns.tolist()

    @staticmethod
    def get_datetime_cols(df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include="datetime").columns.tolist()

    @staticmethod
    def smart_column_match(df: pd.DataFrame, keyword: str) -> Optional[str]:
        """Fuzzy-match a keyword to an actual column name.

        Returns the best matching column or None if no match found.
        """
        kw = keyword.lower().replace(" ", "").replace("_", "")

        # Exact match first
        for col in df.columns:
            if col.lower() == keyword.lower():
                return col

        # Substring match
        for col in df.columns:
            col_norm = col.lower().replace(" ", "").replace("_", "")
            if kw in col_norm or col_norm in kw:
                return col

        return None

    @staticmethod
    def safe_copy(df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy for generate operations — never modify original."""
        return df.copy()
