"""Base handler class and HandlerResult dataclass for all DS-Agent handlers."""
from __future__ import annotations

import re
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


# ── Thai → English keyword map ────────────────────────────────────────────────

THAI_COLUMN_KEYWORDS: dict[str, str] = {
    # Housing domain
    "ห้องนอน": "bedroom", "ห้องน้ำ": "bathroom", "ราคา": "price",
    "พื้นที่": "area", "ชั้น": "floor", "ที่จอดรถ": "garage",
    "สระ": "pool", "ครัว": "kitchen", "ห้องนั่งเล่น": "living",
    "ชั้นใต้ดิน": "basement", "หลังคา": "roof", "รั้ว": "fence",
    "สวน": "lot", "ถนน": "street", "คุณภาพ": "quality",
    "สภาพ": "condition", "ปี": "year", "สร้าง": "built",
    # General data science
    "จำนวน": "count", "ค่าเฉลี่ย": "mean", "ผลรวม": "sum",
    "สูงสุด": "max", "ต่ำสุด": "min", "การกระจาย": "distribution",
    "ความสัมพันธ์": "correlation", "ค่าว่าง": "null", "หายไป": "missing",
}


def translate_thai_keywords(message: str) -> str:
    """Replace Thai domain keywords with English equivalents before matching."""
    result = message
    for thai, english in THAI_COLUMN_KEYWORDS.items():
        result = result.replace(thai, f" {english} ")
    # Collapse multiple spaces
    return " ".join(result.split())


# ── Visualization triggers ────────────────────────────────────────────────────

VISUALIZATION_TRIGGERS = frozenset({
    "plot", "chart", "graph", "visualize", "visualise", "visualization",
    "histogram", "scatter", "bar chart", "line chart", "box plot",
    "pie chart", "heatmap", "violin", "pairplot", "treemap", "sunburst",
    "วิซ", "กราฟ", "แท่ง", "พล็อต",
})


def has_visualization_intent(message: str) -> bool:
    """Return True if the message asks for a chart/plot."""
    msg = message.lower()
    return any(t in msg for t in VISUALIZATION_TRIGGERS)


# ── Useless column filter ────────────────────────────────────────────────────

_ID_PATTERNS = frozenset({"id", "index", "key", "row", "num", "number", "seq", "serial"})


def get_useless_columns(df: pd.DataFrame) -> set[str]:
    """Detect identifier/index columns that should never be used for analysis."""
    useless: set[str] = set()
    for col in df.columns:
        col_lower = col.lower().strip("_")
        # Name-based only: must be a short generic ID-like name
        if col_lower in _ID_PATTERNS:
            useless.add(col)
        elif col_lower.endswith("id") and len(col_lower) <= 5:
            # "Id", "idx" but NOT "SalePrice" or "BedroomAbvGr"
            useless.add(col)
    return useless


# ── Smart chart type selection ────────────────────────────────────────────────

def select_chart_type(df: pd.DataFrame, col: str) -> str:
    """Pick the best chart type based on column characteristics."""
    if col not in df.columns:
        return "bar"
    n_unique = df[col].nunique()
    dtype = str(df[col].dtype)
    if "datetime" in dtype:
        return "line"
    if dtype in ("int64", "float64"):
        return "bar" if n_unique <= 20 else "histogram"
    return "bar"  # categorical → bar


# ── Base handler class ────────────────────────────────────────────────────────

class BaseHandler:
    """Shared utilities inherited by all handler classes."""

    @staticmethod
    def validate_columns(df: pd.DataFrame, required_cols: list[str]) -> list[str]:
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
    def smart_column_match(
        df: pd.DataFrame,
        keyword: str,
        exclude: set[str] | None = None,
    ) -> Optional[str]:
        """Fuzzy-match a keyword to an actual column name.

        Skips columns in the *exclude* set (e.g. useless ID columns).
        """
        kw = keyword.lower().replace(" ", "").replace("_", "")
        candidates = [c for c in df.columns if c not in (exclude or set())]

        # Exact match
        for col in candidates:
            if col.lower() == keyword.lower():
                return col
        # Substring match
        for col in candidates:
            col_norm = col.lower().replace(" ", "").replace("_", "")
            if kw in col_norm or col_norm in kw:
                return col
        return None

    @staticmethod
    def require_column(df: pd.DataFrame, col: str | None, keyword: str = "") -> tuple[str | None, HandlerResult | None]:
        """Validate that a column exists in df. Returns (col, None) if OK,
        or (None, error_result) if column not found.

        If col is None, tries smart_column_match on keyword.
        If still not found, returns error listing actual columns.
        """
        if col and col in df.columns:
            return col, None
        # Try fuzzy match
        if keyword:
            useless = get_useless_columns(df)
            matched = BaseHandler.smart_column_match(df, keyword, exclude=useless)
            if matched:
                return matched, None
        available = [c for c in df.columns if c not in get_useless_columns(df)]
        name = col or keyword or "?"
        return None, HandlerResult(
            success=False,
            error=f"Column '{name}' not found. Available columns: {available}",
        )

    @staticmethod
    def safe_copy(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()
