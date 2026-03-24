"""3-level intent classifier with parameter extraction for DS-Agent."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from api.agents.context_analyzer import DataContext
from api.handlers.base import BaseHandler, get_useless_columns, translate_thai_keywords


@dataclass
class IntentResult:
    """Result of intent classification."""
    category: str = "custom"           # stats|clean|transform|viz|feature|custom
    sub_intent: str = "custom_code"    # specific handler name
    params: dict = field(default_factory=dict)
    output_type: str = "query"         # query|generate
    confidence: float = 0.0
    fallback_to_custom: bool = True


# ── Intent tree: category → sub_intent → trigger phrases ──────────────────────

INTENT_TREE: dict[str, dict[str, list[str]]] = {
    "stats": {
        "describe":         ["describe", "summary", "info", "overview", "statistics", "สรุป"],
        "shape":            ["how many rows", "shape", "size", "dimension", "rows and col", "กี่แถว"],
        "null_report":      ["null", "missing", "nan", "na value", "empty", "how many null", "count null", "ค่าว่าง"],
        "value_counts":     ["how many", "count of", "frequency", "distribution of", "นับ", "จำนวน"],
        "correlation":      ["correlation", "corr", "relationship between", "สหสัมพันธ์"],
        "outlier_report":   ["outlier", "anomaly", "extreme value", "ค่าผิดปกติ"],
        "duplicate_report": ["duplicate", "repeated row", "ซ้ำ"],
        "skewness":         ["skew", "skewness", "ความเบ้"],
        "unique_values":    ["unique", "distinct", "ไม่ซ้ำ"],
        "dtypes":           ["dtype", "data type", "column type", "ชนิดข้อมูล"],
    },
    "clean": {
        "drop_nulls":        ["drop null", "remove null", "remove missing", "delete null", "ลบค่าว่าง"],
        "fill_nulls":        ["fill null", "fill missing", "impute", "fill na", "replace null", "เติมค่า"],
        "remove_duplicates": ["remove duplicate", "drop duplicate", "deduplicate", "ลบซ้ำ"],
        "fix_dtypes":        ["fix type", "convert type", "change dtype", "parse date", "แปลงชนิด"],
        "rename_column":     ["rename", "change column name", "เปลี่ยนชื่อ"],
        "drop_column":       ["drop column", "remove column", "delete column", "ลบคอลัมน์"],
        "strip_whitespace":  ["strip", "trim", "whitespace", "ลบช่องว่าง"],
        "replace_values":    ["replace value", "substitute", "แทนค่า"],
    },
    "transform": {
        "filter":           ["filter", "where", "keep only", "select rows", "show rows where", "กรอง"],
        "assign_value":     ["set all", "assign", "change all", "make all", "ทุก rows", "ตั้งค่าทุก"],
        "sort":             ["sort", "order by", "arrange", "เรียง"],
        "groupby_agg":      ["group by", "groupby", "aggregate", "sum by", "average by", "จัดกลุ่ม"],
        "add_column":       ["add column", "create column", "new column", "calculate", "เพิ่มคอลัมน์"],
        "encode_label":     ["label encode", "label encoding"],
        "encode_onehot":    ["one hot", "onehot", "dummies", "get_dummies"],
        "scale_minmax":     ["minmax", "min max scale", "normalize to 0 1"],
        "scale_standard":   ["standard scale", "standardize", "zscore"],
        "inject_null":      ["inject null", "add null", "insert null", "แทรก null", "แทรก missing",
                             "simulate missing", "introduce null", "สร้าง dataset", "create dataset",
                             "สร้างชุดข้อมูล", "inject noise", "แทรก noise", "add noise", "add random"],
        "sample_rows":      ["sample", "random rows", "take n rows", "สุ่ม"],
        "head":             ["head", "first", "top rows", "แสดงแถวแรก"],
        "tail":             ["tail", "last", "bottom rows", "แถวสุดท้าย"],
    },
    "viz": {
        "bar_chart":        ["bar chart", "bar plot", "bar graph", "แท่ง", "กราฟแท่ง"],
        "histogram":        ["histogram", "hist", "distribution", "การกระจาย"],
        "scatter":          ["scatter", "scatter plot", "จุด"],
        "line_chart":       ["line chart", "line plot", "trend", "over time", "เส้น"],
        "box_plot":         ["box plot", "boxplot", "box whisker"],
        "violin_plot":      ["violin", "violin plot"],
        "heatmap":          ["heatmap", "heat map", "correlation map", "ฮีทแมป"],
        "pie_chart":        ["pie", "pie chart", "proportion", "percentage", "สัดส่วน"],
        "pairplot":         ["pairplot", "pair plot", "scatter matrix"],
        "missing_heatmap":  ["missing heatmap", "null map", "missing pattern"],
        "count_plot":       ["count plot", "frequency plot", "plot count"],
        "treemap":          ["treemap", "tree map"],
        "sunburst":         ["sunburst"],
        "bubble_chart":     ["bubble", "bubble chart"],
    },
    "feature": {
        "feature_importance":  ["feature importance", "important feature", "which feature", "ฟีเจอร์สำคัญ"],
        "pca":                 ["pca", "principal component", "dimensionality reduction"],
        "correlation_filter":  ["remove correlated", "drop correlated", "correlation threshold"],
        "log_transform":       ["log transform", "log1p", "skewed column"],
        "variance_filter":     ["variance filter", "low variance", "near zero variance"],
    },
}

# Output type mapping: category → output_type
_OUTPUT_TYPE_MAP = {
    "stats": "query",
    "clean": "generate",
    "transform": "generate",
    "viz": "query",
    "feature": "generate",
    "custom": "query",
}

# Override: these transform sub-intents are queries, not generates
_QUERY_OVERRIDES = {"head", "tail", "groupby_agg"}

# Structural keywords — never fuzzy-match these against column names
STRUCTURAL_KEYWORDS = frozenset({
    "rows", "row", "columns", "column", "cols", "col", "shape", "size",
    "index", "header", "data", "dataset", "table", "dataframe", "df",
    "count", "total", "number", "how", "many", "much", "all", "every",
    "and", "the", "of", "is", "are", "what", "show", "me", "give",
})


def is_structural_query(message: str) -> bool:
    """Return True if message contains ONLY structural keywords (no column references)."""
    words = set(re.findall(r"[a-zA-Z]+", message.lower()))
    non_structural = words - STRUCTURAL_KEYWORDS
    return len(non_structural) == 0


def classify_intent(
    message: str,
    df_context: DataContext,
    conversation_history: list | None = None,
) -> IntentResult:
    """Classify user message into category + sub_intent + params."""
    # Translate Thai keywords before any matching
    translated = translate_thai_keywords(message)
    msg_lower = translated.lower().strip()

    # Guard: structural-only queries go straight to shape handler
    if is_structural_query(translated):
        return IntentResult(
            category="stats", sub_intent="shape", params={},
            output_type="query", confidence=1.0, fallback_to_custom=False,
        )

    best_category = "custom"
    best_sub = "custom_code"
    best_score = 0.0

    for category, sub_intents in INTENT_TREE.items():
        for sub_intent, phrases in sub_intents.items():
            for phrase in phrases:
                if phrase in msg_lower:
                    score = len(phrase) / max(len(msg_lower), 1)
                    # Boost exact phrase matches
                    if msg_lower.startswith(phrase) or msg_lower.endswith(phrase):
                        score += 0.2
                    if score > best_score:
                        best_score = score
                        best_category = category
                        best_sub = sub_intent

    # Determine output type
    output_type = _OUTPUT_TYPE_MAP.get(best_category, "query")
    if best_sub in _QUERY_OVERRIDES:
        output_type = "query"

    # Confidence: 0 if no match, capped at 1.0
    confidence = min(best_score + 0.3, 1.0) if best_score > 0 else 0.0
    fallback = confidence < 0.3

    # Extract parameters
    params = extract_params(message, best_sub, df_context) if not fallback else {}

    return IntentResult(
        category=best_category,
        sub_intent=best_sub,
        params=params,
        output_type=output_type,
        confidence=confidence,
        fallback_to_custom=fallback,
    )


def extract_params(message: str, sub_intent: str, ctx: DataContext) -> dict:
    """Extract operation parameters from the user message."""
    params: dict = {}
    # Translate Thai → English before matching
    translated = translate_thai_keywords(message)
    msg = translated.strip()

    # Detect useless columns to exclude from matching
    dummy_df = pd.DataFrame(columns=ctx.column_list)
    useless = get_useless_columns(dummy_df) if len(ctx.column_list) > 0 else set()

    # Try to find column names mentioned in the message
    # Skip structural keywords and useless columns
    matched_cols: list[str] = []
    for word in re.split(r"[\s,;]+", msg):
        if len(word) < 2 or word.lower() in STRUCTURAL_KEYWORDS:
            continue
        match = BaseHandler.smart_column_match(dummy_df, word, exclude=useless)
        if match:
            matched_cols.append(match)

    if matched_cols:
        params["column"] = matched_cols[0]
        if len(matched_cols) > 1:
            params["columns"] = matched_cols

    # Extract numeric values
    numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", msg)
    if numbers:
        params["value"] = float(numbers[0]) if "." in numbers[0] else int(numbers[0])
        if len(numbers) > 1:
            params["n"] = int(numbers[0])

    # Intent-specific parameter extraction
    msg_lower = msg.lower()

    if sub_intent == "fill_nulls":
        for strategy in ["mean", "median", "mode", "zero", "0"]:
            if strategy in msg_lower:
                params["strategy"] = "zero" if strategy == "0" else strategy
                break
        else:
            params["strategy"] = "auto"

    elif sub_intent == "filter":
        for op, sym in [(">=", ">="), ("<=", "<="), (">", ">"), ("<", "<"),
                        ("==", "=="), ("!=", "!="), ("=", "==")]:
            if op in msg:
                params["operator"] = sym
                break

    elif sub_intent == "sort":
        params["ascending"] = "desc" not in msg_lower and "descending" not in msg_lower

    elif sub_intent in ("head", "tail", "sample_rows"):
        params["n"] = params.get("value", 10)

    elif sub_intent == "assign_value":
        # "set all [col] to [value]"
        pass  # column and value already extracted above

    elif sub_intent == "groupby_agg":
        for agg in ["sum", "mean", "average", "count", "max", "min", "median"]:
            if agg in msg_lower:
                params["agg"] = "mean" if agg == "average" else agg
                break

    elif sub_intent == "scale_minmax":
        params["scaler"] = "minmax"
    elif sub_intent == "scale_standard":
        params["scaler"] = "standard"

    return params


# ── Thai → English mapping for handler name generation ────────────────────────

_THAI_TO_EN = {
    "สร้าง": "create", "ลบ": "remove", "เพิ่ม": "add", "แสดง": "show",
    "คำนวณ": "calculate", "เปลี่ยน": "change", "กรอง": "filter",
    "จัดเรียง": "sort", "รวม": "merge", "แบ่ง": "split",
    "ทำความสะอาด": "clean", "ข้อมูล": "data", "ชุดข้อมูล": "dataset",
    "แทรก": "inject", "ประมาณ": "approx", "ทุก": "all", "ใหม่": "new",
    "แบบสุ่ม": "random",
}

_STOP_WORDS = frozenset({
    "a", "an", "the", "to", "of", "for", "in", "on", "at",
    "is", "are", "with", "and", "or", "i", "me", "my",
    "can", "you", "do", "it", "that", "this", "into",
})


def generate_handler_name(message: str) -> str:
    """Convert user message to a valid snake_case handler name."""
    msg = message.lower()
    for thai, en in _THAI_TO_EN.items():
        msg = msg.replace(thai, en)
    words = re.findall(r"[a-zA-Z0-9]+", msg)
    words = [w for w in words if w not in _STOP_WORDS][:5]
    if not words:
        return "custom_operation"
    return "_".join(words)[:50]
