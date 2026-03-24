"""Result validator — checks handler output for correctness before returning to user."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from api.agents.intent_classifier import IntentResult
from api.handlers.base import HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


@dataclass
class ValidationResult:
    success: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    retry_strategy: str | None = None


def validate_result(
    result: HandlerResult,
    intent: IntentResult,
    original_df: pd.DataFrame,
) -> ValidationResult:
    """Validate that the handler result makes sense for the given intent."""
    v = ValidationResult()

    if not result.success:
        v.success = False
        v.errors.append(result.error or "Handler returned failure")
        v.retry_strategy = "fallback_to_custom"
        return v

    # Generate intents must produce a DataFrame
    if intent.output_type == "generate":
        if result.result_df is None:
            v.success = False
            v.errors.append("Generate intent produced no DataFrame")
            v.retry_strategy = "fallback_to_custom"
            return v

        if result.result_df.empty:
            v.success = False
            v.errors.append("Result DataFrame is empty (0 rows)")
            v.retry_strategy = "relax_filter"
            return v

        # Assign should keep same row count
        if intent.sub_intent == "assign_value":
            if len(result.result_df) != len(original_df):
                v.success = False
                v.errors.append(
                    f"Assign changed row count: {len(original_df)} → {len(result.result_df)} "
                    "(should be same)"
                )
                v.retry_strategy = "use_assign_not_filter"
                return v

        # Filter should reduce or keep rows
        if intent.sub_intent == "filter":
            if len(result.result_df) > len(original_df):
                v.warnings.append("Filter result has MORE rows than input — unexpected")

    # Viz intents should produce charts
    if intent.category == "viz":
        if not result.charts_plotly and result.result_df is None:
            v.success = False
            v.errors.append("Viz intent produced no chart and no data")
            v.retry_strategy = "fallback_to_custom"

    # Query intents: result_df or charts should exist
    if intent.output_type == "query":
        if result.result_df is None and not result.charts_plotly and not result.stdout:
            v.warnings.append("Query returned no output")

    # Check for all-NaN result
    if result.result_df is not None:
        num_cols = result.result_df.select_dtypes(include="number").columns
        if len(num_cols) > 0 and result.result_df[num_cols].isna().all().all():
            v.warnings.append("All numeric values in result are NaN")

    log.info("  validator: success=%s warnings=%d errors=%d", v.success, len(v.warnings), len(v.errors))
    return v
