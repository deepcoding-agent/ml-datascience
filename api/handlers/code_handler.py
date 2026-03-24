"""Code handler — fallback to sandbox exec for arbitrary code."""
from __future__ import annotations

from typing import Any

import pandas as pd

from api.handlers.base import HandlerResult
from api.logger import get_logger
from api.sandbox import run_code

log = get_logger(__name__)


class CodeHandler:

    @staticmethod
    def handle_custom_code(
        df: pd.DataFrame,
        code: str,
        extra_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> HandlerResult:
        """Execute arbitrary code in the sandbox — used when no handler matches."""
        log.info("  code_handler: executing custom code (%d chars)", len(code))

        stdout, result_df, chart_b64, sandbox_df, chart_json = run_code(code, df, extra_dfs)

        if stdout.startswith("Code execution error"):
            return HandlerResult(success=False, error=stdout)

        charts: list[str] = []
        if chart_json:
            charts.append(chart_json)

        # Determine output_type from result
        has_new_data = result_df is not None and len(result_df) > 10 and len(result_df.columns) >= 3
        output_type = "generate" if has_new_data else "query"

        return HandlerResult(
            success=True,
            result_df=result_df,
            charts_plotly=charts,
            stdout=stdout,
            output_type=output_type,
            metadata={
                "chart_image": chart_b64,
                "sandbox_df_shape": sandbox_df.shape if sandbox_df is not None else None,
            },
        )
