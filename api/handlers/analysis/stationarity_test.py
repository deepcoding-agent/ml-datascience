"""handle_stationarity_test handler."""
from __future__ import annotations

import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_stationarity_test(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Augmented Dickey-Fuller test for time series stationarity."""
    try:
        from statsmodels.tsa.stattools import adfuller

        target_col = params.get("column")

        # Determine which columns to test
        if target_col and target_col in df.columns:
            test_cols = [target_col]
        else:
            test_cols = BaseHandler.get_numeric_cols(df)

        if not test_cols:
            return HandlerResult(
                success=False,
                error="No numeric columns found for stationarity testing.",
            )

        rows: list[dict] = []
        for col in test_cols:
            series = df[col].dropna()
            if len(series) < 10:
                rows.append({
                    "column": col,
                    "adf_statistic": None,
                    "p_value": None,
                    "is_stationary": "insufficient data",
                    "critical_1pct": None,
                    "critical_5pct": None,
                    "critical_10pct": None,
                })
                continue

            try:
                result = adfuller(series, autolag="AIC")
                adf_stat = round(result[0], 4)
                p_val = round(result[1], 6)
                crit = result[4]
                rows.append({
                    "column": col,
                    "adf_statistic": adf_stat,
                    "p_value": p_val,
                    "is_stationary": "yes" if p_val < 0.05 else "no",
                    "critical_1pct": round(crit["1%"], 4),
                    "critical_5pct": round(crit["5%"], 4),
                    "critical_10pct": round(crit["10%"], 4),
                })
            except Exception as col_err:
                log.warning("ADF test failed for column %s: %s", col, col_err)
                rows.append({
                    "column": col,
                    "adf_statistic": None,
                    "p_value": None,
                    "is_stationary": f"error: {col_err}",
                    "critical_1pct": None,
                    "critical_5pct": None,
                    "critical_10pct": None,
                })

        result_df = pd.DataFrame(rows)
        stationary_count = sum(1 for r in rows if r["is_stationary"] == "yes")

        return HandlerResult(
            success=True,
            result_df=result_df,
            output_type="query",
            summary=(
                f"ADF stationarity test: {stationary_count}/{len(rows)} columns "
                f"are stationary (p < 0.05)"
            ),
        )
    except Exception as e:
        log.exception("stationarity_test failed")
        return HandlerResult(success=False, error=f"stationarity_test failed: {e}")
