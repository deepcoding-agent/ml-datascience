"""handle_parse_json_column handler."""
from __future__ import annotations
import json
import pandas as pd
from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger
log = get_logger(__name__)

def handle_parse_json_column(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Parse a column containing JSON strings into separate columns."""
    col = params.get("column")
    col, err = BaseHandler.require_column(df, col, params.get("column", ""))
    if err:
        return err
    result = df.copy()
    parsed = []
    errors = 0
    for val in result[col].astype(str):
        try:
            parsed.append(json.loads(val))
        except (json.JSONDecodeError, TypeError):
            parsed.append({})
            errors += 1
    expanded = pd.json_normalize(parsed)
    expanded.columns = [f"{col}_{c}" for c in expanded.columns]
    result = pd.concat([result.reset_index(drop=True), expanded.reset_index(drop=True)], axis=1)
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Parsed JSON in \'{col}\' → {len(expanded.columns)} new columns ({errors} parse errors)")
