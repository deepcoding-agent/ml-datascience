"""handle_coalesce_columns handler."""
from __future__ import annotations
import pandas as pd
from api.handlers.base import HandlerResult
from api.logger import get_logger
log = get_logger(__name__)

def handle_coalesce_columns(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Coalesce multiple columns — take first non-null value per row."""
    columns = params.get("columns", [])
    new_name = params.get("new_name", "coalesced")
    if not columns or len(columns) < 2:
        return HandlerResult(success=False, error="Provide 'columns' list with at least 2 column names")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        return HandlerResult(success=False, error=f"Columns not found: {missing}")
    result = df.copy()
    result[new_name] = result[columns[0]]
    for c in columns[1:]:
        result[new_name] = result[new_name].fillna(result[c])
    filled = result[new_name].notna().sum()
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Coalesced {columns} → \'{new_name}\' ({filled}/{len(df)} non-null)")
