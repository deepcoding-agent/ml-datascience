"""handle_cast_columns handler."""
from __future__ import annotations
import pandas as pd
from api.handlers.base import HandlerResult
from api.logger import get_logger
log = get_logger(__name__)

def handle_cast_columns(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Cast one or more columns to a target dtype (int, float, str, bool, category)."""
    col = params.get("column")
    columns = params.get("columns", [col] if col else [])
    dtype = params.get("dtype", params.get("type", "float"))
    if not columns:
        return HandlerResult(success=False, error="Specify column(s) to cast")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        return HandlerResult(success=False, error=f"Columns not found: {missing}")
    result = df.copy()
    converted = []
    for c in columns:
        try:
            if dtype in ("int", "int64"):
                result[c] = pd.to_numeric(result[c], errors="coerce").astype("Int64")
            elif dtype in ("float", "float64"):
                result[c] = pd.to_numeric(result[c], errors="coerce")
            elif dtype in ("str", "string", "object"):
                result[c] = result[c].astype(str)
            elif dtype in ("bool", "boolean"):
                result[c] = result[c].astype(bool)
            elif dtype == "category":
                result[c] = result[c].astype("category")
            else:
                result[c] = result[c].astype(dtype)
            converted.append(c)
        except Exception as e:
            log.warning(f"Failed to cast {c}: {e}")
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Cast {converted} to {dtype}")
