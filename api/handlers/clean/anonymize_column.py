"""handle_anonymize_column handler."""
from __future__ import annotations
import hashlib
import pandas as pd
from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger
log = get_logger(__name__)

def handle_anonymize_column(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Anonymize a column by hashing or masking values (for PII protection)."""
    col = params.get("column")
    method = params.get("method", "hash")  # hash | mask | redact
    col, err = BaseHandler.require_column(df, col, params.get("column", ""))
    if err:
        return err
    result = df.copy()
    if method == "hash":
        result[col] = result[col].astype(str).apply(
            lambda x: hashlib.sha256(x.encode()).hexdigest()[:12] if x and x != "nan" else x)
    elif method == "mask":
        def _mask(v):
            s = str(v)
            if len(s) <= 2:
                return "**"
            return s[0] + "*" * (len(s) - 2) + s[-1]
        result[col] = result[col].astype(str).apply(lambda x: _mask(x) if x != "nan" else x)
    else:  # redact
        result[col] = "[REDACTED]"
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Anonymized \'{col}\' using {method} method")
