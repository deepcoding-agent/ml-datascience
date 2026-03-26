"""handle_named_entity_extract handler."""
from __future__ import annotations
import re
import pandas as pd
from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.nlp._helpers import _get_text_cols, _PATTERNS
from api.logger import get_logger
log = get_logger(__name__)

def handle_named_entity_extract(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Extract named entities from text using pattern-based NER (emails, URLs, phones, numbers, hashtags, mentions)."""
    col = params.get("column")
    entity = params.get("entity", "all")  # email, url, phone, number, hashtag, mention, all
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")
    target = text_cols[0]
    result = df.copy()
    patterns = _PATTERNS if entity == "all" else {entity: _PATTERNS.get(entity, entity)}
    for name, pattern in patterns.items():
        result[f"{target}_{name}"] = result[target].astype(str).apply(
            lambda x: ", ".join(re.findall(pattern, x)) if x else "")
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Extracted entities ({list(patterns.keys())}) from \'{target}\'")
