"""handle_fix_encoding handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_fix_encoding(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Fix common mojibake/encoding issues in string columns."""
    col = params.get("column")
    result = df.copy()
    replacements = {
        "\xc3\xa2\xe2\x82\xac\xe2\x84\xa2": "'",
        "\xc3\xa2\xe2\x82\xac\xcb\x9c": "'",
        "\xc3\xa2\xe2\x82\xac\xc5\x93": '"',
        "\xc3\xa2\xe2\x82\xac\xc2\x9d": '"',
        "\xc3\xa2\xe2\x82\xac\xe2\x80\x9c": "\u2014",
        "\xc3\xa2\xe2\x82\xac\xe2\x80\x9d": "\u2013",
        "\xc3\xa2\xe2\x82\xac\xc2\xa6": "\u2026",
        "\xc3\x83\xc2\xa9": "\u00e9",
        "\xc3\x83\xc2\xa8": "\u00e8",
        "\xc3\x83\xc2\xbc": "\u00fc",
        "\xc3\x83\xc2\xb6": "\u00f6",
        "\xc3\x83\xc2\xa4": "\u00e4",
        "\xc3\x83\xc2\xb1": "\u00f1",
        "\xc2\xc2": "",
        "\x00": "",
    }
    cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
    for c in cols:
        for bad, good in replacements.items():
            result[c] = result[c].astype(str).str.replace(bad, good, regex=False)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Fixed encoding issues in {len(cols)} string column(s)",
    )
