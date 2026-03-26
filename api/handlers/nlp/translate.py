"""handle_translate handler."""
from __future__ import annotations

import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.nlp._helpers import _get_text_cols
from api.logger import get_logger

log = get_logger(__name__)

# Supported language codes (subset — Google Translate supports 100+)
_LANG_ALIASES: dict[str, str] = {
    "thai": "th", "th": "th",
    "english": "en", "en": "en", "eng": "en",
    "chinese": "zh-CN", "zh": "zh-CN", "cn": "zh-CN",
    "japanese": "ja", "ja": "ja", "jp": "ja",
    "korean": "ko", "ko": "ko",
    "french": "fr", "fr": "fr",
    "german": "de", "de": "de",
    "spanish": "es", "es": "es",
    "portuguese": "pt", "pt": "pt",
    "russian": "ru", "ru": "ru",
    "arabic": "ar", "ar": "ar",
    "hindi": "hi", "hi": "hi",
    "italian": "it", "it": "it",
    "dutch": "nl", "nl": "nl",
    "vietnamese": "vi", "vi": "vi",
    "indonesian": "id", "id": "id",
    "malay": "ms", "ms": "ms",
    "turkish": "tr", "tr": "tr",
    "polish": "pl", "pl": "pl",
    "swedish": "sv", "sv": "sv",
    "czech": "cs", "cs": "cs",
    "danish": "da", "da": "da",
    "finnish": "fi", "fi": "fi",
    "greek": "el", "el": "el",
    "hebrew": "he", "he": "he",
    "hungarian": "hu", "hu": "hu",
    "norwegian": "no", "no": "no",
    "romanian": "ro", "ro": "ro",
    "ukrainian": "uk", "uk": "uk",
    "auto": "auto",
}


def _resolve_lang(raw: str) -> str:
    """Resolve a human-friendly language name to a Google Translate code."""
    key = raw.strip().lower()
    return _LANG_ALIASES.get(key, key)


def handle_translate(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Translate text column(s) from one language to another.

    Uses Google Translate via deep-translator (free, no API key).

    Params
    ------
    column : str | None  — target text column (auto-detect if omitted)
    source : str          — source language (default "auto")
    target : str          — target language (default "en")
    """
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        return HandlerResult(
            success=False,
            error="deep-translator is not installed. Run: pip install deep-translator",
        )

    col = params.get("column")
    source_lang = _resolve_lang(params.get("source", params.get("from", "auto")))
    target_lang = _resolve_lang(params.get("target", params.get("to", "en")))

    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found to translate")

    target_col = text_cols[0]
    result = df.copy()

    new_col = f"{target_col}_{target_lang}"

    translator = GoogleTranslator(source=source_lang, target=target_lang)

    translated: list[str] = []
    errors = 0
    for val in result[target_col].astype(str):
        text = str(val).strip()
        if not text or text.lower() in ("nan", "none", ""):
            translated.append("")
            continue
        try:
            translated.append(translator.translate(text) or "")
        except Exception:
            translated.append(text)  # keep original on failure
            errors += 1

    result[new_col] = translated

    lang_label = f"{source_lang} → {target_lang}"
    summary = (
        f"Translated column '{target_col}' ({lang_label}). "
        f"Result in new column '{new_col}'. "
        f"{len(df)} rows processed"
    )
    if errors:
        summary += f", {errors} rows failed (kept original)"

    return HandlerResult(
        success=True,
        result_df=result,
        output_type="generate",
        summary=summary,
        metadata={"source_lang": source_lang, "target_lang": target_lang, "errors": errors},
    )
