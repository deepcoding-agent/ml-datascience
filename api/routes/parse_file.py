"""POST /parse-file — parse binary columnar formats (parquet, feather, arrow)
into a list of row dicts that the frontend can save as a Dataset.

Client-side parsers in the web app handle text formats (csv/json/xlsx/…)
via SheetJS. Binary columnar formats need pandas + pyarrow on the backend.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from api.logger import get_logger

router = APIRouter()
log = get_logger(__name__)

# 100 MB upload cap — matches what Render allows by default for raw body.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

_SUPPORTED_EXTS = {"parquet", "pq", "feather", "arrow"}


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert DataFrame to JSON-safe records — handles NaN, datetimes, numpy scalars."""
    safe = df.replace({np.nan: None})
    return safe.to_dict(orient="records")


@router.post("/parse-file")
async def parse_file(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in _SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported extension '.{ext}'. Supported: {sorted(_SUPPORTED_EXTS)}",
        )

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw)} bytes). Limit is {MAX_UPLOAD_BYTES} bytes.",
        )

    log.info("▶ /parse-file  name=%s  ext=%s  size=%d", filename, ext, len(raw))
    buf = io.BytesIO(raw)

    try:
        if ext in {"parquet", "pq"}:
            df = pd.read_parquet(buf)
        else:  # feather or arrow
            df = pd.read_feather(buf)
    except Exception as exc:
        log.error("✘ /parse-file parse error for %s: %s", filename, exc)
        raise HTTPException(status_code=422, detail=f"Failed to parse {ext} file: {exc}")

    records = _df_to_records(df)
    log.info("✔ /parse-file  rows=%d  cols=%d", len(records), len(df.columns))
    return {
        "data": records,
        "columns": list(df.columns),
        "rowCount": len(records),
    }
