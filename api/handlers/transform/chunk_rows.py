"""handle_chunk_rows handler."""
from __future__ import annotations

import pandas as pd

from api.handlers.base import HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_chunk_rows(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Split a DataFrame into fixed-size row chunks.

    Adds a ``_chunk_id`` column (0-based) so the user can filter by chunk.
    """
    chunk_size = int(params.get("chunk_size", params.get("value", 50)))
    if chunk_size < 1:
        return HandlerResult(success=False, error="chunk_size must be >= 1")

    total_rows = len(df)
    n_chunks = (total_rows + chunk_size - 1) // chunk_size  # ceil division

    result = df.copy()
    result["_chunk_id"] = [i // chunk_size for i in range(total_rows)]

    return HandlerResult(
        success=True,
        result_df=result,
        output_type="generate",
        summary=(
            f"Split {total_rows:,} rows into {n_chunks} chunks of "
            f"{chunk_size} rows each. Column `_chunk_id` (0–{n_chunks - 1}) "
            f"added to identify each chunk."
        ),
        metadata={"chunk_size": chunk_size, "n_chunks": n_chunks},
    )
