"""handle_shuffle handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_shuffle(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Randomly shuffle all rows. Useful before train/test splits."""
    seed = int(params.get("seed", 42))
    result = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Shuffled {len(result):,} rows (seed={seed})",
    )
