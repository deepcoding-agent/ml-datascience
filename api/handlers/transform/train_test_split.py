"""handle_train_test_split handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_train_test_split(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Split dataset into train and test sets. Adds a _split column (train/test)."""
    test_size = float(params.get("test_size", 0.2))
    seed = int(params.get("seed", 42))
    stratify_col = params.get("column")

    result = df.copy()
    n_test = max(1, int(len(result) * test_size))
    n_train = len(result) - n_test

    if stratify_col and stratify_col in result.columns:
        try:
            from sklearn.model_selection import train_test_split as sk_split
            train_idx, test_idx = sk_split(
                result.index, test_size=test_size, random_state=seed,
                stratify=result[stratify_col],
            )
            result.loc[train_idx, "_split"] = "train"
            result.loc[test_idx, "_split"] = "test"
        except Exception:
            rng = np.random.RandomState(seed)
            mask = rng.rand(len(result)) >= test_size
            result["_split"] = np.where(mask, "train", "test")
    else:
        rng = np.random.RandomState(seed)
        mask = rng.rand(len(result)) >= test_size
        result["_split"] = np.where(mask, "train", "test")

    train_n = int((result["_split"] == "train").sum())
    test_n = int((result["_split"] == "test").sum())
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Split: {train_n} train + {test_n} test ({test_size*100:.0f}% test)",
    )
