"""Pydantic request / response models shared across all routes."""
from __future__ import annotations

from pydantic import BaseModel


# ── Shared payload ────────────────────────────────────────────────────────────

class DatasetPayload(BaseModel):
    name: str
    data: list[dict]


class ChatMessage(BaseModel):
    role: str
    content: str


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    datasets: list[DatasetPayload] = []
    conversation_history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    response: str
    artifacts: dict = {}


# ── Suggest target ────────────────────────────────────────────────────────────

class SuggestTargetRequest(BaseModel):
    columns: list[str]
    sample_data: list[dict] = []


class SuggestTargetResponse(BaseModel):
    target_column: str
    reason: str


# ── EDA report ────────────────────────────────────────────────────────────────

class EDARequest(BaseModel):
    dataset: DatasetPayload


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_count: int = 0
    null_pct: float = 0.0
    unique_count: int = 0
    top_values: dict = {}               # value → count (top 5)
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    skewness: float | None = None


class EDAResponse(BaseModel):
    rows: int
    columns: int
    memory_mb: float
    dtypes: dict[str, int] = {}         # dtype → count
    column_profiles: list[ColumnProfile] = []
    correlation: dict = {}              # nested dict of correlation matrix
    charts: list[str] = []             # base64 PNG images
    duration_seconds: float = 0.0


# ── Data preparation ──────────────────────────────────────────────────────────

class PrepConfig(BaseModel):
    missing_strategy: str = "auto"          # auto|mean|median|mode|drop
    scaling_method: str = "standard"        # standard|minmax|robust|none
    encoding_method: str = "auto"           # auto|onehot|label|ordinal
    outlier_treatment: str = "iqr"          # iqr|zscore|none
    outlier_threshold: float = 1.5
    correlation_threshold: float = 0.95
    test_size: float = 0.2
    random_state: int = 42
    drop_threshold: float = 0.4             # drop col if missing > this ratio


class PrepareRequest(BaseModel):
    dataset: DatasetPayload
    target_column: str | None = None
    mode: str = "full"                      # "full" | "clean" | "cleaning"
    config: PrepConfig = PrepConfig()


class PrepReportDetail(BaseModel):
    """Structured preprocessing report returned alongside the data."""
    steps_applied: list[str] = []
    columns_dropped: list[str] = []
    missing_filled: dict[str, str] = {}     # col → strategy used
    duplicates_removed: int = 0
    outliers_clipped: dict[str, int] = {}   # col → count clipped
    encodings_applied: dict[str, str] = {}  # col → method (onehot|label|ordinal)
    scaler_used: str = "none"
    dtype_inferred: dict[str, str] = {}     # col → new dtype
    rows_before: int = 0
    rows_after: int = 0
    columns_before: int = 0
    columns_after: int = 0
    duration_seconds: float = 0.0


class PrepareResponse(BaseModel):
    success: bool
    mode: str = "full"
    report: str                             # human-readable markdown report
    report_detail: PrepReportDetail = PrepReportDetail()
    target_column: str = ""
    target_type: str = ""
    feature_names: list[str] = []
    train_rows: int = 0
    test_rows: int = 0
    n_features: int = 0
    label_mappings: dict = {}
    target_label_map: dict | None = None
    X_train: list[dict] = []
    X_test: list[dict] = []
    y_train: list = []
    y_test: list = []
    error: str = ""
