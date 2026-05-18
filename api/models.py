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
    model_id: str | None = None            # e.g. "gpt-4o-mini", "claude-sonnet-4-5"


class ChatResponse(BaseModel):
    response: str
    artifacts: dict = {}
    output_type: str = "text"              # text|table|chart|dataset|chart+dataset
    should_activate: bool = False          # frontend should auto-switch to new dataset
    model_used: str | None = None          # echo back which model was used


class ModelInfo(BaseModel):
    """Used by GET /models endpoint."""
    id: str
    label: str
    provider: str                          # "openai" | "anthropic"
    badge: str                             # "Fast" | "Smart" | "Powerful"
    available: bool = False


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


# ── /feature — manual + AI-assisted feature engineering ──────────────────────

class ColumnPlan(BaseModel):
    """Per-column override; None means inherit global default."""
    null_strategy: str | None = None        # drop|mean|median|mode|ffill|constant
    null_fill_value: str | float | None = None
    encoding: str | None = None             # onehot|label|target|ordinal|frequency|none
    drop: bool = False


class DerivedFeatureSpec(BaseModel):
    name: str                               # generated column name
    kind: str                               # date_part|interaction|ratio|log|polynomial
    sources: list[str]                      # source columns
    detail: str = ""                        # short human-readable rationale
    enabled: bool = True


class FeatureConfig(BaseModel):
    # Task & target
    task: str = "classification"            # classification|regression|clustering|unsupervised
    target_column: str | None = None

    # Cleaning — global defaults
    null_default: str = "auto"              # auto|drop|mean|median|mode|ffill|constant
    outlier_strategy: str = "none"          # none|iqr_clip|zscore_remove|winsorize
    outlier_threshold: float = 1.5

    # Encoding & scaling — global defaults
    encoding_default: str = "auto"          # auto|onehot|label|target|ordinal|frequency|none
    scaling: str = "standard"               # none|standard|minmax|robust

    # Per-column overrides
    columns: dict[str, ColumnPlan] = {}

    # Feature selection
    selection_method: str = "none"          # none|variance|correlation|mutual_info|rfe
    selection_top_n: int = 20

    # Derived features (AI proposes, user toggles via .enabled)
    derived: list[DerivedFeatureSpec] = []

    # Dimensionality reduction
    reduction_method: str = "none"          # none|pca
    reduction_n_components: int = 5

    # Output — single dataset by default; split optionally
    split: str = "none"                     # none|train_test
    test_size: float = 0.2
    random_state: int = 42


class FeatureColumnProfile(BaseModel):
    name: str
    dtype: str
    role: str                               # numeric|categorical|datetime|text
    null_pct: float = 0.0
    n_unique: int = 0
    suggested_null_strategy: str = "auto"
    suggested_encoding: str | None = None
    note: str = ""


class FeatureSuggestion(BaseModel):
    config: FeatureConfig
    reasoning: str = ""
    column_profiles: list[FeatureColumnProfile] = []
    derived_candidates: list[DerivedFeatureSpec] = []
    suggested_targets: list[str] = []       # ordered by likelihood
    detected_task: str = "classification"


class FeatureAnalyzeRequest(BaseModel):
    dataset_name: str
    data: list[dict]
    model_id: str | None = None


class FeatureAnalyzeResponse(BaseModel):
    success: bool = True
    suggestion: FeatureSuggestion | None = None
    error: str = ""


class FeatureStep(BaseModel):
    kind: str                               # null_fill|encode|scale|outlier|select|reduce|derive|drop|split
    detail: str
    affected: list[str] = []
    metrics: dict = {}


class FeatureRankItem(BaseModel):
    name: str
    score: float
    method: str                             # mutual_info|variance|correlation|model_importance


class FeatureDroppedItem(BaseModel):
    name: str
    reason: str


class FeatureReportArtifact(BaseModel):
    success: bool = True
    task: str = "classification"
    target_column: str | None = None
    rows_before: int = 0
    cols_before: int = 0
    rows_after: int = 0
    cols_after: int = 0
    steps: list[FeatureStep] = []
    rankings: list[FeatureRankItem] = []
    dropped_columns: list[FeatureDroppedItem] = []
    derived_columns: list[str] = []
    duration_seconds: float = 0.0
    error: str = ""


class FeatureRequest(BaseModel):
    dataset_name: str
    data: list[dict]
    config: FeatureConfig = FeatureConfig()
    model_id: str | None = None


class FeatureResponse(BaseModel):
    success: bool = True
    dataset_name: str = ""
    columns: list[str] = []
    rows: list[dict] = []
    has_split: bool = False
    train_rows: list[dict] = []
    test_rows: list[dict] = []
    report: FeatureReportArtifact = FeatureReportArtifact()
    error: str = ""
    # ID of the saved feature pipeline that can replay this transformation on
    # new raw data later (used by /predict when input is raw).
    feature_pipeline_id: str = ""


# ── FeaturePipeline — fitted transform state for replay on raw data ──────────

class FeaturePipelineState(BaseModel):
    """Serializable snapshot of every transform /feature applied — with the
    fitted parameters needed to replay on new raw data later.

    Stored as JSON next to model artifacts. Linked by id from /train models
    so /predict can detect raw input and auto-apply the same transforms.
    """
    pipeline_id: str
    source_dataset_name: str = ""
    created_at: str = ""
    task: str = "classification"
    target_column: str | None = None
    raw_columns: list[str] = []            # input columns BEFORE any transform
    engineered_columns: list[str] = []     # output columns AFTER all transforms

    # Step state — every value here is what `apply_pipeline` needs at replay
    dropped_columns: list[str] = []
    null_fills: dict[str, dict] = {}       # col → {"strategy": str, "value": Any}
    outlier_bounds: dict[str, dict] = {}   # col → {"low": float, "high": float}
    derived_specs: list[DerivedFeatureSpec] = []   # only enabled ones
    encoders: dict[str, dict] = {}         # col → {"method": str, "state": {...}}
    scaler: dict | None = None             # {"method": str, "columns": [..], "params": {col: {...}}}
    kept_after_selection: list[str] | None = None  # None = no selection applied
    pca: dict | None = None                # {"n_components": int, "feature_cols": [..], "mean": [...], "components": [[...]]}


class FeaturePipelineMeta(BaseModel):
    """Lightweight metadata returned to the UI — does not include fitted state."""
    pipeline_id: str
    source_dataset_name: str
    raw_columns: list[str]
    engineered_columns: list[str]
    target_column: str | None = None
    task: str
    created_at: str


# ── /train/analyze — AI-suggested training setup ─────────────────────────────

class TrainAnalyzeRequest(BaseModel):
    dataset_name: str
    data: list[dict]
    model_id: str | None = None


class TrainSuggestion(BaseModel):
    target_column: str | None = None
    task_type: str = "auto"             # auto|classification|regression|clustering
    scoring_metric: str = "auto"        # auto|accuracy|f1|auc|precision|recall|r2|rmse|mae|silhouette
    cv_folds: int = 5
    tun_trials: int = 50
    test_size: float = 0.2
    reasoning: str = ""
    suggested_targets: list[str] = []


class TrainAnalyzeResponse(BaseModel):
    success: bool = True
    suggestion: TrainSuggestion | None = None
    error: str = ""
