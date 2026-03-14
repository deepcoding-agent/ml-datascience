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


# ── Data preparation ──────────────────────────────────────────────────────────

class PrepareRequest(BaseModel):
    dataset: DatasetPayload
    target_column: str | None = None
    test_size: float = 0.2
    scale: bool = True
    correlation_threshold: float = 0.95
    mode: str = "full"   # "full" | "clean" | "cleaning"


class PrepareResponse(BaseModel):
    success: bool
    mode: str = "full"
    report: str
    steps: list[str] = []
    target_column: str = ""
    target_type: str = ""
    feature_names: list[str] = []
    train_rows: int = 0
    test_rows: int = 0
    n_features: int = 0
    dropped_columns: list[str] = []
    corr_dropped: list[str] = []
    encoded_columns: list[str] = []
    scaled_columns: list[str] = []
    label_mappings: dict = {}
    target_label_map: dict | None = None
    X_train: list[dict] = []
    X_test: list[dict] = []
    y_train: list = []
    y_test: list = []
    error: str = ""
