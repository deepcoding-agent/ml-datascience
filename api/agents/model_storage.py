"""Model Storage — save and load trained models as .joblib files.

Models are stored in ml-datascience/models/ with metadata JSON sidecars.
Sprint 8 will migrate this to S3/GCS.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from api.logger import get_logger

log = get_logger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def save_model(
    model: Any,
    pipeline: Any | None,
    conversation_id: str,
    dataset_id: str,
    task_type: str,
    algorithm: str,
    algorithm_display: str,
    target_column: str,
    feature_columns: list[str],
    metrics: dict[str, float],
    hyperparameters: dict,
    training_duration: float,
    dataset_shape: tuple[int, int],
) -> dict:
    """Save a trained model + preprocessing pipeline to disk.

    Returns metadata dict including model_id and file paths.
    """
    model_id = str(uuid.uuid4())
    model_path = MODELS_DIR / f"{model_id}.joblib"
    meta_path = MODELS_DIR / f"{model_id}.json"

    bundle = {
        "model": model,
        "pipeline": pipeline,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "task_type": task_type,
    }
    joblib.dump(bundle, model_path, compress=3)

    metadata = {
        "model_id": model_id,
        "conversation_id": conversation_id,
        "dataset_id": dataset_id,
        "task_type": task_type,
        "algorithm": algorithm,
        "algorithm_display": algorithm_display,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "hyperparameters": hyperparameters,
        "training_duration_seconds": training_duration,
        "dataset_shape": list(dataset_shape),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_file": str(model_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))

    log.info("Saved model %s (%s) → %s", model_id, algorithm_display, model_path)
    return metadata


def load_model(model_id: str) -> dict:
    """Load a model bundle and its metadata.

    Returns dict with keys: model, pipeline, metadata.
    """
    model_path = MODELS_DIR / f"{model_id}.joblib"
    meta_path = MODELS_DIR / f"{model_id}.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    bundle = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    return {
        "model": bundle["model"],
        "pipeline": bundle.get("pipeline"),
        "feature_columns": bundle.get("feature_columns", []),
        "target_column": bundle.get("target_column", ""),
        "task_type": bundle.get("task_type", ""),
        "metadata": metadata,
    }


def list_models(conversation_id: str) -> list[dict]:
    """List all saved model metadata for a conversation."""
    results = []
    for meta_path in MODELS_DIR.glob("*.json"):
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("conversation_id") == conversation_id:
                results.append(meta)
        except (json.JSONDecodeError, KeyError):
            continue
    results.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return results


def get_model_path(model_id: str) -> Path | None:
    """Return the .joblib file path if it exists."""
    p = MODELS_DIR / f"{model_id}.joblib"
    return p if p.exists() else None
