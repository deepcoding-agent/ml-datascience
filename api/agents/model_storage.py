"""Model Storage — save and load trained models as .joblib files.

Models are stored in ``$MODELS_DIR`` (default ``ml-datascience/models``)
with metadata JSON sidecars. In production on Fly.io this points at a
persistent volume mounted at ``/app/models`` (see ``fly.toml``).
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from api.logger import get_logger

log = get_logger(__name__)

# Production: Fly volume at /app/models (set via Dockerfile ENV)
# Dev: repo-relative fallback so local pytest + dev server keep working
_DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR = Path(os.environ.get("MODELS_DIR") or _DEFAULT_MODELS_DIR)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
log.info("Model storage dir: %s", MODELS_DIR)


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


def rename_model(model_id: str, display_name: str) -> dict:
    """Set a friendly display name on a saved model's metadata."""
    meta_path = MODELS_DIR / f"{model_id}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {model_id}")
    meta = json.loads(meta_path.read_text())
    meta["display_name"] = display_name
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    log.info("Renamed model %s -> %s", model_id, display_name)
    return meta


def set_library_flag(model_id: str, saved: bool) -> dict:
    """Toggle the saved_to_library flag on a model's metadata."""
    meta_path = MODELS_DIR / f"{model_id}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {model_id}")
    meta = json.loads(meta_path.read_text())
    meta["saved_to_library"] = bool(saved)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    log.info("Library flag for %s -> %s", model_id, saved)
    return meta


def delete_model(model_id: str) -> bool:
    """Remove both the .joblib bundle and .json metadata for a model."""
    model_path = MODELS_DIR / f"{model_id}.joblib"
    meta_path = MODELS_DIR / f"{model_id}.json"
    deleted = False
    for p in (model_path, meta_path):
        if p.exists():
            p.unlink()
            deleted = True
    if deleted:
        log.info("Deleted model %s", model_id)
    return deleted


def list_library_models() -> list[dict]:
    """List models the user explicitly saved to the cross-conversation library."""
    results: list[dict] = []
    for meta_path in MODELS_DIR.glob("*.json"):
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("saved_to_library"):
                results.append(meta)
        except (json.JSONDecodeError, KeyError):
            continue
    results.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return results
