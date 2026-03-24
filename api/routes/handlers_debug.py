"""GET /debug/handlers — list all registered + dynamically generated handlers."""
from __future__ import annotations

from fastapi import APIRouter

from api.agents.handler_generator import GENERATED_HANDLERS_DIR, get_cached_handlers
from api.handlers import HANDLER_REGISTRY

router = APIRouter()


@router.get("/debug/handlers")
def list_handlers() -> dict:
    builtin = [f"{cat}.{sub}" for cat, sub in HANDLER_REGISTRY.keys()]
    dynamic_disk = [
        f.stem for f in GENERATED_HANDLERS_DIR.glob("*.py")
        if f.name not in ("__init__.py", ".gitkeep")
    ]
    return {
        "builtin_count": len(builtin),
        "builtin": builtin,
        "dynamic_disk_count": len(dynamic_disk),
        "dynamic_disk": dynamic_disk,
        "dynamic_cached_count": len(get_cached_handlers()),
    }
