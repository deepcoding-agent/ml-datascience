"""Smoke tests — verify the app boots and the health endpoint responds.

Sprint 6 will add real coverage. These tests only ensure CI doesn't regress
on basic import/wiring failures.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> TestClient:
    from api.main import app

    return TestClient(app)


def test_app_imports() -> None:
    from api.main import app

    assert app is not None


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body.get("status") in {"ok", "healthy"} or "status" in body


def test_format_history_helper() -> None:
    from api.llm import format_history_for_prompt

    assert "no previous conversation" in format_history_for_prompt([])
    assert "no previous conversation" in format_history_for_prompt(None)
