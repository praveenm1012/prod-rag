"""Pytest fixtures."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app
from app.observability.init import reset_observability


@pytest.fixture(autouse=True)
def reset_observability_between_tests() -> None:
    """Avoid leaking observability state across tests."""
    get_settings.cache_clear()
    reset_observability()
    yield
    get_settings.cache_clear()
    reset_observability()


@pytest.fixture
def client() -> TestClient:
    """Return a test client for the FastAPI application."""
    return TestClient(create_app())
