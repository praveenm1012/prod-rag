"""Observability test fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.core.config import get_settings
from app.observability.init import reset_observability


@pytest.fixture(autouse=True)
def reset_observability_state() -> Iterator[None]:
    """Ensure each test gets a fresh observability backend."""
    get_settings.cache_clear()
    reset_observability()
    yield
    get_settings.cache_clear()
    reset_observability()
