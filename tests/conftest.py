"""Shared pytest fixtures for API tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.api.rate_limit import limiter
from src.config.settings import Settings


@pytest.fixture(autouse=True)
def disable_api_rate_limits():
    limiter.reset()
    disabled = Settings(rate_limit_enabled=False)
    with patch("src.api.rate_limit.get_settings", return_value=disabled):
        yield
    limiter.reset()
