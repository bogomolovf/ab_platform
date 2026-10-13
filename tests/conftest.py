"""Общие фикстуры для тестов."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Детерминированный генератор случайных чисел."""
    return np.random.default_rng(12345)
