"""Тесты CUPED: несмещённость и снижение дисперсии."""

from __future__ import annotations

import numpy as np
import pytest

from abtest import cuped


def _make_data(rng, rho, effect, n):
    x_c = rng.normal(0, 1, n)
    x_t = rng.normal(0, 1, n)
    eps_c = rng.normal(0, np.sqrt(1 - rho**2), n)
    eps_t = rng.normal(0, np.sqrt(1 - rho**2), n)
    y_c = rho * x_c + eps_c
    y_t = rho * x_t + eps_t + effect
    return y_c, x_c, y_t, x_t


def test_cuped_reduces_variance(rng):
    """CUPED уменьшает дисперсию скорректированной метрики."""
    y_c, x_c, y_t, x_t = _make_data(rng, rho=0.7, effect=0.1, n=5000)
    res = cuped.cuped_ttest(y_c, x_c, y_t, x_t)
    var_before = 0.5 * (np.var(y_c, ddof=1) + np.var(y_t, ddof=1))
    var_after = 0.5 * (
        np.var(res.control_adjusted, ddof=1) + np.var(res.treatment_adjusted, ddof=1)
    )
    assert var_after < var_before
    # Теоретическое снижение ~= rho^2.
    assert res.variance_reduction == pytest.approx(0.49, abs=0.05)


def test_cuped_unbiased(rng):
    """Оценка эффекта после CUPED в среднем совпадает с истинным эффектом."""
    est = []
    for i in range(200):
        r = np.random.default_rng(i)
        y_c, x_c, y_t, x_t = _make_data(r, rho=0.8, effect=0.2, n=1000)
        res = cuped.cuped_ttest(y_c, x_c, y_t, x_t)
        est.append(res.test_adjusted.absolute_effect)
    assert np.mean(est) == pytest.approx(0.2, abs=0.02)


def test_cuped_narrows_ci(rng):
    """CI после CUPED уже, чем до."""
    y_c, x_c, y_t, x_t = _make_data(rng, rho=0.75, effect=0.15, n=3000)
    res = cuped.cuped_ttest(y_c, x_c, y_t, x_t)
    width_before = res.test_unadjusted.absolute_ci[1] - res.test_unadjusted.absolute_ci[0]
    width_after = res.test_adjusted.absolute_ci[1] - res.test_adjusted.absolute_ci[0]
    assert width_after < width_before


def test_theta_zero_when_uncorrelated(rng):
    """При отсутствии корреляции θ ~= 0 и дисперсия почти не меняется."""
    y_c, x_c, y_t, x_t = _make_data(rng, rho=0.0, effect=0.1, n=5000)
    _, _, theta, corr = cuped.apply_cuped(y_c, x_c, y_t, x_t)
    assert abs(theta) < 0.1
    assert abs(corr) < 0.1
