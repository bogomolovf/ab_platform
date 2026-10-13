"""Тесты модуля планирования: сверка со statsmodels и внутренняя согласованность."""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.stats.power as smp

from abtest import power


@pytest.mark.parametrize("effect,sd", [(0.2, 1.0), (0.5, 2.0), (0.1, 1.0)])
def test_sample_size_mean_matches_statsmodels(effect, sd):
    """Размер выборки для средних совпадает со statsmodels (+/-1 из-за округления)."""
    res = power.sample_size_mean(effect=effect, sd=sd, alpha=0.05, power=0.8)
    sm_n = smp.TTestIndPower().solve_power(
        effect_size=effect / sd, alpha=0.05, power=0.8, ratio=1.0, alternative="two-sided"
    )
    assert abs(res.n_control - np.ceil(sm_n)) <= 1
    assert res.power >= 0.8 - 1e-6


@pytest.mark.parametrize("n", [50, 200, 1000])
def test_power_mean_matches_statsmodels(n):
    """Функция мощности для средних совпадает со statsmodels до 4 знаков."""
    mine = power.power_mean(0.3, 1.0, n, n, alpha=0.05)
    sm = smp.TTestIndPower().power(effect_size=0.3, nobs1=n, alpha=0.05, ratio=1.0)
    assert mine == pytest.approx(sm, abs=1e-4)


def test_mde_mean_roundtrip():
    """MDE, поданный обратно в power, даёт ровно целевую мощность."""
    mde = power.mde_mean(sd=1.0, n_control=394, alpha=0.05, power=0.8)
    achieved = power.power_mean(mde, 1.0, 394, 394, alpha=0.05)
    assert achieved == pytest.approx(0.8, abs=1e-3)


def test_sample_size_proportion_achieves_power_by_simulation(rng):
    """Расчётный размер выборки для долей даёт эмпирическую мощность ~= целевой."""
    p_c, p_t = 0.10, 0.13
    res = power.sample_size_proportion(p_c, p_t, alpha=0.05, power=0.8)
    n = res.n_control
    n_sims = 4000
    xc = rng.binomial(n, p_c, n_sims)
    xt = rng.binomial(n, p_t, n_sims)
    pc, pt = xc / n, xt / n
    ppool = (xc + xt) / (2 * n)
    se = np.sqrt(ppool * (1 - ppool) * (2.0 / n))
    z = np.where(se > 0, (pt - pc) / se, 0.0)
    from scipy import stats

    pvals = 2 * stats.norm.sf(np.abs(z))
    emp_power = np.mean(pvals < 0.05)
    assert emp_power == pytest.approx(0.8, abs=0.03)


def test_mde_proportion_relative():
    """Относительный MDE согласован с абсолютным."""
    abs_mde = power.mde_proportion(0.10, 5000, relative=False)
    rel_mde = power.mde_proportion(0.10, 5000, relative=True)
    assert rel_mde == pytest.approx(abs_mde / 0.10, rel=1e-6)


def test_power_curve_monotonic():
    """Мощность растёт с размером выборки."""
    curve = power.power_curve_mean(0.2, 1.0)
    assert np.all(np.diff(curve.power) >= -1e-9)
    assert curve.power[0] < curve.power[-1]


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        power.sample_size_mean(effect=0.0, sd=1.0)
    with pytest.raises(ValueError):
        power.sample_size_proportion(0.1, 0.1)
    with pytest.raises(ValueError):
        power.power_mean(0.2, -1.0, 100, 100)
