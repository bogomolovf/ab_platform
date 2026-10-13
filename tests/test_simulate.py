"""Тесты симуляционного модуля (лёгкие прогоны с запасом на шум)."""

from __future__ import annotations

import numpy as np

from abtest import simulate


def test_fpr_calibrated_conversion():
    """A/A по долям: FPR в пределах ДИ вокруг α."""
    res = simulate.simulate_false_positive_rate(
        metric_type="conversion", n_sims=3000, n_per_group=4000, seed=0
    )
    assert abs(res.fpr - 0.05) < 0.02


def test_fpr_calibrated_continuous():
    res = simulate.simulate_false_positive_rate(
        metric_type="continuous", n_sims=3000, n_per_group=500, seed=0
    )
    assert abs(res.fpr - 0.05) < 0.02


def test_power_empirical_matches_analytic():
    res = simulate.simulate_power(n_per_group=400, n_sims=1500, seed=0)
    gap = np.max(np.abs(res.empirical_power - res.analytic_power))
    assert gap < 0.05


def test_cuped_increases_power():
    res = simulate.simulate_cuped(correlation=0.7, n_sims=1500, seed=0)
    assert res.power_with > res.power_without
    assert res.var_reduction_observed > 0.4  # ~= rho^2 = 0.49


def test_peeking_inflates_naive_but_methods_control():
    res = simulate.simulate_peeking(n_per_group=1500, n_looks=15, n_sims=2000, seed=0)
    # Наивное подглядывание раздувает ошибку сильно выше α.
    assert res.cumulative_fpr_naive[-1] > 0.12
    # Корректные методы держатся около α.
    assert res.cumulative_fpr_msprt[-1] <= 0.06
    assert res.cumulative_fpr_spending[-1] <= 0.08
