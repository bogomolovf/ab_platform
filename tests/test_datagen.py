"""Тесты генератора синтетических данных."""

from __future__ import annotations

import numpy as np
import pytest

from abtest import datagen, srm, tests


def test_reproducible_with_seed():
    a = datagen.generate_experiment(seed=42).data
    b = datagen.generate_experiment(seed=42).data
    assert a.equals(b)


def test_different_seed_differs():
    a = datagen.generate_experiment(seed=1).data
    b = datagen.generate_experiment(seed=2).data
    assert not a.equals(b)


def test_srm_injection_detectable():
    """Умышленная поломка сплита ловится проверкой SRM."""
    d = datagen.generate_experiment(
        metric_type="conversion", n_users=60_000, srm=True, srm_actual_share=0.46, seed=0
    )
    res = srm.check_srm(d.counts)
    assert res.srm_detected


def test_no_srm_when_clean():
    d = datagen.generate_experiment(metric_type="conversion", n_users=60_000, seed=0)
    res = srm.check_srm(d.counts)
    assert not res.srm_detected


def test_true_effect_recovered_continuous():
    """На большой выборке оценка эффекта близка к заложенному истинному."""
    d = datagen.generate_experiment(
        metric_type="continuous",
        n_users=100_000,
        baseline_mean=100,
        sd=20,
        true_effect=2.0,
        pre_correlation=0.7,
        seed=0,
    )
    c, t = d.group_arrays("metric")
    res = tests.welch_ttest(c, t)
    assert res.absolute_effect == pytest.approx(2.0, abs=0.4)


def test_pre_period_correlated_continuous():
    """Пред-период коррелирует с метрикой на заданном уровне."""
    d = datagen.generate_experiment(
        metric_type="continuous", n_users=50_000, pre_correlation=0.7, seed=0
    )
    corr = np.corrcoef(d.data["metric"], d.data["pre_metric"])[0, 1]
    assert corr == pytest.approx(0.7, abs=0.05)


def test_conversion_marginal_rate():
    """Маргинальная конверсия контроля близка к baseline_rate."""
    d = datagen.generate_experiment(
        metric_type="conversion", n_users=100_000, baseline_rate=0.12, true_effect=0.0, seed=0
    )
    rate = d.data.loc[d.data.group == "control", "converted"].mean()
    assert rate == pytest.approx(0.12, abs=0.01)
