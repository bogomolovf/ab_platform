"""Тесты анализа: сверка с scipy/statsmodels и покрытие ДИ."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

from abtest import tests


def test_welch_matches_scipy(rng):
    """t-тест Уэлча совпадает со scipy.ttest_ind(equal_var=False)."""
    c = rng.normal(0, 1, 300)
    t = rng.normal(0.25, 1.5, 250)
    res = tests.welch_ttest(c, t)
    sp_t, sp_p = stats.ttest_ind(t, c, equal_var=False)
    assert res.t_stat == pytest.approx(sp_t, rel=1e-9)
    assert res.p_value == pytest.approx(sp_p, rel=1e-9)


def test_welch_ci_contains_effect(rng):
    """95% ДИ Уэлча покрывает истинную разницу примерно в 95% случаев."""
    covered = 0
    trials = 300
    true_diff = 0.3
    for i in range(trials):
        r = np.random.default_rng(i)
        c = r.normal(0.0, 1.0, 200)
        t = r.normal(true_diff, 1.0, 200)
        res = tests.welch_ttest(c, t)
        lo, hi = res.absolute_ci
        covered += lo <= true_diff <= hi
    assert 0.90 <= covered / trials <= 0.99


def test_proportion_ztest_matches_statsmodels():
    """z-статистика долей совпадает со statsmodels (пулированная дисперсия)."""
    sc, nc, st_, nt = 120, 1000, 150, 1000
    res = tests.proportion_ztest(sc, nc, st_, nt)
    z_sm, p_sm = proportions_ztest([st_, sc], [nt, nc])
    assert res.z_stat == pytest.approx(z_sm, rel=1e-6)
    assert res.p_value == pytest.approx(p_sm, rel=1e-6)


def test_proportion_relative_lift():
    """Относительный lift считается корректно."""
    res = tests.proportion_ztest(100, 1000, 120, 1000)
    assert res.relative_effect == pytest.approx(0.12 / 0.10 - 1.0, rel=1e-6)


def test_bootstrap_ci_close_to_analytic(rng):
    """Бутстрап-ДИ (BCa и percentile) близок к аналитическому ДИ Уэлча."""
    c = rng.normal(0.0, 1.0, 400)
    t = rng.normal(0.3, 1.0, 400)
    analytic = tests.welch_ttest(c, t).absolute_ci
    for method in ("percentile", "bca"):
        res = tests.bootstrap_mean_diff(c, t, n_boot=3000, method=method, seed=1)
        assert res.ci[0] == pytest.approx(analytic[0], abs=0.06)
        assert res.ci[1] == pytest.approx(analytic[1], abs=0.06)


def test_mann_whitney_matches_scipy(rng):
    """Манна-Уитни совпадает со scipy."""
    c = rng.normal(0, 1, 150)
    t = rng.normal(0.4, 1, 150)
    res = tests.mann_whitney(c, t)
    u_sp, p_sp = stats.mannwhitneyu(t, c, alternative="two-sided")
    assert res.u_stat == pytest.approx(u_sp, rel=1e-9)
    assert res.p_value == pytest.approx(p_sp, rel=1e-9)
    assert res.rank_biserial > 0  # тест стохастически больше


def test_significant_flag(rng):
    """При крупном эффекте результат значим, при нулевом - обычно нет."""
    c = rng.normal(0, 1, 500)
    t = rng.normal(1.0, 1, 500)
    assert tests.welch_ttest(c, t).significant
