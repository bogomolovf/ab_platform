"""Тесты SRM и баланса ковариат."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from abtest import srm


def test_no_srm_on_balanced_split():
    """Ровный сплит 50/50 не даёт сигнала SRM."""
    res = srm.check_srm({"control": 10_000, "treatment": 9_980})
    assert not res.srm_detected
    assert res.p_value > 0.05


def test_srm_detected_on_skew():
    """Заметный перекос ловится критерием (p ниже порога)."""
    res = srm.check_srm({"control": 10_000, "treatment": 9_400})
    assert res.srm_detected
    assert res.p_value < 0.001


def test_srm_chi2_matches_scipy():
    """χ^2 согласия совпадает с ручным расчётом/scipy для равных ожиданий."""
    counts = [5200, 4800]
    res = srm.check_srm(counts)
    chi2_expected, p_expected = stats.chisquare(counts, [5000, 5000])
    assert res.chi2 == pytest.approx(chi2_expected, rel=1e-9)
    assert res.p_value == pytest.approx(p_expected, rel=1e-6)


def test_srm_custom_ratio():
    """Ожидаемое соотношение 90/10 не помечается как SRM при совпадении."""
    res = srm.check_srm(
        {"control": 9000, "treatment": 1000}, expected_ratio={"control": 0.9, "treatment": 0.1}
    )
    assert not res.srm_detected


def test_covariate_balance_flags_imbalance(rng):
    """Дисбаланс числовой и категориальной ковариаты обнаруживается."""
    n = 4000
    group = np.array(["control"] * n + ["treatment"] * n)
    # tenure сдвинут в тесте; is_mobile тоже перекошен.
    tenure = np.concatenate([rng.normal(30, 10, n), rng.normal(45, 10, n)])
    mobile = np.concatenate([rng.binomial(1, 0.5, n), rng.binomial(1, 0.7, n)])
    import pandas as pd

    df = pd.DataFrame({"group": group, "tenure_days": tenure, "is_mobile": mobile})
    res = srm.covariate_balance(df, "group", ["tenure_days", "is_mobile"], "control", "treatment")
    assert not res.set_index("covariate").loc["tenure_days", "balanced"]
    assert not res.set_index("covariate").loc["is_mobile", "balanced"]


def test_covariate_balance_ok_when_balanced(rng):
    """Сбалансированные ковариаты помечаются как balanced."""
    import pandas as pd

    n = 4000
    group = np.array(["control"] * n + ["treatment"] * n)
    x = rng.normal(30, 10, 2 * n)
    df = pd.DataFrame({"group": group, "x": x})
    res = srm.covariate_balance(df, "group", ["x"], "control", "treatment")
    assert bool(res.loc[0, "balanced"])
