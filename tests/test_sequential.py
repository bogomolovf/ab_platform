"""Тесты последовательного анализа: mSPRT и alpha-spending."""

from __future__ import annotations

import numpy as np
import pytest

from abtest import sequential


def test_spending_function_endpoints():
    """Функция расходования тратит 0 в начале и весь α в конце (t=1)."""
    t = np.array([0.25, 0.5, 0.75, 1.0])
    spent = sequential.spending_function(t, alpha=0.05, spending="obrien-fleming")
    assert spent[-1] == pytest.approx(0.05, abs=1e-6)
    assert np.all(np.diff(spent) > 0)
    assert spent[0] < 0.01  # OBF почти ничего не тратит рано


def test_obf_boundaries_decreasing():
    """Границы O'Brien-Fleming убывают: рано - строго, поздно - мягче."""
    res = sequential.alpha_spending_boundaries(
        np.array([0.25, 0.5, 0.75, 1.0]), alpha=0.05, spending="obrien-fleming"
    )
    assert np.all(np.diff(res.z_boundaries) < 0)
    assert res.z_boundaries[-1] < res.z_boundaries[0]
    # Последняя граница близка к ~2.0 (двусторонняя OBF).
    assert 1.9 < res.z_boundaries[-1] < 2.1


def test_alpha_spending_controls_type_i_error():
    """Групповой дизайн OBF держит суммарную ошибку I рода около α под H0."""
    info = np.array([0.25, 0.5, 0.75, 1.0])
    res = sequential.alpha_spending_boundaries(info, alpha=0.05, spending="obrien-fleming")
    z_bounds = res.z_boundaries
    n = 2000
    looks = (info * n).astype(int)
    n_sims = 3000
    rng = np.random.default_rng(7)
    rejected = 0
    for _ in range(n_sims):
        c = rng.normal(0, 1, n)
        t = rng.normal(0, 1, n)
        crossed = False
        for j, k in enumerate(looks):
            mc, mt = c[:k].mean(), t[:k].mean()
            vc, vt = c[:k].var(ddof=1), t[:k].var(ddof=1)
            se = np.sqrt((vc + vt) / k)
            z = (mt - mc) / se
            if abs(z) >= z_bounds[j]:
                crossed = True
                break
        rejected += crossed
    fpr = rejected / n_sims
    assert fpr <= 0.075  # около α=0.05 с запасом на шум


def test_msprt_controls_type_i_error():
    """mSPRT под H0 держит долю отвержений не выше α (always-valid)."""
    rng = np.random.default_rng(11)
    n = 2000
    n_sims = 1500
    tau2 = sequential.suggest_tau2(0.05, 1.0, n)
    rejected = 0
    for _ in range(n_sims):
        c = rng.normal(0, 1, n)
        t = rng.normal(0, 1, n)
        res = sequential.msprt_stream(c, t, tau2=tau2, alpha=0.05)
        rejected += res.rejected
    assert rejected / n_sims <= 0.05


def test_msprt_detects_true_effect():
    """При реальном эффекте mSPRT в большинстве прогонов отвергает H0."""
    rng = np.random.default_rng(3)
    n = 3000
    tau2 = sequential.suggest_tau2(0.15, 1.0, n)
    rej = 0
    for _ in range(200):
        c = rng.normal(0, 1, n)
        t = rng.normal(0.15, 1, n)
        res = sequential.msprt_stream(c, t, tau2=tau2, alpha=0.05)
        rej += res.rejected
    assert rej / 200 > 0.6


def test_msprt_lambda_matches_manual():
    """Λ mSPRT совпадает с ручной формулой смешанного правдоподобия."""
    effect = np.array([0.1, 0.2])
    var = np.array([0.01, 0.005])
    tau2 = 0.02
    lam = sequential.msprt_lambda(effect, var, tau2)
    manual = np.sqrt(var / (var + tau2)) * np.exp(effect**2 / 2 * (tau2 / (var * (var + tau2))))
    assert np.allclose(lam, manual)


def test_pocock_boundaries_flat():
    """Границы Pocock примерно постоянны по проверкам."""
    res = sequential.alpha_spending_boundaries(
        np.array([0.25, 0.5, 0.75, 1.0]), alpha=0.05, spending="pocock"
    )
    assert np.ptp(res.z_boundaries) < 0.3  # почти плоские
