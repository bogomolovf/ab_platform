"""Тесты байесовского beta-binomial анализа."""

from __future__ import annotations

import pytest

from abtest import bayes


def test_prob_b_better_high_when_b_wins():
    res = bayes.beta_binomial(100, 1000, 160, 1000, seed=1)
    assert res.prob_b_better > 0.98
    assert res.expected_loss_choose_b < res.expected_loss_choose_a


def test_prob_symmetric_when_equal():
    """При одинаковых данных P(B>A) ~= 0.5."""
    res = bayes.beta_binomial(120, 1000, 120, 1000, seed=2)
    assert res.prob_b_better == pytest.approx(0.5, abs=0.03)


def test_expected_losses_nonnegative():
    res = bayes.beta_binomial(100, 1000, 105, 1000, seed=3)
    assert res.expected_loss_choose_a >= 0
    assert res.expected_loss_choose_b >= 0


def test_monotonic_in_treatment_successes():
    """Больше успехов в B -> выше P(B>A)."""
    low = bayes.beta_binomial(100, 1000, 110, 1000, seed=4).prob_b_better
    high = bayes.beta_binomial(100, 1000, 140, 1000, seed=4).prob_b_better
    assert high > low


def test_recommendation_ship_b_when_clear():
    res = bayes.beta_binomial(100, 5000, 300, 5000, seed=5, loss_threshold=1e-3)
    assert "раскатывать B" in res.recommendation


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        bayes.beta_binomial(2000, 1000, 100, 1000)
