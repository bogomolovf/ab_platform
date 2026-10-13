"""Тесты поправок на множественность (сверка со statsmodels)."""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from abtest import multiple


@pytest.fixture
def pvals():
    return np.array([0.001, 0.008, 0.02, 0.04, 0.13, 0.51])


def test_bonferroni_matches_statsmodels(pvals):
    res = multiple.adjust_pvalues(pvals, alpha=0.05, method="bonferroni")
    reject_sm, p_sm, _, _ = multipletests(pvals, alpha=0.05, method="bonferroni")
    assert np.allclose(res.adjusted, p_sm)
    assert np.array_equal(res.reject, reject_sm)


def test_bh_matches_statsmodels(pvals):
    res = multiple.adjust_pvalues(pvals, alpha=0.05, method="benjamini-hochberg")
    reject_sm, p_sm, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    assert np.allclose(res.adjusted, p_sm)
    assert np.array_equal(res.reject, reject_sm)


def test_bh_more_powerful_than_bonferroni(pvals):
    """BH отвергает не меньше гипотез, чем Bonferroni."""
    bh = multiple.adjust_pvalues(pvals, method="benjamini-hochberg")
    bonf = multiple.adjust_pvalues(pvals, method="bonferroni")
    assert bh.n_significant >= bonf.n_significant


def test_invalid_pvalues_raise():
    with pytest.raises(ValueError):
        multiple.adjust_pvalues(np.array([0.1, 1.5]))
    with pytest.raises(ValueError):
        multiple.adjust_pvalues(np.array([]))
