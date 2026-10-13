"""Тесты автоматической интерпретации (вердикта)."""

from __future__ import annotations

from abtest import report, srm, tests


def _sig_positive():
    # 1000 vs 1000, конверсия 10% vs 14% - заведомо значимо и практически заметно.
    return tests.proportion_ztest(100, 1000, 140, 1000)


def test_verdict_rollout_when_significant_and_practical():
    res = _sig_positive()
    v = report.interpret(res, practical_threshold=0.01, achieved_power=0.9)
    assert v.statistically_significant
    assert v.practically_significant
    assert "аскатывать" in v.recommendation


def test_verdict_srm_blocks():
    res = _sig_positive()
    bad_srm = srm.check_srm({"control": 10_000, "treatment": 9_300})
    v = report.interpret(res, srm_result=bad_srm)
    assert any("SRM" in w for w in v.warnings)
    assert "SRM" in v.headline


def test_verdict_underpowered_warning():
    # Незначимый результат при низкой мощности -> предупреждение + «больше данных».
    res = tests.proportion_ztest(100, 1000, 105, 1000)
    v = report.interpret(res, achieved_power=0.4)
    assert any("мощност" in w.lower() for w in v.warnings)


def test_verdict_multiplicity_warning():
    res = _sig_positive()
    v = report.interpret(res, n_comparisons=5)
    assert any("множествен" in w.lower() for w in v.warnings)


def test_verdict_stat_but_not_practical():
    # Значимо, но эффект меньше высокого практического порога.
    res = tests.proportion_ztest(10000, 100000, 10300, 100000)
    v = report.interpret(res, practical_threshold=0.02)
    assert v.statistically_significant
    assert v.practically_significant is False


def test_verdict_text_renders():
    v = report.interpret(_sig_positive(), practical_threshold=0.01)
    assert "ВЕРДИКТ" in v.text
    assert "Рекомендация" in v.text
