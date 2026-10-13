"""abtest - платформа для A/B-экспериментов.

Пакет закрывает весь жизненный цикл эксперимента:

* планирование (:mod:`abtest.power`) - размер выборки, мощность, MDE, кривые мощности;
* дизайн и валидация (:mod:`abtest.srm`) - SRM и баланс ковариат;
* анализ (:mod:`abtest.tests`) - z-тест долей, тест Уэлча, бутстрап, Манна-Уитни;
* снижение дисперсии (:mod:`abtest.cuped`) - CUPED по пред-периоду;
* корректное «подглядывание» (:mod:`abtest.sequential`) - mSPRT и alpha-spending;
* байесовский анализ (:mod:`abtest.bayes`) - beta-binomial;
* множественные сравнения (:mod:`abtest.multiple`) - Bonferroni, Benjamini-Hochberg;
* синтетические данные (:mod:`abtest.datagen`);
* Монте-Карло симуляции (:mod:`abtest.simulate`);
* человекочитаемая интерпретация (:mod:`abtest.report`).
"""

from __future__ import annotations

__version__ = "0.1.0"

from abtest import (
    bayes,
    cuped,
    datagen,
    multiple,
    power,
    report,
    sequential,
    simulate,
    srm,
    tests,
)

__all__ = [
    "bayes",
    "cuped",
    "datagen",
    "multiple",
    "power",
    "report",
    "sequential",
    "simulate",
    "srm",
    "tests",
    "__version__",
]
