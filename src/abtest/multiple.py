"""Поправки на множественные сравнения.

Когда в эксперименте несколько метрик или вариантов, число тестов растёт, а с
ним - вероятность хотя бы одной ложноположительной находки. Реализованы:

* **Bonferroni** - контроль FWER (family-wise error rate): порог α делится на
  число тестов. Консервативно, но строго.
* **Benjamini-Hochberg (BH)** - контроль FDR (false discovery rate): допускает
  контролируемую долю ложных открытий среди отвергнутых, мощнее Bonferroni.

Функции возвращают скорректированные p-values и флаги отвержения. Реализация
самостоятельная; в тестах сверяется со ``statsmodels.stats.multitest``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = ["MultipleTestResult", "adjust_pvalues"]

Method = Literal["bonferroni", "benjamini-hochberg"]


@dataclass(frozen=True)
class MultipleTestResult:
    """Результат поправки на множественные сравнения."""

    pvalues: np.ndarray
    adjusted: np.ndarray
    reject: np.ndarray
    method: str
    alpha: float

    @property
    def n_significant(self) -> int:
        return int(self.reject.sum())

    @property
    def summary(self) -> str:
        return (
            f"{self.method}: значимых {self.n_significant} из {self.pvalues.size} "
            f"при α={self.alpha}"
        )


def adjust_pvalues(
    pvalues: np.ndarray,
    alpha: float = 0.05,
    method: Method = "benjamini-hochberg",
) -> MultipleTestResult:
    """Скорректировать p-values на множественность.

    Args:
        pvalues: массив исходных p-values.
        alpha: уровень (FWER для Bonferroni, FDR для BH).
        method: ``bonferroni`` или ``benjamini-hochberg``.

    Returns:
        :class:`MultipleTestResult` со скорректированными p-values и флагами.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("pvalues должен быть непустым одномерным массивом.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values должны быть в [0, 1].")
    m = p.size

    if method == "bonferroni":
        adjusted = np.minimum(p * m, 1.0)
        reject = adjusted < alpha
    elif method == "benjamini-hochberg":
        order = np.argsort(p)
        ranked = p[order]
        ranks = np.arange(1, m + 1)
        # Скорректированные p (BH), с обеспечением монотонности справа налево.
        raw = ranked * m / ranks
        adj_sorted = np.minimum.accumulate(raw[::-1])[::-1]
        adj_sorted = np.minimum(adj_sorted, 1.0)
        adjusted = np.empty(m)
        adjusted[order] = adj_sorted
        reject = adjusted < alpha
    else:
        raise ValueError("method должен быть 'bonferroni' или 'benjamini-hochberg'.")

    return MultipleTestResult(
        pvalues=p,
        adjusted=adjusted,
        reject=reject,
        method=method,
        alpha=alpha,
    )
