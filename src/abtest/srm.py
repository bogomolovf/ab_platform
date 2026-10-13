"""Валидация дизайна: SRM и баланс ковариат.

* **SRM (Sample Ratio Mismatch)** - расхождение фактического соотношения
  размеров групп с запланированным. Это индикатор поломки сплитования
  (баг в рандомизации, потеря событий, боты). Проверяется критерием
  хи-квадрат согласия. Порог по p-value традиционно жёсткий (0.001), потому
  что при большом трафике даже крошечный, но систематический перекос критичен.
* **Баланс ковариат** - до-экспериментные признаки должны быть распределены
  одинаково между группами. Для непрерывных считается стандартизированная
  разность средних (SMD), для категориальных - критерий хи-квадрат и Cramer's V.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["SRMResult", "check_srm", "covariate_balance"]


@dataclass(frozen=True)
class SRMResult:
    """Результат проверки Sample Ratio Mismatch."""

    observed: dict[str, int]
    expected: dict[str, float]
    chi2: float
    p_value: float
    dof: int
    threshold: float
    srm_detected: bool

    @property
    def summary(self) -> str:
        status = "ОБНАРУЖЕН" if self.srm_detected else "не обнаружен"
        return f"SRM {status}: χ^2={self.chi2:.3f}, p={self.p_value:.4g} (порог {self.threshold})"


def check_srm(
    counts: Mapping[str, int] | Sequence[int],
    expected_ratio: Mapping[str, float] | Sequence[float] | None = None,
    threshold: float = 0.001,
) -> SRMResult:
    """Проверить соответствие фактического сплита ожидаемому (хи-квадрат).

    Args:
        counts: наблюдаемые размеры групп (словарь ``имя->n`` или список).
        expected_ratio: ожидаемые доли/веса групп (нормируются автоматически).
            По умолчанию - равные доли.
        threshold: порог по p-value для сигнала SRM (по умолчанию 0.001).

    Returns:
        :class:`SRMResult`.
    """
    if isinstance(counts, Mapping):
        names = list(counts.keys())
        observed = np.array([counts[k] for k in names], dtype=float)
    else:
        observed = np.asarray(list(counts), dtype=float)
        names = [f"group_{i}" for i in range(len(observed))]

    if observed.size < 2:
        raise ValueError("Нужно минимум две группы.")
    if np.any(observed < 0):
        raise ValueError("Размеры групп не могут быть отрицательными.")

    if expected_ratio is None:
        weights = np.ones_like(observed)
    elif isinstance(expected_ratio, Mapping):
        weights = np.array([expected_ratio[k] for k in names], dtype=float)
    else:
        weights = np.asarray(list(expected_ratio), dtype=float)
    if np.any(weights <= 0):
        raise ValueError("Ожидаемые веса должны быть положительны.")

    total = observed.sum()
    expected = weights / weights.sum() * total

    chi2 = float(np.sum((observed - expected) ** 2 / expected))
    dof = len(observed) - 1
    p_value = float(stats.chi2.sf(chi2, dof))

    return SRMResult(
        observed={n: int(o) for n, o in zip(names, observed)},
        expected={n: float(e) for n, e in zip(names, expected)},
        chi2=chi2,
        p_value=p_value,
        dof=dof,
        threshold=threshold,
        srm_detected=bool(p_value < threshold),
    )


def _smd(x_c: np.ndarray, x_t: np.ndarray) -> float:
    """Стандартизированная разность средних (Cohen-style, объединённое SD)."""
    n_c, n_t = x_c.size, x_t.size
    v_c, v_t = x_c.var(ddof=1), x_t.var(ddof=1)
    pooled_sd = np.sqrt(((n_c - 1) * v_c + (n_t - 1) * v_t) / (n_c + n_t - 2))
    if pooled_sd == 0:
        return 0.0
    return float((x_t.mean() - x_c.mean()) / pooled_sd)


def _cramers_v(contingency: np.ndarray, chi2: float) -> float:
    n = contingency.sum()
    k = min(contingency.shape) - 1
    if n == 0 or k == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * k)))


def covariate_balance(
    data: pd.DataFrame,
    group_col: str,
    covariates: Sequence[str],
    control_label: str,
    treatment_label: str,
    alpha: float = 0.05,
    smd_threshold: float = 0.1,
) -> pd.DataFrame:
    """Проверить баланс ковариат между двумя группами.

    Для числовых ковариат считаются SMD и p-value t-теста Уэлча; ковариата
    считается сбалансированной, если ``|SMD| < smd_threshold``. Для категориальных
    - критерий хи-квадрат и Cramer's V; баланс определяется по p-value.

    Args:
        data: датафрейм с колонкой группы и ковариатами.
        group_col: имя колонки группы.
        covariates: список проверяемых ковариат.
        control_label: метка контрольной группы в ``group_col``.
        treatment_label: метка тестовой группы.
        alpha: уровень значимости для тестов.
        smd_threshold: порог |SMD| для числовых ковариат.

    Returns:
        DataFrame со столбцами: ``covariate, type, effect_size, statistic,
        p_value, balanced``.
    """
    rows = []
    sub = data[data[group_col].isin([control_label, treatment_label])]
    mask_c = sub[group_col] == control_label
    mask_t = sub[group_col] == treatment_label

    for cov in covariates:
        col = sub[cov]
        is_numeric = pd.api.types.is_numeric_dtype(col) and col.nunique() > 10
        if is_numeric:
            x_c = col[mask_c].to_numpy(dtype=float)
            x_t = col[mask_t].to_numpy(dtype=float)
            smd = _smd(x_c, x_t)
            _, p = stats.ttest_ind(x_t, x_c, equal_var=False)
            rows.append(
                {
                    "covariate": cov,
                    "type": "numeric",
                    "effect_size": abs(smd),
                    "statistic": smd,
                    "p_value": float(p),
                    "balanced": bool(abs(smd) < smd_threshold),
                }
            )
        else:
            table = pd.crosstab(sub[group_col], col)
            chi2, p, _, _ = stats.chi2_contingency(table)
            v = _cramers_v(table.to_numpy(), chi2)
            rows.append(
                {
                    "covariate": cov,
                    "type": "categorical",
                    "effect_size": v,
                    "statistic": float(chi2),
                    "p_value": float(p),
                    "balanced": bool(p >= alpha),
                }
            )
    return pd.DataFrame(rows)
