"""Планирование эксперимента: размер выборки, мощность и MDE.

Модуль реализует расчёты для двух типов метрик:

* **Конверсионные (доли)** - нормальная аппроксимация двухвыборочного z-теста для
  разности пропорций с учётом пула дисперсии под нулевой гипотезой и раздельной
  дисперсии под альтернативой. Поддерживается неравное распределение по группам.
* **Непрерывные (средние)** - точный расчёт через нецентральное t-распределение
  (:class:`scipy.stats.nct`), что совпадает с ``statsmodels`` ``TTestIndPower``.

Ключевые понятия:

* *power* (мощность) - вероятность отвергнуть H0, когда истинный эффект существует;
* *MDE* (minimum detectable effect) - минимальный эффект, который эксперимент
  способен обнаружить при заданных мощности, α и размере выборки.

Все функции чистые и детерминированные; численные решения находятся через
:func:`scipy.optimize.brentq`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import optimize, stats

Alternative = Literal["two-sided", "larger", "smaller"]

__all__ = [
    "SampleSizeResult",
    "PowerCurve",
    "z_critical",
    "sample_size_proportion",
    "power_proportion",
    "mde_proportion",
    "sample_size_mean",
    "power_mean",
    "mde_mean",
    "power_curve_proportion",
    "power_curve_mean",
]


@dataclass(frozen=True)
class SampleSizeResult:
    """Результат расчёта размера выборки.

    Attributes:
        n_control: рекомендуемый размер контрольной группы.
        n_treatment: рекомендуемый размер тестовой группы.
        n_total: суммарный размер выборки.
        power: достигаемая мощность при этих размерах.
        alpha: уровень значимости.
        detail: словарь с промежуточными величинами (эффект, дисперсии).
    """

    n_control: int
    n_treatment: int
    n_total: int
    power: float
    alpha: float
    detail: dict[str, float]


@dataclass(frozen=True)
class PowerCurve:
    """Кривая мощности для построения графиков.

    Attributes:
        x: значения аргумента (размер выборки на группу или величина эффекта).
        power: соответствующие значения мощности.
        xlabel: подпись оси X.
    """

    x: np.ndarray
    power: np.ndarray
    xlabel: str


def _validate_alpha_power(alpha: float, power: float | None = None) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha должна быть в (0, 1), получено {alpha}")
    if power is not None and not 0.0 < power < 1.0:
        raise ValueError(f"power должна быть в (0, 1), получено {power}")


def z_critical(alpha: float, alternative: Alternative = "two-sided") -> float:
    """Критическое значение стандартного нормального распределения.

    Args:
        alpha: уровень значимости.
        alternative: тип альтернативы.

    Returns:
        Квантиль ``z`` для порога отвержения H0.
    """
    _validate_alpha_power(alpha)
    if alternative == "two-sided":
        return float(stats.norm.ppf(1.0 - alpha / 2.0))
    return float(stats.norm.ppf(1.0 - alpha))


# --------------------------------------------------------------------------- #
# Конверсионные метрики (доли)
# --------------------------------------------------------------------------- #
def power_proportion(
    p_control: float,
    p_treatment: float,
    n_control: int,
    n_treatment: int,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> float:
    """Мощность двухвыборочного z-теста для разности долей.

    Используется дисперсия под H0 (пул) для порога и раздельная дисперсия под
    альтернативой - так же, как в аналитической формуле размера выборки.

    Args:
        p_control: истинная конверсия контроля.
        p_treatment: истинная конверсия теста.
        n_control: размер контрольной группы.
        n_treatment: размер тестовой группы.
        alpha: уровень значимости.
        alternative: тип альтернативы.

    Returns:
        Мощность в диапазоне [0, 1].
    """
    _validate_alpha_power(alpha)
    if min(p_control, p_treatment) < 0 or max(p_control, p_treatment) > 1:
        raise ValueError("Доли должны быть в [0, 1].")
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("Размеры групп должны быть положительны.")

    n1, n2 = float(n_control), float(n_treatment)
    p_bar = (n1 * p_control + n2 * p_treatment) / (n1 + n2)
    se0 = np.sqrt(p_bar * (1.0 - p_bar) * (1.0 / n1 + 1.0 / n2))
    se1 = np.sqrt(p_control * (1.0 - p_control) / n1 + p_treatment * (1.0 - p_treatment) / n2)
    if se1 == 0:
        return 1.0 if p_control != p_treatment else float(alpha)
    diff = p_treatment - p_control

    if alternative == "two-sided":
        z = z_critical(alpha, "two-sided")
        upper = (diff - z * se0) / se1
        lower = (-diff - z * se0) / se1
        return float(stats.norm.cdf(upper) + stats.norm.cdf(lower))
    z = z_critical(alpha, alternative)
    if alternative == "larger":
        return float(stats.norm.cdf((diff - z * se0) / se1))
    return float(stats.norm.cdf((-diff - z * se0) / se1))


def sample_size_proportion(
    p_control: float,
    p_treatment: float,
    alpha: float = 0.05,
    power: float = 0.8,
    alternative: Alternative = "two-sided",
    ratio: float = 1.0,
) -> SampleSizeResult:
    """Размер выборки для сравнения двух долей.

    Реализована стандартная формула нормальной аппроксимации с пулированной
    дисперсией под H0 и раздельной под альтернативой (Chow, Shao & Wang).

    Args:
        p_control: базовая конверсия контроля.
        p_treatment: ожидаемая конверсия теста.
        alpha: уровень значимости.
        power: требуемая мощность.
        alternative: тип альтернативы.
        ratio: отношение размеров ``n_treatment / n_control``.

    Returns:
        :class:`SampleSizeResult` с округлёнными вверх размерами групп.
    """
    _validate_alpha_power(alpha, power)
    if ratio <= 0:
        raise ValueError("ratio должен быть положительным.")
    if p_control == p_treatment:
        raise ValueError("p_control и p_treatment не должны совпадать.")

    kappa = ratio
    z_a = z_critical(alpha, alternative)
    z_b = float(stats.norm.ppf(power))
    p_bar = (p_control + kappa * p_treatment) / (1.0 + kappa)

    term_null = z_a * np.sqrt(p_bar * (1.0 - p_bar) * (1.0 + 1.0 / kappa))
    term_alt = z_b * np.sqrt(
        p_control * (1.0 - p_control) + p_treatment * (1.0 - p_treatment) / kappa
    )
    n1 = ((term_null + term_alt) ** 2) / (p_control - p_treatment) ** 2

    n_control = int(np.ceil(n1))
    n_treatment = int(np.ceil(n1 * kappa))
    achieved = power_proportion(p_control, p_treatment, n_control, n_treatment, alpha, alternative)
    return SampleSizeResult(
        n_control=n_control,
        n_treatment=n_treatment,
        n_total=n_control + n_treatment,
        power=achieved,
        alpha=alpha,
        detail={
            "p_control": p_control,
            "p_treatment": p_treatment,
            "absolute_effect": p_treatment - p_control,
            "relative_effect": (p_treatment - p_control) / p_control,
            "ratio": ratio,
        },
    )


def mde_proportion(
    p_control: float,
    n_control: int,
    n_treatment: int | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    alternative: Alternative = "two-sided",
    relative: bool = False,
) -> float:
    """Минимально детектируемый эффект для доли при заданном размере выборки.

    Численно решает уравнение ``power(effect) = target_power`` относительно
    конверсии теста методом Брента.

    Args:
        p_control: базовая конверсия.
        n_control: размер контроля.
        n_treatment: размер теста (по умолчанию равен контролю).
        alpha: уровень значимости.
        power: требуемая мощность.
        alternative: тип альтернативы.
        relative: если ``True`` - вернуть относительный MDE (доля от базы),
            иначе абсолютную разницу долей.

    Returns:
        Величина MDE (абсолютная или относительная).
    """
    _validate_alpha_power(alpha, power)
    n_treatment = n_treatment or n_control

    def deficit(p_treatment: float) -> float:
        return (
            power_proportion(p_control, p_treatment, n_control, n_treatment, alpha, alternative)
            - power
        )

    upper = min(0.999999, 1.0 - 1e-9)
    if deficit(upper) < 0:
        raise ValueError("Даже максимальный эффект не даёт требуемой мощности при таком n.")
    p_treatment = optimize.brentq(deficit, p_control + 1e-9, upper, xtol=1e-8)
    effect = p_treatment - p_control
    return float(effect / p_control) if relative else float(effect)


# --------------------------------------------------------------------------- #
# Непрерывные метрики (средние) - точный расчёт через нецентральное t
# --------------------------------------------------------------------------- #
def power_mean(
    effect: float,
    sd: float,
    n_control: int,
    n_treatment: int,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> float:
    """Мощность двухвыборочного t-теста (объединённая дисперсия).

    Расчёт точный: используется нецентральное t-распределение с параметром
    нецентральности ``ncp = effect / (sd * sqrt(1/n1 + 1/n2))``.

    Args:
        effect: абсолютная разница средних (μ_treatment - μ_control).
        sd: общее стандартное отклонение метрики.
        n_control: размер контроля.
        n_treatment: размер теста.
        alpha: уровень значимости.
        alternative: тип альтернативы.

    Returns:
        Мощность в диапазоне [0, 1].
    """
    _validate_alpha_power(alpha)
    if sd <= 0:
        raise ValueError("sd должно быть положительным.")
    if n_control < 2 or n_treatment < 2:
        raise ValueError("Нужно как минимум по 2 наблюдения в группе.")

    n1, n2 = float(n_control), float(n_treatment)
    df = n1 + n2 - 2.0
    ncp = effect / (sd * np.sqrt(1.0 / n1 + 1.0 / n2))

    if alternative == "two-sided":
        t_crit = stats.t.ppf(1.0 - alpha / 2.0, df)
        value = stats.nct.sf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    elif alternative == "larger":
        t_crit = stats.t.ppf(1.0 - alpha, df)
        value = stats.nct.sf(t_crit, df, ncp)
    else:
        t_crit = stats.t.ppf(1.0 - alpha, df)
        value = stats.nct.cdf(-t_crit, df, ncp)

    # При очень больших df/ncp нецентральное t численно нестабильно (nan);
    # в этом пределе t -> нормальное, поэтому используем нормальную аппроксимацию.
    if not np.isfinite(value):
        z = z_critical(alpha, alternative)
        if alternative == "two-sided":
            value = stats.norm.sf(z - ncp) + stats.norm.cdf(-z - ncp)
        elif alternative == "larger":
            value = stats.norm.sf(z - ncp)
        else:
            value = stats.norm.cdf(-z - ncp)
    return float(np.clip(value, 0.0, 1.0))


def sample_size_mean(
    effect: float,
    sd: float,
    alpha: float = 0.05,
    power: float = 0.8,
    alternative: Alternative = "two-sided",
    ratio: float = 1.0,
) -> SampleSizeResult:
    """Размер выборки для сравнения двух средних (точный t-расчёт).

    Решает ``power_mean(n) = target_power`` относительно размера контроля
    методом Брента, поддерживая неравное распределение по группам.

    Args:
        effect: абсолютная разница средних, которую нужно детектировать.
        sd: общее стандартное отклонение.
        alpha: уровень значимости.
        power: требуемая мощность.
        alternative: тип альтернативы.
        ratio: отношение размеров ``n_treatment / n_control``.

    Returns:
        :class:`SampleSizeResult`.
    """
    _validate_alpha_power(alpha, power)
    if sd <= 0:
        raise ValueError("sd должно быть положительным.")
    if effect == 0:
        raise ValueError("effect не должен быть нулевым.")
    if ratio <= 0:
        raise ValueError("ratio должен быть положительным.")

    def deficit(n1: float) -> float:
        n1_int = max(2, int(round(n1)))
        n2_int = max(2, int(round(n1 * ratio)))
        return power_mean(effect, sd, n1_int, n2_int, alpha, alternative) - power

    # Границы поиска: от 2 до очень большого n.
    lo, hi = 2.0, 2.0
    while deficit(hi) < 0:
        hi *= 2.0
        if hi > 5e8:
            raise ValueError("Требуемая мощность недостижима в разумных пределах.")
    n1 = optimize.brentq(deficit, lo, hi, xtol=0.5)

    n_control = max(2, int(np.ceil(n1)))
    n_treatment = max(2, int(np.ceil(n1 * ratio)))
    achieved = power_mean(effect, sd, n_control, n_treatment, alpha, alternative)
    return SampleSizeResult(
        n_control=n_control,
        n_treatment=n_treatment,
        n_total=n_control + n_treatment,
        power=achieved,
        alpha=alpha,
        detail={
            "effect": effect,
            "sd": sd,
            "cohen_d": effect / sd,
            "ratio": ratio,
        },
    )


def mde_mean(
    sd: float,
    n_control: int,
    n_treatment: int | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    alternative: Alternative = "two-sided",
) -> float:
    """Минимально детектируемый эффект (абсолютная разница средних).

    Args:
        sd: общее стандартное отклонение.
        n_control: размер контроля.
        n_treatment: размер теста (по умолчанию равен контролю).
        alpha: уровень значимости.
        power: требуемая мощность.
        alternative: тип альтернативы.

    Returns:
        Абсолютная величина MDE.
    """
    _validate_alpha_power(alpha, power)
    n_treatment = n_treatment or n_control

    def deficit(effect: float) -> float:
        return power_mean(effect, sd, n_control, n_treatment, alpha, alternative) - power

    hi = sd
    while deficit(hi) < 0:
        hi *= 2.0
        if hi > 1e6 * sd:
            raise ValueError("MDE недостижим при таком размере выборки.")
    effect = optimize.brentq(deficit, 1e-12, hi, xtol=1e-9)
    return float(effect)


# --------------------------------------------------------------------------- #
# Кривые мощности
# --------------------------------------------------------------------------- #
def power_curve_proportion(
    p_control: float,
    p_treatment: float,
    n_grid: np.ndarray | None = None,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    ratio: float = 1.0,
) -> PowerCurve:
    """Мощность как функция размера выборки на группу (доли)."""
    if n_grid is None:
        n_grid = np.unique(np.round(np.logspace(1.5, 5, 60)).astype(int))
    powers = np.array(
        [
            power_proportion(
                p_control, p_treatment, int(n), int(round(n * ratio)), alpha, alternative
            )
            for n in n_grid
        ]
    )
    return PowerCurve(x=np.asarray(n_grid, dtype=float), power=powers, xlabel="Размер контроля, n")


def power_curve_mean(
    effect: float,
    sd: float,
    n_grid: np.ndarray | None = None,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    ratio: float = 1.0,
) -> PowerCurve:
    """Мощность как функция размера выборки на группу (средние)."""
    if n_grid is None:
        n_grid = np.unique(np.round(np.logspace(0.5, 4.5, 60)).astype(int))
        n_grid = n_grid[n_grid >= 2]
    powers = np.array(
        [
            power_mean(effect, sd, int(n), max(2, int(round(n * ratio))), alpha, alternative)
            for n in n_grid
        ]
    )
    return PowerCurve(x=np.asarray(n_grid, dtype=float), power=powers, xlabel="Размер контроля, n")
