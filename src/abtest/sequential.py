"""Последовательный анализ: корректное «подглядывание».

Наивное ежедневное подглядывание в фиксированный z/t-тест раздувает ошибку
I рода: чем чаще смотрим, тем выше шанс хотя бы раз случайно увидеть p < α.
Здесь реализованы два корректных подхода, контролирующих ошибку при любом
числе промежуточных проверок:

1. **mSPRT (mixture Sequential Probability Ratio Test)** - always-valid
   p-values (Johari, Pekelis, Walsh, 2017; Robbins, 1970). На каждом шаге
   считается смешанное отношение правдоподобия ``Λ_n`` с нормальным
   смешивающим распределением эффекта ``N(0, τ^2)``. Последовательность
   ``Λ_n`` - это тест-мартингал под H0, поэтому по неравенству Вилля
   ``P(exists n: Λ_n >= 1/α) <= α``. Anytime-valid p-value: ``p_n = min(1, 1/max_k Λ_k)``.

2. **Alpha-spending (Lan-DeMets)** - групповой последовательный дизайн с
   границами O'Brien-Fleming или Pocock. Функция расходования распределяет α
   по промежуточным анализам; границы вычисляются рекурсией Армитеджа
   (численное распространение плотности случайного блуждания).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import optimize, signal, stats

__all__ = [
    "SequentialResult",
    "SpendingResult",
    "msprt_lambda",
    "msprt_stream",
    "suggest_tau2",
    "spending_function",
    "alpha_spending_boundaries",
]

Spending = Literal["obrien-fleming", "pocock"]


# --------------------------------------------------------------------------- #
# mSPRT - always-valid p-values
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SequentialResult:
    """Результат mSPRT по потоку данных."""

    n_looks: int
    lambdas: np.ndarray
    anytime_pvalues: np.ndarray
    tau2: float
    alpha: float
    rejected: bool
    stop_index: int | None  # индекс первого пересечения границы (0-based) или None

    @property
    def summary(self) -> str:
        if self.rejected:
            return (
                f"H0 отвергнута на проверке #{self.stop_index + 1} из {self.n_looks}; "
                f"итоговый anytime p={self.anytime_pvalues[-1]:.4g}"
            )
        return (
            f"H0 не отвергнута за {self.n_looks} проверок; anytime p={self.anytime_pvalues[-1]:.4g}"
        )


def suggest_tau2(expected_effect: float, sd: float, n_per_group_planned: int) -> float:
    """Разумное значение τ^2 для mSPRT.

    Оптимальная мощность mSPRT достигается, когда τ^2 близко к дисперсии оценки
    эффекта на планируемом размере выборки при ожидаемом эффекте. Используем
    ``τ^2 = Var(Δ_hat) = 2*sd^2 / n`` как масштаб.

    Args:
        expected_effect: ожидаемый абсолютный эффект (используется как ориентир).
        sd: стандартное отклонение метрики.
        n_per_group_planned: планируемый размер группы.

    Returns:
        Значение τ^2.
    """
    if n_per_group_planned <= 0:
        raise ValueError("n_per_group_planned должно быть положительным.")
    base = 2.0 * sd**2 / n_per_group_planned
    # Не даём τ^2 схлопнуться, если ожидаемый эффект крупный.
    return float(max(base, (expected_effect / 3.0) ** 2, 1e-12))


def msprt_lambda(effect_hat: np.ndarray, var: np.ndarray, tau2: float) -> np.ndarray:
    """Смешанное отношение правдоподобия mSPRT для потока оценок эффекта.

    Для оценки эффекта ``Δ_hat_n ~ N(θ, V_n)`` и смешивающего распределения
    ``N(0, τ^2)``:

    .. math::

        \\Lambda_n = \\sqrt{\\frac{V_n}{V_n + \\tau^2}}
        \\exp\\!\\left(\\frac{\\hat{\\Delta}_n^2}{2}\\cdot
        \\frac{\\tau^2}{V_n (V_n + \\tau^2)}\\right)

    Args:
        effect_hat: массив кумулятивных оценок эффекта на каждой проверке.
        var: массив дисперсий оценки ``V_n`` на каждой проверке.
        tau2: параметр смешивающего распределения τ^2.

    Returns:
        Массив значений ``Λ_n``.
    """
    effect_hat = np.asarray(effect_hat, dtype=float)
    var = np.asarray(var, dtype=float)
    if effect_hat.shape != var.shape:
        raise ValueError("effect_hat и var должны быть одной длины.")
    if np.any(var <= 0):
        raise ValueError("Дисперсии должны быть положительны.")
    if tau2 <= 0:
        raise ValueError("tau2 должно быть положительным.")

    coef = np.sqrt(var / (var + tau2))
    expo = (effect_hat**2 / 2.0) * (tau2 / (var * (var + tau2)))
    # Λ может легитимно быть огромным (сильное свидетельство против H0);
    # ограничиваем экспоненту, чтобы избежать переполнения - решение не меняется.
    expo = np.minimum(expo, 700.0)
    return coef * np.exp(expo)


def msprt_stream(
    control_stream: np.ndarray,
    treatment_stream: np.ndarray,
    tau2: float,
    alpha: float = 0.05,
    look_points: np.ndarray | None = None,
) -> SequentialResult:
    """Прогнать mSPRT по накопительным потокам наблюдений двух групп.

    На каждой контрольной точке считаются кумулятивная оценка разности средних
    ``Δ_hat`` и её дисперсия ``V = s_pooled^2(1/n_c + 1/n_t)``.

    Args:
        control_stream: последовательность наблюдений контроля (в порядке прихода).
        treatment_stream: последовательность наблюдений теста.
        tau2: параметр τ^2 смешивающего распределения.
        alpha: уровень значимости (граница отвержения Λ >= 1/α).
        look_points: индексы (число наблюдений на группу) для проверок; по
            умолчанию - 50 равномерных точек.

    Returns:
        :class:`SequentialResult` с траекторией Λ и anytime p-values.
    """
    c = np.asarray(control_stream, dtype=float)
    t = np.asarray(treatment_stream, dtype=float)
    n = min(c.size, t.size)
    if n < 2:
        raise ValueError("Нужно как минимум по 2 наблюдения в группе.")

    if look_points is None:
        look_points = np.unique(np.linspace(2, n, 50).astype(int))
    look_points = np.asarray(look_points, dtype=int)
    look_points = look_points[(look_points >= 2) & (look_points <= n)]

    effects = np.empty(look_points.size)
    variances = np.empty(look_points.size)
    for i, k in enumerate(look_points):
        cc, tt = c[:k], t[:k]
        pooled_var = 0.5 * (cc.var(ddof=1) + tt.var(ddof=1))
        pooled_var = max(pooled_var, 1e-12)
        effects[i] = tt.mean() - cc.mean()
        variances[i] = pooled_var * (1.0 / k + 1.0 / k)

    lambdas = msprt_lambda(effects, variances, tau2)
    running_max = np.maximum.accumulate(lambdas)
    anytime_p = np.minimum(1.0, 1.0 / running_max)

    crossed = np.where(lambdas >= 1.0 / alpha)[0]
    stop_index = int(crossed[0]) if crossed.size > 0 else None

    return SequentialResult(
        n_looks=look_points.size,
        lambdas=lambdas,
        anytime_pvalues=anytime_p,
        tau2=tau2,
        alpha=alpha,
        rejected=stop_index is not None,
        stop_index=stop_index,
    )


# --------------------------------------------------------------------------- #
# Alpha-spending (Lan-DeMets)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SpendingResult:
    """Границы группового последовательного дизайна."""

    info_fractions: np.ndarray
    z_boundaries: np.ndarray
    nominal_alpha: np.ndarray  # номинальный двусторонний α на каждой проверке
    cumulative_spent: np.ndarray
    alpha: float
    spending: str
    sides: int

    @property
    def summary(self) -> str:
        bounds = ", ".join(f"{z:.3f}" for z in self.z_boundaries)
        return f"{self.spending}: z-границы = [{bounds}] (α={self.alpha}, {self.sides}-сторон.)"


def spending_function(
    info_fractions: np.ndarray,
    alpha: float = 0.05,
    spending: Spending = "obrien-fleming",
    sides: int = 2,
) -> np.ndarray:
    """Кумулятивная функция расходования α на моменты информации ``t``.

    Args:
        info_fractions: доли информации ``t_k in (0, 1]`` (возрастающие).
        alpha: полный уровень значимости.
        spending: ``obrien-fleming`` или ``pocock``.
        sides: 1 или 2.

    Returns:
        Массив кумулятивно израсходованного α (та же длина, что info_fractions).
    """
    t = np.asarray(info_fractions, dtype=float)
    if np.any(t <= 0) or np.any(t > 1.0 + 1e-9):
        raise ValueError("info_fractions должны быть в (0, 1].")
    if not np.all(np.diff(t) > 0):
        raise ValueError("info_fractions должны строго возрастать.")

    if spending == "obrien-fleming":
        z = stats.norm.ppf(1.0 - alpha / sides)
        spent = sides * stats.norm.sf(z / np.sqrt(t))
    elif spending == "pocock":
        spent = alpha * np.log(1.0 + (np.e - 1.0) * t)
    else:
        raise ValueError("spending должен быть 'obrien-fleming' или 'pocock'.")
    return np.minimum(spent, alpha)


def _tail_mass(density: np.ndarray, grid: np.ndarray, dz: float, c: float, sides: int) -> float:
    mask = np.abs(grid) >= c if sides == 2 else grid >= c
    return float(density[mask].sum() * dz)


def alpha_spending_boundaries(
    info_fractions: np.ndarray,
    alpha: float = 0.05,
    spending: Spending = "obrien-fleming",
    sides: int = 2,
    grid_limit: float = 9.0,
    grid_step: float = 0.005,
) -> SpendingResult:
    """Границы z для группового последовательного дизайна (Lan-DeMets).

    Границы находятся рекурсией Армитеджа: плотность накопленной статистики
    (случайного блуждания в шкале информации) численно распространяется между
    проверками, а на каждой проверке граница подбирается так, чтобы «потратить»
    ровно приращение функции расходования α.

    Args:
        info_fractions: доли информации на промежуточных анализах.
        alpha: полный уровень значимости.
        spending: тип функции расходования.
        sides: 1 или 2.
        grid_limit: границы сетки статистики (в шкале ``S = Z*sqrt(t)``).
        grid_step: шаг сетки.

    Returns:
        :class:`SpendingResult` с z-границами и номинальными α.
    """
    t = np.asarray(info_fractions, dtype=float)
    spent = spending_function(t, alpha, spending, sides)
    increments = np.diff(np.concatenate([[0.0], spent]))

    grid = np.arange(-grid_limit, grid_limit + grid_step, grid_step)
    z_bounds = np.empty(t.size)
    density: np.ndarray | None = None
    t_prev = 0.0

    for k, tk in enumerate(t):
        dv = tk - t_prev
        if density is None:
            density = stats.norm.pdf(grid, 0.0, np.sqrt(dv))
        else:
            kernel = stats.norm.pdf(grid, 0.0, np.sqrt(dv)) * grid_step
            density = signal.fftconvolve(density, kernel, mode="same")
            density = np.clip(density, 0.0, None)

        incr = increments[k]

        # Подбор границы c (в S-шкале) под требуемое приращение α.
        # incr и density биндим как аргументы по умолчанию (замыкание внутри цикла).
        def mass(c: float, _d: np.ndarray = density, _incr: float = incr) -> float:
            return _tail_mass(_d, grid, grid_step, c, sides) - _incr

        if incr <= 0 or mass(0.0) <= 0:
            c = grid_limit  # почти ничего не тратим - граница уходит за сетку
        else:
            c = optimize.brentq(mass, 0.0, grid_limit, xtol=grid_step / 2.0)

        z_bounds[k] = c / np.sqrt(tk)
        # Обрезаем плотность до области продолжения (не пересекли границу).
        cont = np.abs(grid) < c if sides == 2 else grid < c
        density = np.where(cont, density, 0.0)
        t_prev = tk

    nominal = (sides * stats.norm.sf(z_bounds)) if sides == 2 else stats.norm.sf(z_bounds)
    return SpendingResult(
        info_fractions=t,
        z_boundaries=z_bounds,
        nominal_alpha=nominal,
        cumulative_spent=spent,
        alpha=alpha,
        spending=spending,
        sides=sides,
    )
