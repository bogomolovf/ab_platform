"""CUPED - снижение дисперсии за счёт пред-периодной ковариаты.

CUPED (Controlled-experiment Using Pre-Experiment Data, Deng et al., 2013)
корректирует метрику, вычитая линейную зависимость от ковариаты ``X``,
измеренной *до* эксперимента и потому не зависящей от воздействия:

.. math::

    Y_{cuped} = Y - \\theta (X - \\bar{X}), \\quad \\theta = \\frac{cov(Y, X)}{var(X)}

Ожидание метрики не смещается (так как ``X`` не зависит от группы), а дисперсия
падает в ``1 - \\rho^2`` раз, где ``\\rho`` - корреляция между ``Y`` и ``X``.
Это напрямую повышает мощность без увеличения выборки.

``\\theta`` и ``\\bar{X}`` оцениваются на объединённых данных обеих групп, что
корректно, поскольку ковариата не зависит от назначения в группу.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from abtest.tests import MeanTestResult, welch_ttest

__all__ = ["CupedResult", "estimate_theta", "apply_cuped", "cuped_ttest"]


@dataclass(frozen=True)
class CupedResult:
    """Результат применения CUPED и сравнения до/после."""

    theta: float
    correlation: float
    variance_reduction: float
    control_adjusted: np.ndarray
    treatment_adjusted: np.ndarray
    test_unadjusted: MeanTestResult
    test_adjusted: MeanTestResult

    @property
    def summary(self) -> str:
        return (
            f"θ={self.theta:.4f}, ρ={self.correlation:+.3f}, "
            f"снижение дисперсии={self.variance_reduction:.1%}; "
            f"ДИ до={_fmt_ci(self.test_unadjusted.absolute_ci)} -> "
            f"после={_fmt_ci(self.test_adjusted.absolute_ci)}"
        )


def _fmt_ci(ci: tuple[float, float]) -> str:
    return f"[{ci[0]:+.4f}; {ci[1]:+.4f}]"


def estimate_theta(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Оценить коэффициент θ и корреляцию между метрикой и ковариатой.

    Args:
        y: значения метрики (объединённые по группам).
        x: значения пред-периодной ковариаты (объединённые по группам).

    Returns:
        Кортеж ``(theta, correlation)``.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.shape != x.shape:
        raise ValueError("y и x должны быть одной длины.")
    var_x = x.var(ddof=1)
    if var_x == 0:
        return 0.0, 0.0
    cov = np.cov(y, x, ddof=1)[0, 1]
    theta = cov / var_x
    std_y = y.std(ddof=1)
    corr = cov / (std_y * x.std(ddof=1)) if std_y > 0 else 0.0
    return float(theta), float(corr)


def apply_cuped(
    y_control: np.ndarray,
    x_control: np.ndarray,
    y_treatment: np.ndarray,
    x_treatment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Применить CUPED-коррекцию к обеим группам.

    θ и среднее ковариаты оцениваются на объединённых данных.

    Args:
        y_control: метрика контроля.
        x_control: пред-период контроля.
        y_treatment: метрика теста.
        x_treatment: пред-период теста.

    Returns:
        ``(y_control_adj, y_treatment_adj, theta, correlation)``.
    """
    y_c = np.asarray(y_control, dtype=float)
    x_c = np.asarray(x_control, dtype=float)
    y_t = np.asarray(y_treatment, dtype=float)
    x_t = np.asarray(x_treatment, dtype=float)

    y_all = np.concatenate([y_c, y_t])
    x_all = np.concatenate([x_c, x_t])
    theta, corr = estimate_theta(y_all, x_all)
    x_mean = x_all.mean()

    y_c_adj = y_c - theta * (x_c - x_mean)
    y_t_adj = y_t - theta * (x_t - x_mean)
    return y_c_adj, y_t_adj, theta, corr


def cuped_ttest(
    y_control: np.ndarray,
    x_control: np.ndarray,
    y_treatment: np.ndarray,
    x_treatment: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> CupedResult:
    """Полный CUPED-анализ: t-тест до и после коррекции.

    Args:
        y_control: метрика контроля.
        x_control: пред-период контроля.
        y_treatment: метрика теста.
        x_treatment: пред-период теста.
        alpha: уровень значимости.
        alternative: тип альтернативы.

    Returns:
        :class:`CupedResult` с обоими тестами и величиной снижения дисперсии.
    """
    y_c_adj, y_t_adj, theta, corr = apply_cuped(y_control, x_control, y_treatment, x_treatment)
    test_before = welch_ttest(y_control, y_treatment, alpha=alpha, alternative=alternative)  # type: ignore[arg-type]
    test_after = welch_ttest(y_c_adj, y_t_adj, alpha=alpha, alternative=alternative)  # type: ignore[arg-type]

    var_reduction = float(corr**2)  # теоретическое снижение дисперсии = ρ^2
    return CupedResult(
        theta=theta,
        correlation=corr,
        variance_reduction=var_reduction,
        control_adjusted=y_c_adj,
        treatment_adjusted=y_t_adj,
        test_unadjusted=test_before,
        test_adjusted=test_after,
    )
