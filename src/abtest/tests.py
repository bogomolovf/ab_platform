"""Анализ результатов эксперимента: тесты и доверительные интервалы.

Реализованы основные фриквентистские процедуры сравнения двух групп:

* **Доли** - двухвыборочный z-тест разности пропорций (пулированная дисперсия
  для статистики, раздельная - для доверительных интервалов). Возвращаются
  абсолютный и относительный lift с ДИ (относительный - через дельта-метод для
  отношения рисков).
* **Средние** - t-тест Уэлча (не предполагает равенства дисперсий).
* **Бутстрап** - перцентильный и BCa доверительные интервалы для разности средних.
* **Непараметрика** - критерий Манна-Уитни (U) с оценкой размера эффекта
  (rank-biserial correlation).

Все процедуры возвращают неизменяемые dataclass-результаты с полем ``significant``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

Alternative = Literal["two-sided", "larger", "smaller"]

__all__ = [
    "ProportionTestResult",
    "MeanTestResult",
    "BootstrapResult",
    "MannWhitneyResult",
    "proportion_ztest",
    "welch_ttest",
    "bootstrap_mean_diff",
    "mann_whitney",
]


# --------------------------------------------------------------------------- #
# Доли
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProportionTestResult:
    """Результат z-теста для двух долей."""

    p_control: float
    p_treatment: float
    n_control: int
    n_treatment: int
    absolute_effect: float
    absolute_ci: tuple[float, float]
    relative_effect: float
    relative_ci: tuple[float, float]
    z_stat: float
    p_value: float
    alpha: float
    alternative: Alternative
    significant: bool

    @property
    def summary(self) -> str:
        lo, hi = self.absolute_ci
        return (
            f"Δ={self.absolute_effect:+.4f} (95% ДИ [{lo:+.4f}; {hi:+.4f}]), "
            f"относит. lift={self.relative_effect:+.2%}, p={self.p_value:.4g}"
        )


def _counts_to_rate(successes: int, n: int) -> float:
    if n <= 0:
        raise ValueError("Размер группы должен быть положительным.")
    if not 0 <= successes <= n:
        raise ValueError("Число успехов вне диапазона [0, n].")
    return successes / n


def proportion_ztest(
    successes_control: int,
    n_control: int,
    successes_treatment: int,
    n_treatment: int,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> ProportionTestResult:
    """Двухвыборочный z-тест разности долей.

    Args:
        successes_control: число конверсий в контроле.
        n_control: размер контроля.
        successes_treatment: число конверсий в тесте.
        n_treatment: размер теста.
        alpha: уровень значимости.
        alternative: ``two-sided`` | ``larger`` | ``smaller`` (для теста vs контроля).

    Returns:
        :class:`ProportionTestResult`.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha должна быть в (0, 1).")
    p_c = _counts_to_rate(successes_control, n_control)
    p_t = _counts_to_rate(successes_treatment, n_treatment)
    diff = p_t - p_c

    # Статистика теста: пулированная дисперсия под H0.
    p_pool = (successes_control + successes_treatment) / (n_control + n_treatment)
    se_pool = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_control + 1.0 / n_treatment))
    z = diff / se_pool if se_pool > 0 else 0.0

    if alternative == "two-sided":
        p_value = 2.0 * stats.norm.sf(abs(z))
        z_ci = stats.norm.ppf(1.0 - alpha / 2.0)
    elif alternative == "larger":
        p_value = stats.norm.sf(z)
        z_ci = stats.norm.ppf(1.0 - alpha)
    else:
        p_value = stats.norm.cdf(z)
        z_ci = stats.norm.ppf(1.0 - alpha)
    p_value = float(min(1.0, p_value))

    # ДИ на абсолютную разницу: раздельная (Wald) дисперсия.
    se_unpooled = np.sqrt(p_c * (1.0 - p_c) / n_control + p_t * (1.0 - p_t) / n_treatment)
    abs_ci = (diff - z_ci * se_unpooled, diff + z_ci * se_unpooled)

    # Относительный lift и ДИ - дельта-метод для log(p_t / p_c).
    if p_c > 0 and p_t > 0:
        rel = p_t / p_c - 1.0
        var_log = (1.0 - p_t) / (p_t * n_treatment) + (1.0 - p_c) / (p_c * n_control)
        se_log = np.sqrt(max(var_log, 0.0))
        log_ratio = np.log(p_t / p_c)
        rel_ci = (
            float(np.exp(log_ratio - z_ci * se_log) - 1.0),
            float(np.exp(log_ratio + z_ci * se_log) - 1.0),
        )
    else:
        rel = float("nan")
        rel_ci = (float("nan"), float("nan"))

    return ProportionTestResult(
        p_control=p_c,
        p_treatment=p_t,
        n_control=n_control,
        n_treatment=n_treatment,
        absolute_effect=float(diff),
        absolute_ci=(float(abs_ci[0]), float(abs_ci[1])),
        relative_effect=float(rel),
        relative_ci=rel_ci,
        z_stat=float(z),
        p_value=p_value,
        alpha=alpha,
        alternative=alternative,
        significant=bool(p_value < alpha),
    )


# --------------------------------------------------------------------------- #
# Средние: t-тест Уэлча
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MeanTestResult:
    """Результат t-теста Уэлча для двух средних."""

    mean_control: float
    mean_treatment: float
    n_control: int
    n_treatment: int
    absolute_effect: float
    absolute_ci: tuple[float, float]
    relative_effect: float
    t_stat: float
    df: float
    p_value: float
    alpha: float
    alternative: Alternative
    significant: bool
    method: str = "welch"

    @property
    def summary(self) -> str:
        lo, hi = self.absolute_ci
        return (
            f"Δ={self.absolute_effect:+.4f} (95% ДИ [{lo:+.4f}; {hi:+.4f}]), "
            f"t={self.t_stat:.3f}, df={self.df:.1f}, p={self.p_value:.4g}"
        )


def welch_ttest(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> MeanTestResult:
    """t-тест Уэлча (без предположения о равенстве дисперсий).

    Args:
        control: наблюдения контрольной группы.
        treatment: наблюдения тестовой группы.
        alpha: уровень значимости.
        alternative: тип альтернативы.

    Returns:
        :class:`MeanTestResult` с ДИ на разность средних.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha должна быть в (0, 1).")
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    if c.size < 2 or t.size < 2:
        raise ValueError("Нужно как минимум по 2 наблюдения в группе.")

    m_c, m_t = c.mean(), t.mean()
    v_c, v_t = c.var(ddof=1), t.var(ddof=1)
    n_c, n_t = c.size, t.size
    se = np.sqrt(v_c / n_c + v_t / n_t)
    diff = m_t - m_c

    # Степени свободы Уэлча-Саттертуэйта.
    if se > 0:
        df = (v_c / n_c + v_t / n_t) ** 2 / (
            (v_c / n_c) ** 2 / (n_c - 1) + (v_t / n_t) ** 2 / (n_t - 1)
        )
        t_stat = diff / se
    else:
        df = float(n_c + n_t - 2)
        t_stat = 0.0

    sm_alt = {"two-sided": "two-sided", "larger": "greater", "smaller": "less"}[alternative]
    # p-value согласован со scipy.stats.ttest_ind(equal_var=False).
    if sm_alt == "two-sided":
        p_value = 2.0 * stats.t.sf(abs(t_stat), df)
        t_ci = stats.t.ppf(1.0 - alpha / 2.0, df)
    elif sm_alt == "greater":
        p_value = stats.t.sf(t_stat, df)
        t_ci = stats.t.ppf(1.0 - alpha, df)
    else:
        p_value = stats.t.cdf(t_stat, df)
        t_ci = stats.t.ppf(1.0 - alpha, df)
    p_value = float(min(1.0, p_value))

    abs_ci = (diff - t_ci * se, diff + t_ci * se)
    rel = float(diff / m_c) if m_c != 0 else float("nan")

    return MeanTestResult(
        mean_control=float(m_c),
        mean_treatment=float(m_t),
        n_control=int(n_c),
        n_treatment=int(n_t),
        absolute_effect=float(diff),
        absolute_ci=(float(abs_ci[0]), float(abs_ci[1])),
        relative_effect=rel,
        t_stat=float(t_stat),
        df=float(df),
        p_value=p_value,
        alpha=alpha,
        alternative=alternative,
        significant=bool(p_value < alpha),
    )


# --------------------------------------------------------------------------- #
# Бутстрап
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BootstrapResult:
    """Результат бутстрап-оценки разности средних."""

    point_estimate: float
    ci: tuple[float, float]
    method: str
    n_boot: int
    alpha: float
    significant: bool

    @property
    def summary(self) -> str:
        lo, hi = self.ci
        return f"Δ={self.point_estimate:+.4f} ({self.method}, {1 - self.alpha:.0%} ДИ [{lo:+.4f}; {hi:+.4f}])"


def bootstrap_mean_diff(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    method: Literal["percentile", "bca"] = "bca",
    seed: int | None = None,
) -> BootstrapResult:
    """Бутстрап-доверительный интервал для разности средних (μ_treat - μ_control).

    Args:
        control: наблюдения контроля.
        treatment: наблюдения теста.
        alpha: 1 - уровень доверия.
        n_boot: число бутстрап-реплик.
        method: ``percentile`` или ``bca`` (bias-corrected and accelerated).
        seed: seed генератора для воспроизводимости.

    Returns:
        :class:`BootstrapResult`. Значимость определяется как «0 вне ДИ».
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha должна быть в (0, 1).")
    rng = np.random.default_rng(seed)
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    n_c, n_t = c.size, t.size
    if n_c < 2 or n_t < 2:
        raise ValueError("Нужно как минимум по 2 наблюдения в группе.")

    theta_hat = t.mean() - c.mean()
    boot = np.empty(n_boot)
    for i in range(n_boot):
        bc = c[rng.integers(0, n_c, n_c)]
        bt = t[rng.integers(0, n_t, n_t)]
        boot[i] = bt.mean() - bc.mean()

    if method == "percentile":
        lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    elif method == "bca":
        lo, hi = _bca_interval(boot, theta_hat, c, t, alpha)
    else:
        raise ValueError("method должен быть 'percentile' или 'bca'.")

    return BootstrapResult(
        point_estimate=float(theta_hat),
        ci=(float(lo), float(hi)),
        method=method,
        n_boot=n_boot,
        alpha=alpha,
        significant=bool(lo > 0 or hi < 0),
    )


def _bca_interval(
    boot: np.ndarray,
    theta_hat: float,
    c: np.ndarray,
    t: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """BCa-границы: поправка на смещение (z0) и ускорение (a) через jackknife."""
    z0_frac = np.mean(boot < theta_hat)
    z0_frac = min(max(z0_frac, 1e-6), 1 - 1e-6)
    z0 = stats.norm.ppf(z0_frac)

    # Jackknife по объединённой оценке разности средних.
    jack = []
    for i in range(c.size):
        jc = np.delete(c, i)
        jack.append(t.mean() - jc.mean())
    for j in range(t.size):
        jt = np.delete(t, j)
        jack.append(jt.mean() - c.mean())
    jack = np.asarray(jack)
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0

    z_lo, z_hi = stats.norm.ppf(alpha / 2.0), stats.norm.ppf(1.0 - alpha / 2.0)
    a1 = stats.norm.cdf(z0 + (z0 + z_lo) / (1.0 - a * (z0 + z_lo)))
    a2 = stats.norm.cdf(z0 + (z0 + z_hi) / (1.0 - a * (z0 + z_hi)))
    a1 = min(max(a1, 1e-6), 1 - 1e-6)
    a2 = min(max(a2, 1e-6), 1 - 1e-6)
    return tuple(np.quantile(boot, [a1, a2]))


# --------------------------------------------------------------------------- #
# Непараметрика: Манна-Уитни
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MannWhitneyResult:
    """Результат критерия Манна-Уитни."""

    u_stat: float
    p_value: float
    rank_biserial: float
    alpha: float
    alternative: Alternative
    significant: bool

    @property
    def summary(self) -> str:
        return (
            f"U={self.u_stat:.1f}, rank-biserial={self.rank_biserial:+.3f}, "
            f"p={self.p_value:.4g}"
        )


def mann_whitney(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> MannWhitneyResult:
    """Непараметрический критерий Манна-Уитни (U-тест).

    Размер эффекта - rank-biserial correlation ``r = 2U/(n1*n2) - 1`` для группы
    ``treatment``. Значение ``r > 0`` означает, что тест стохастически больше.

    Args:
        control: наблюдения контроля.
        treatment: наблюдения теста.
        alpha: уровень значимости.
        alternative: тип альтернативы.

    Returns:
        :class:`MannWhitneyResult`.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha должна быть в (0, 1).")
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    if c.size < 1 or t.size < 1:
        raise ValueError("Обе группы должны быть непустыми.")

    sm_alt = {"two-sided": "two-sided", "larger": "greater", "smaller": "less"}[alternative]
    u_stat, p_value = stats.mannwhitneyu(t, c, alternative=sm_alt)
    rank_biserial = 2.0 * u_stat / (t.size * c.size) - 1.0

    return MannWhitneyResult(
        u_stat=float(u_stat),
        p_value=float(p_value),
        rank_biserial=float(rank_biserial),
        alpha=alpha,
        alternative=alternative,
        significant=bool(p_value < alpha),
    )
