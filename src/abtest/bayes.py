"""Байесовский анализ конверсии: beta-binomial модель.

Для доли с сопряжённым априори ``Beta(α0, β0)`` апостериорное распределение после
``s`` успехов из ``n`` - это ``Beta(α0 + s, β0 + n - s)``. По апостериорным
распределениям групп считаются:

* ``P(θ_B > θ_A)`` - вероятность, что вариант B лучше;
* **expected loss** (ожидаемые потери) - ``E[(θ_A - θ_B)^+]`` при выборе B и
  наоборот. Решение о раскатке принимается, когда ожидаемые потери меньше
  порога практической незначимости (ε).

Вероятности оцениваются методом Монте-Карло из апостериорных распределений
(с фиксируемым seed для воспроизводимости).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BayesResult", "beta_binomial"]


@dataclass(frozen=True)
class BayesResult:
    """Результат байесовского сравнения двух конверсий."""

    prob_b_better: float
    expected_loss_choose_b: float
    expected_loss_choose_a: float
    lift_mean: float
    lift_ci: tuple[float, float]
    posterior_a: tuple[float, float]  # (alpha, beta)
    posterior_b: tuple[float, float]
    loss_threshold: float
    recommendation: str

    @property
    def summary(self) -> str:
        return (
            f"P(B>A)={self.prob_b_better:.1%}, ожидаемые потери при выборе B="
            f"{self.expected_loss_choose_b:.2e}, lift={self.lift_mean:+.2%}; "
            f"рекомендация: {self.recommendation}"
        )


def beta_binomial(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    loss_threshold: float = 1e-4,
    n_samples: int = 200_000,
    ci: float = 0.95,
    seed: int | None = None,
) -> BayesResult:
    """Байесовское сравнение долей (A = контроль, B = тест).

    Args:
        successes_a: успехи в контроле.
        n_a: размер контроля.
        successes_b: успехи в тесте.
        n_b: размер теста.
        prior_alpha: параметр α априорного Beta (1 = равномерное).
        prior_beta: параметр β априорного Beta.
        loss_threshold: порог ожидаемых потерь ε (практическая значимость).
        n_samples: число сэмплов Монте-Карло из апостериорных распределений.
        ci: уровень доверия для интервала на относительный lift.
        seed: seed генератора.

    Returns:
        :class:`BayesResult`.
    """
    if min(n_a, n_b) <= 0:
        raise ValueError("Размеры групп должны быть положительны.")
    for s, n in ((successes_a, n_a), (successes_b, n_b)):
        if not 0 <= s <= n:
            raise ValueError("Число успехов вне диапазона [0, n].")

    rng = np.random.default_rng(seed)
    a_post = (prior_alpha + successes_a, prior_beta + n_a - successes_a)
    b_post = (prior_alpha + successes_b, prior_beta + n_b - successes_b)

    theta_a = rng.beta(a_post[0], a_post[1], n_samples)
    theta_b = rng.beta(b_post[0], b_post[1], n_samples)

    prob_b_better = float(np.mean(theta_b > theta_a))
    # Ожидаемые потери: сколько теряем в конверсии, если ошиблись с выбором.
    expected_loss_b = float(np.mean(np.maximum(theta_a - theta_b, 0.0)))
    expected_loss_a = float(np.mean(np.maximum(theta_b - theta_a, 0.0)))

    with np.errstate(divide="ignore", invalid="ignore"):
        lift = theta_b / theta_a - 1.0
    lift = lift[np.isfinite(lift)]
    lo_q, hi_q = (1.0 - ci) / 2.0, 1.0 - (1.0 - ci) / 2.0
    lift_ci = (float(np.quantile(lift, lo_q)), float(np.quantile(lift, hi_q)))
    lift_mean = float(np.mean(lift))

    if expected_loss_b < loss_threshold:
        rec = "раскатывать B (ожидаемые потери ниже порога)"
    elif expected_loss_a < loss_threshold:
        rec = "оставить A (B не лучше в пределах порога)"
    else:
        rec = "нужно больше данных (потери обеих сторон выше порога)"

    return BayesResult(
        prob_b_better=prob_b_better,
        expected_loss_choose_b=expected_loss_b,
        expected_loss_choose_a=expected_loss_a,
        lift_mean=lift_mean,
        lift_ci=lift_ci,
        posterior_a=(float(a_post[0]), float(a_post[1])),
        posterior_b=(float(b_post[0]), float(b_post[1])),
        loss_threshold=loss_threshold,
        recommendation=rec,
    )
