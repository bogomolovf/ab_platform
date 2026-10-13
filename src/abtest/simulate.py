"""Монте-Карло симуляции: калибровка, мощность, CUPED и вред от подглядывания.

Модуль наглядно проверяет корректность методов на данных с *известной* истиной:

* :func:`simulate_false_positive_rate` - под H0 доля значимых результатов ~= α
  (калибровка ошибки I рода);
* :func:`simulate_power` - эмпирическая мощность совпадает с аналитической;
* :func:`simulate_cuped` - CUPED снижает дисперсию и повышает мощность;
* :func:`simulate_peeking` - наивное ежедневное подглядывание раздувает ошибку
  I рода, а mSPRT и alpha-spending её контролируют.

Функции разделены на *вычисление* (возвращают dataclass с массивами) и
*визуализацию* (строят фигуры Plotly), чтобы их можно было переиспользовать и в
дашборде, и в CLI. Запуск ``python -m abtest.simulate --outdir docs/figures``
сохраняет графики в PNG.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from scipy import stats

from abtest import power as power_mod
from abtest.sequential import alpha_spending_boundaries, msprt_lambda, suggest_tau2

__all__ = [
    "FPRResult",
    "PowerSimResult",
    "CupedSimResult",
    "PeekingResult",
    "simulate_false_positive_rate",
    "simulate_power",
    "simulate_cuped",
    "simulate_peeking",
    "fig_false_positive_rate",
    "fig_power",
    "fig_cuped",
    "fig_peeking",
    "main",
]

# Палитра, читаемая в тёмной и светлой теме.
_COLORS = {
    "control": "#6b7280",
    "naive": "#ef4444",
    "msprt": "#3b82f6",
    "spending": "#10b981",
    "with": "#3b82f6",
    "without": "#9ca3af",
    "target": "#f59e0b",
    "empirical": "#3b82f6",
    "analytic": "#10b981",
}


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Доверительный интервал Уилсона для доли (для оценки FPR/мощности)."""
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (float(center - half), float(center + half))


# --------------------------------------------------------------------------- #
# 1. Калибровка ошибки I рода (A/A)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FPRResult:
    """Результат симуляции false positive rate под H0."""

    metric_type: str
    fpr: float
    ci: tuple[float, float]
    alpha: float
    n_sims: int
    n_per_group: int
    pvalues: np.ndarray


def simulate_false_positive_rate(
    metric_type: Literal["conversion", "continuous"] = "conversion",
    n_sims: int = 5000,
    n_per_group: int = 5000,
    alpha: float = 0.05,
    baseline_rate: float = 0.1,
    sd: float = 1.0,
    seed: int | None = 0,
) -> FPRResult:
    """A/A-симуляция: под нулём доля ложноположительных должна быть ~= α.

    Args:
        metric_type: тип метрики.
        n_sims: число симуляций.
        n_per_group: размер каждой группы.
        alpha: уровень значимости.
        baseline_rate: конверсия под H0 (для долей).
        sd: стандартное отклонение (для непрерывной метрики).
        seed: seed генератора.

    Returns:
        :class:`FPRResult`.
    """
    rng = np.random.default_rng(seed)
    n = n_per_group

    if metric_type == "conversion":
        xc = rng.binomial(n, baseline_rate, n_sims)
        xt = rng.binomial(n, baseline_rate, n_sims)
        pc, pt = xc / n, xt / n
        ppool = (xc + xt) / (2 * n)
        se = np.sqrt(ppool * (1 - ppool) * (2.0 / n))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(se > 0, (pt - pc) / se, 0.0)
        pvalues = 2.0 * stats.norm.sf(np.abs(z))
    elif metric_type == "continuous":
        c = rng.normal(0.0, sd, size=(n_sims, n))
        t = rng.normal(0.0, sd, size=(n_sims, n))
        mc, mt = c.mean(1), t.mean(1)
        vc, vt = c.var(1, ddof=1), t.var(1, ddof=1)
        se = np.sqrt(vc / n + vt / n)
        tstat = (mt - mc) / se
        df = (vc / n + vt / n) ** 2 / ((vc / n) ** 2 / (n - 1) + (vt / n) ** 2 / (n - 1))
        pvalues = 2.0 * stats.t.sf(np.abs(tstat), df)
    else:
        raise ValueError("metric_type должен быть 'conversion' или 'continuous'.")

    k = int(np.sum(pvalues < alpha))
    return FPRResult(
        metric_type=metric_type,
        fpr=k / n_sims,
        ci=_wilson_ci(k, n_sims),
        alpha=alpha,
        n_sims=n_sims,
        n_per_group=n_per_group,
        pvalues=pvalues,
    )


# --------------------------------------------------------------------------- #
# 2. Мощность: эмпирика vs аналитика
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PowerSimResult:
    """Сравнение эмпирической и аналитической мощности по сетке эффектов."""

    effects: np.ndarray
    empirical_power: np.ndarray
    analytic_power: np.ndarray
    empirical_ci_low: np.ndarray
    empirical_ci_high: np.ndarray
    n_per_group: int
    alpha: float
    sd: float


def simulate_power(
    effects: np.ndarray | None = None,
    n_per_group: int = 500,
    sd: float = 1.0,
    alpha: float = 0.05,
    n_sims: int = 2000,
    seed: int | None = 0,
) -> PowerSimResult:
    """Эмпирическая мощность t-теста против аналитической формулы.

    Args:
        effects: сетка абсолютных эффектов; по умолчанию 0..0.4*sd.
        n_per_group: размер группы.
        sd: стандартное отклонение метрики.
        alpha: уровень значимости.
        n_sims: число симуляций на каждую точку.
        seed: seed генератора.

    Returns:
        :class:`PowerSimResult`.
    """
    if effects is None:
        effects = np.linspace(0.0, 0.4 * sd, 9)
    effects = np.asarray(effects, dtype=float)
    rng = np.random.default_rng(seed)
    n = n_per_group

    emp = np.empty(effects.size)
    ci_lo = np.empty(effects.size)
    ci_hi = np.empty(effects.size)
    ana = np.empty(effects.size)
    for i, eff in enumerate(effects):
        c = rng.normal(0.0, sd, size=(n_sims, n))
        t = rng.normal(eff, sd, size=(n_sims, n))
        mc, mt = c.mean(1), t.mean(1)
        vc, vt = c.var(1, ddof=1), t.var(1, ddof=1)
        se = np.sqrt(vc / n + vt / n)
        tstat = (mt - mc) / se
        df = (vc / n + vt / n) ** 2 / ((vc / n) ** 2 / (n - 1) + (vt / n) ** 2 / (n - 1))
        pvalues = 2.0 * stats.t.sf(np.abs(tstat), df)
        k = int(np.sum(pvalues < alpha))
        emp[i] = k / n_sims
        ci_lo[i], ci_hi[i] = _wilson_ci(k, n_sims)
        ana[i] = power_mod.power_mean(eff, sd, n, n, alpha) if eff > 0 else alpha

    return PowerSimResult(
        effects=effects,
        empirical_power=emp,
        analytic_power=ana,
        empirical_ci_low=ci_lo,
        empirical_ci_high=ci_hi,
        n_per_group=n,
        alpha=alpha,
        sd=sd,
    )


# --------------------------------------------------------------------------- #
# 3. CUPED
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CupedSimResult:
    """Эффект CUPED: мощность и снижение дисперсии."""

    correlation: float
    power_without: float
    power_with: float
    power_without_ci: tuple[float, float]
    power_with_ci: tuple[float, float]
    var_reduction_observed: float
    var_reduction_theoretical: float
    n_per_group: int
    effect: float
    alpha: float


def simulate_cuped(
    correlation: float = 0.7,
    effect: float = 0.05,
    n_per_group: int = 800,
    sd: float = 1.0,
    alpha: float = 0.05,
    n_sims: int = 2000,
    seed: int | None = 0,
) -> CupedSimResult:
    """Симуляция выигрыша CUPED в мощности при заданной корреляции с пред-периодом.

    Args:
        correlation: корреляция метрики с пред-периодной ковариатой.
        effect: истинный абсолютный эффект.
        n_per_group: размер группы.
        sd: стандартное отклонение метрики.
        alpha: уровень значимости.
        n_sims: число симуляций.
        seed: seed.

    Returns:
        :class:`CupedSimResult`.
    """
    rng = np.random.default_rng(seed)
    n = n_per_group
    rho = float(np.clip(correlation, -0.999, 0.999))

    rej_without = 0
    rej_with = 0
    var_raw_acc = 0.0
    var_adj_acc = 0.0

    for _ in range(n_sims):
        x_c = rng.normal(0.0, sd, n)
        x_t = rng.normal(0.0, sd, n)
        eps_c = rng.normal(0.0, sd * np.sqrt(1 - rho**2), n)
        eps_t = rng.normal(0.0, sd * np.sqrt(1 - rho**2), n)
        y_c = rho * x_c + eps_c
        y_t = rho * x_t + eps_t + effect

        # Без CUPED
        p_raw = _welch_p(y_c, y_t)
        rej_without += int(p_raw < alpha)

        # CUPED
        x_all = np.concatenate([x_c, x_t])
        y_all = np.concatenate([y_c, y_t])
        theta = np.cov(y_all, x_all, ddof=1)[0, 1] / x_all.var(ddof=1)
        xbar = x_all.mean()
        yc_adj = y_c - theta * (x_c - xbar)
        yt_adj = y_t - theta * (x_t - xbar)
        p_adj = _welch_p(yc_adj, yt_adj)
        rej_with += int(p_adj < alpha)

        var_raw_acc += 0.5 * (y_c.var(ddof=1) + y_t.var(ddof=1))
        var_adj_acc += 0.5 * (yc_adj.var(ddof=1) + yt_adj.var(ddof=1))

    var_raw = var_raw_acc / n_sims
    var_adj = var_adj_acc / n_sims
    return CupedSimResult(
        correlation=rho,
        power_without=rej_without / n_sims,
        power_with=rej_with / n_sims,
        power_without_ci=_wilson_ci(rej_without, n_sims),
        power_with_ci=_wilson_ci(rej_with, n_sims),
        var_reduction_observed=float(1 - var_adj / var_raw),
        var_reduction_theoretical=float(rho**2),
        n_per_group=n,
        effect=effect,
        alpha=alpha,
    )


def _welch_p(c: np.ndarray, t: np.ndarray) -> float:
    _, p = stats.ttest_ind(t, c, equal_var=False)
    return float(p)


# --------------------------------------------------------------------------- #
# 4. Подглядывание (peeking)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PeekingResult:
    """Кумулятивная ошибка I рода при многократном подглядывании."""

    look_fractions: np.ndarray
    cumulative_fpr_naive: np.ndarray
    cumulative_fpr_msprt: np.ndarray
    cumulative_fpr_spending: np.ndarray
    alpha: float
    n_looks: int
    n_per_group: int
    n_sims: int


def simulate_peeking(
    n_per_group: int = 2000,
    n_looks: int = 20,
    alpha: float = 0.05,
    sd: float = 1.0,
    n_sims: int = 3000,
    seed: int | None = 0,
) -> PeekingResult:
    """Сравнить накопленную ошибку I рода трёх стратегий под H0.

    Стратегии: наивный z-тест на каждой проверке (без поправки), mSPRT
    (always-valid), alpha-spending (O'Brien-Fleming).

    Args:
        n_per_group: итоговый размер группы.
        n_looks: число равномерных промежуточных проверок.
        alpha: уровень значимости.
        sd: стандартное отклонение.
        n_sims: число симуляций.
        seed: seed.

    Returns:
        :class:`PeekingResult` с кумулятивной долей отвергнутых по проверкам.
    """
    rng = np.random.default_rng(seed)
    n = n_per_group
    looks = np.unique(np.linspace(n // n_looks, n, n_looks).astype(int))
    looks = looks[looks >= 2]
    info_frac = looks / looks[-1]

    # Границы alpha-spending считаем один раз.
    spending = alpha_spending_boundaries(info_frac, alpha=alpha, spending="obrien-fleming")
    z_bounds = spending.z_boundaries
    tau2 = suggest_tau2(0.05 * sd, sd, n)

    # Аккумуляторы «хотя бы раз пересёк границу к проверке k».
    ever_naive = np.zeros(looks.size)
    ever_msprt = np.zeros(looks.size)
    ever_spending = np.zeros(looks.size)

    batch = 500  # обрабатываем симуляции пачками ради памяти
    done = 0
    while done < n_sims:
        b = min(batch, n_sims - done)
        c = rng.normal(0.0, sd, size=(b, n))
        t = rng.normal(0.0, sd, size=(b, n))
        cs_c, cs_t = np.cumsum(c, axis=1), np.cumsum(t, axis=1)
        cq_c, cq_t = np.cumsum(c**2, axis=1), np.cumsum(t**2, axis=1)

        idx = looks - 1
        k_arr = looks.astype(float)
        mean_c = cs_c[:, idx] / k_arr
        mean_t = cs_t[:, idx] / k_arr
        var_c = (cq_c[:, idx] - k_arr * mean_c**2) / (k_arr - 1)
        var_t = (cq_t[:, idx] - k_arr * mean_t**2) / (k_arr - 1)
        pooled = 0.5 * (var_c + var_t)
        se = np.sqrt(pooled * (2.0 / k_arr))
        diff = mean_t - mean_c
        z = np.where(se > 0, diff / se, 0.0)

        # Наивный: p<alpha на любой проверке.
        p_naive = 2.0 * stats.norm.sf(np.abs(z))
        rej_naive = p_naive < alpha
        # mSPRT: Λ>=1/alpha.
        var_est = pooled * (2.0 / k_arr)
        lam = msprt_lambda(diff, np.maximum(var_est, 1e-12), tau2)
        rej_msprt = lam >= 1.0 / alpha
        # alpha-spending: |Z_k|>= z_bound_k.
        rej_spending = np.abs(z) >= z_bounds[np.newaxis, :]

        ever_naive += np.cumsum(rej_naive, axis=1).astype(bool).sum(axis=0)
        ever_msprt += np.cumsum(rej_msprt, axis=1).astype(bool).sum(axis=0)
        ever_spending += np.cumsum(rej_spending, axis=1).astype(bool).sum(axis=0)
        done += b

    return PeekingResult(
        look_fractions=info_frac,
        cumulative_fpr_naive=ever_naive / n_sims,
        cumulative_fpr_msprt=ever_msprt / n_sims,
        cumulative_fpr_spending=ever_spending / n_sims,
        alpha=alpha,
        n_looks=looks.size,
        n_per_group=n,
        n_sims=n_sims,
    )


# --------------------------------------------------------------------------- #
# Визуализация (Plotly)
# --------------------------------------------------------------------------- #
def _apply_layout(fig: go.Figure, title: str, template: str) -> go.Figure:
    fig.update_layout(
        title=title,
        template=template,
        font=dict(size=13),
        margin=dict(l=60, r=30, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def fig_false_positive_rate(result: FPRResult, template: str = "plotly_white") -> go.Figure:
    """Гистограмма p-values под H0 (должна быть равномерной) + фактический FPR."""
    fig = go.Figure()
    fig.add_histogram(
        x=result.pvalues,
        nbinsx=20,
        histnorm="probability density",
        name="p-values",
        marker_color=_COLORS["empirical"],
        opacity=0.8,
    )
    fig.add_hline(
        y=1.0, line_dash="dash", line_color=_COLORS["target"], annotation_text="равномерное U(0,1)"
    )
    fig = _apply_layout(
        fig,
        f"Калибровка под H0: FPR={result.fpr:.3f} "
        f"(ожидается α={result.alpha}); ДИ=[{result.ci[0]:.3f}; {result.ci[1]:.3f}]",
        template,
    )
    fig.update_xaxes(title="p-value")
    fig.update_yaxes(title="плотность")
    return fig


def fig_power(result: PowerSimResult, template: str = "plotly_white") -> go.Figure:
    """Кривая мощности: эмпирика (с ДИ) против аналитики."""
    fig = go.Figure()
    fig.add_scatter(
        x=result.effects,
        y=result.analytic_power,
        mode="lines",
        name="аналитическая",
        line=dict(color=_COLORS["analytic"], width=3),
    )
    fig.add_scatter(
        x=result.effects,
        y=result.empirical_power,
        mode="markers",
        name="эмпирическая",
        marker=dict(color=_COLORS["empirical"], size=9),
        error_y=dict(
            type="data",
            symmetric=False,
            array=result.empirical_ci_high - result.empirical_power,
            arrayminus=result.empirical_power - result.empirical_ci_low,
        ),
    )
    fig = _apply_layout(
        fig, f"Мощность t-теста (n={result.n_per_group}/группа, α={result.alpha})", template
    )
    fig.update_xaxes(title="истинный эффект (абсолютный)")
    fig.update_yaxes(title="мощность", range=[0, 1.02])
    return fig


def fig_cuped(result: CupedSimResult, template: str = "plotly_white") -> go.Figure:
    """Столбики: мощность без и с CUPED."""
    fig = go.Figure()
    fig.add_bar(
        x=["без CUPED", "с CUPED"],
        y=[result.power_without, result.power_with],
        marker_color=[_COLORS["without"], _COLORS["with"]],
        error_y=dict(
            type="data",
            symmetric=False,
            array=[
                result.power_without_ci[1] - result.power_without,
                result.power_with_ci[1] - result.power_with,
            ],
            arrayminus=[
                result.power_without - result.power_without_ci[0],
                result.power_with - result.power_with_ci[0],
            ],
        ),
        text=[f"{result.power_without:.2f}", f"{result.power_with:.2f}"],
        textposition="outside",
    )
    fig = _apply_layout(
        fig,
        f"CUPED (ρ={result.correlation:.2f}): снижение дисперсии "
        f"{result.var_reduction_observed:.1%} (теория ρ^2={result.var_reduction_theoretical:.1%})",
        template,
    )
    fig.update_yaxes(title="мощность", range=[0, 1.05])
    return fig


def fig_peeking(result: PeekingResult, template: str = "plotly_white") -> go.Figure:
    """Кумулятивная ошибка I рода по проверкам: наивно vs mSPRT vs alpha-spending."""
    x = np.arange(1, result.look_fractions.size + 1)
    fig = go.Figure()
    fig.add_scatter(
        x=x,
        y=result.cumulative_fpr_naive,
        mode="lines+markers",
        name="наивный (без поправки)",
        line=dict(color=_COLORS["naive"], width=3),
    )
    fig.add_scatter(
        x=x,
        y=result.cumulative_fpr_msprt,
        mode="lines+markers",
        name="mSPRT (always-valid)",
        line=dict(color=_COLORS["msprt"], width=3),
    )
    fig.add_scatter(
        x=x,
        y=result.cumulative_fpr_spending,
        mode="lines+markers",
        name="alpha-spending (OBF)",
        line=dict(color=_COLORS["spending"], width=3),
    )
    fig.add_hline(
        y=result.alpha,
        line_dash="dash",
        line_color=_COLORS["target"],
        annotation_text=f"номинал α={result.alpha}",
    )
    fig = _apply_layout(fig, "Вред подглядывания: накопленная ошибка I рода под H0", template)
    fig.update_xaxes(title="номер проверки")
    fig.update_yaxes(title="доля ложных отвержений (FPR)")
    return fig


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    """CLI: прогнать все симуляции и сохранить графики в PNG."""
    parser = argparse.ArgumentParser(description="Монте-Карло симуляции A/B-методов")
    parser.add_argument("--outdir", default="docs/figures", help="каталог для PNG")
    parser.add_argument("--quick", action="store_true", help="быстрый режим (меньше симуляций)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ns = 1000 if args.quick else 5000
    ns_small = 500 if args.quick else 2000

    print(">> [1/4] Калибровка FPR (A/A)...")
    fpr = simulate_false_positive_rate(n_sims=ns, seed=args.seed)
    print(f"   FPR={fpr.fpr:.4f} (α={fpr.alpha}), ДИ={fpr.ci}")

    print(">> [2/4] Мощность: эмпирика vs аналитика...")
    pw = simulate_power(n_per_group=500, n_sims=ns_small, seed=args.seed)
    max_gap = float(np.max(np.abs(pw.empirical_power - pw.analytic_power)))
    print(f"   макс. расхождение эмпирика/аналитика = {max_gap:.3f}")

    print(">> [3/4] CUPED...")
    cu = simulate_cuped(correlation=0.7, n_sims=ns_small, seed=args.seed)
    print(
        f"   мощность {cu.power_without:.3f} -> {cu.power_with:.3f}, "
        f"снижение дисперсии {cu.var_reduction_observed:.1%}"
    )

    print(">> [4/4] Подглядывание...")
    pk = simulate_peeking(n_sims=ns, seed=args.seed)
    print(
        f"   наивный FPR={pk.cumulative_fpr_naive[-1]:.3f}, "
        f"mSPRT={pk.cumulative_fpr_msprt[-1]:.3f}, "
        f"alpha-spending={pk.cumulative_fpr_spending[-1]:.3f}"
    )

    figs = {
        "sim_false_positive_rate": fig_false_positive_rate(fpr),
        "sim_power_curve": fig_power(pw),
        "sim_cuped": fig_cuped(cu),
        "sim_peeking": fig_peeking(pk),
    }
    for name, fig in figs.items():
        _save_figure(fig, outdir, name)
    print(">> Готово.")


def _save_figure(fig: go.Figure, outdir: Path, name: str) -> None:
    """Сохранить фигуру в PNG; при отсутствии движка рендеринга - в HTML."""
    png_path = outdir / f"{name}.png"
    try:
        fig.write_image(str(png_path), width=900, height=520, scale=2)
        print(f"   сохранено: {png_path}")
    except Exception as exc:  # noqa: BLE001 - деградация к HTML, а не падение
        html_path = outdir / f"{name}.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        print(
            f"   PNG недоступен ({type(exc).__name__}); сохранён интерактивный "
            f"HTML: {html_path}\n"
            "   (для PNG установите Chrome: `plotly_get_chrome` или "
            "переменную окружения BROWSER_PATH)"
        )


if __name__ == "__main__":
    main()
