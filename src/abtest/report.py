"""Автоматическая человекочитаемая интерпретация эксперимента.

Собирает воедино статистический вывод и превращает его в понятную рекомендацию:
значим ли результат, каков размер эффекта с ДИ, отличается ли *практическая*
значимость от статистической, и что делать (раскатывать / не раскатывать /
собрать больше данных). Дополнительно формирует предупреждения о рисках:
SRM, недоборе мощности и множественных сравнениях.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from abtest.srm import SRMResult

__all__ = ["Verdict", "interpret"]


@runtime_checkable
class _EffectResult(Protocol):
    """Минимальный интерфейс результата теста, который умеет интерпретатор."""

    absolute_effect: float
    absolute_ci: tuple[float, float]
    relative_effect: float
    p_value: float
    significant: bool
    alpha: float


@dataclass
class Verdict:
    """Структурированный вердикт по эксперименту."""

    headline: str
    recommendation: str
    statistically_significant: bool
    practically_significant: bool | None
    effect: float
    ci: tuple[float, float]
    relative_effect: float
    p_value: float
    warnings: list[str] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Готовый многострочный текст вывода на русском."""
        lines = [
            f"ВЕРДИКТ: {self.headline}",
            "",
            f"Рекомендация: {self.recommendation}",
            "",
            f"Эффект (абсолютный): {self.effect:+.4f}, "
            f"95% ДИ [{self.ci[0]:+.4f}; {self.ci[1]:+.4f}]",
        ]
        if self.relative_effect == self.relative_effect:  # не NaN
            lines.append(f"Относительный lift: {self.relative_effect:+.2%}")
        lines.append(f"p-value: {self.p_value:.4g} (порог α={self.details.get('alpha')})")
        lines.append(
            f"Статистическая значимость: {'да' if self.statistically_significant else 'нет'}"
        )
        if self.practically_significant is not None:
            lines.append(
                f"Практическая значимость: " f"{'да' if self.practically_significant else 'нет'}"
            )
        if self.warnings:
            lines.append("")
            lines.append("Предупреждения:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)


def interpret(
    result: _EffectResult,
    practical_threshold: float | None = None,
    srm_result: SRMResult | None = None,
    achieved_power: float | None = None,
    adjusted_pvalue: float | None = None,
    n_comparisons: int = 1,
    higher_is_better: bool = True,
) -> Verdict:
    """Построить вердикт по результату анализа.

    Args:
        result: результат теста (доли или средние) с полями эффекта и ДИ.
        practical_threshold: минимальный практически значимый абсолютный эффект.
            Если задан, оценивается практическая значимость по ДИ.
        srm_result: результат проверки SRM (если был).
        achieved_power: апостериорная/плановая мощность (для предупреждения о недоборе).
        adjusted_pvalue: p-value после поправки на множественность (если применимо).
        n_comparisons: число одновременных сравнений (для предупреждения).
        higher_is_better: считается ли рост метрики положительным исходом.

    Returns:
        :class:`Verdict`.
    """
    alpha = result.alpha
    p_used = adjusted_pvalue if adjusted_pvalue is not None else result.p_value
    stat_sig = p_used < alpha
    effect = result.absolute_effect
    lo, hi = result.absolute_ci
    warnings: list[str] = []

    # Направление эффекта.
    good_direction = (effect > 0) if higher_is_better else (effect < 0)

    # Практическая значимость: весь ДИ по «хорошую» сторону от порога.
    practically_sig: bool | None = None
    if practical_threshold is not None:
        thr = abs(practical_threshold)
        if higher_is_better:
            practically_sig = lo >= thr
            ruled_out = hi < thr  # можно исключить практически значимый рост
        else:
            practically_sig = hi <= -thr
            ruled_out = lo > -thr
    else:
        ruled_out = False

    # --- Предупреждения ---------------------------------------------------- #
    srm_blocker = False
    if srm_result is not None and srm_result.srm_detected:
        srm_blocker = True
        warnings.append(
            f"Обнаружен SRM (p={srm_result.p_value:.2g}): сплитование сломано, "
            "результатам доверять нельзя до устранения причины."
        )
    if achieved_power is not None and achieved_power < 0.8:
        warnings.append(
            f"Недобор мощности (power~={achieved_power:.0%} < 80%): "
            "отрицательный результат может быть ложным (мало данных)."
        )
    if n_comparisons > 1 and adjusted_pvalue is None:
        warnings.append(
            f"Проводится {n_comparisons} сравнений, но поправка на множественность "
            "не применена - часть значимых находок может быть ложной."
        )
    if stat_sig and not good_direction:
        warnings.append("Значимый эффект направлен в «нежелательную» сторону - это ухудшение.")

    # --- Рекомендация ------------------------------------------------------ #
    if srm_blocker:
        headline = "Результат недействителен из-за SRM"
        recommendation = "Не принимать решение: сначала починить рандомизацию/логирование."
    elif stat_sig and good_direction and (practically_sig in (True, None)):
        headline = "Значимый положительный эффект"
        if practically_sig is True:
            recommendation = "Раскатывать вариант: эффект статистически и практически значим."
        else:
            recommendation = (
                "Скорее раскатывать: эффект статистически значим "
                "(порог практической значимости не задан)."
            )
    elif stat_sig and good_direction and practically_sig is False:
        headline = "Значимо статистически, но не практически"
        recommendation = (
            "Решать по экономике: эффект реален, но меньше практического порога - "
            "оценить стоимость внедрения."
        )
    elif stat_sig and not good_direction:
        headline = "Значимое ухудшение"
        recommendation = "Не раскатывать: вариант статистически значимо хуже контроля."
    elif (not stat_sig) and ruled_out:
        headline = "Значимого эффекта нет"
        recommendation = (
            "Не раскатывать: практически значимый эффект можно исключить "
            "(ДИ не выходит за порог)."
        )
    else:
        headline = "Результат неопределённый"
        recommendation = (
            "Нужно больше данных: ДИ слишком широк, чтобы подтвердить или "
            "исключить значимый эффект."
        )

    return Verdict(
        headline=headline,
        recommendation=recommendation,
        statistically_significant=stat_sig,
        practically_significant=practically_sig,
        effect=float(effect),
        ci=(float(lo), float(hi)),
        relative_effect=float(getattr(result, "relative_effect", float("nan"))),
        p_value=float(p_used),
        warnings=warnings,
        details={
            "alpha": alpha,
            "raw_pvalue": result.p_value,
            "adjusted_pvalue": adjusted_pvalue,
            "achieved_power": achieved_power,
            "practical_threshold": practical_threshold,
            "n_comparisons": n_comparisons,
        },
    )
