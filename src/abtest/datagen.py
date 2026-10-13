"""Генератор синтетических данных экспериментов.

Создаёт воспроизводимые (фиксируемый seed) датасеты с известным *истинным*
эффектом, что позволяет проверять корректность методов. Поддерживаются:

* конверсионные и непрерывные метрики;
* пред-периодная ковариата с заданной корреляцией (для демонстрации CUPED);
* умышленные «поломки»: SRM (перекос сплита) и дисбаланс ковариат между группами.

Непрерывная метрика строится так, чтобы её маргинальная дисперсия и корреляция
с пред-периодом были ровно заданными:

.. math::

    Y = \\mu_Y + \\rho \\frac{\\sigma_Y}{\\sigma_X}(X - \\mu_X) + \\varepsilon,
    \\quad \\varepsilon \\sim N(0, \\sigma_Y^2 (1 - \\rho^2))

плюс аддитивный эффект воздействия в тестовой группе.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "ExperimentConfig",
    "ExperimentData",
    "generate_experiment",
    "ensure_sample_datasets",
    "SAMPLE_SPECS",
    "main",
]

MetricType = Literal["conversion", "continuous"]


@dataclass
class ExperimentConfig:
    """Конфигурация генератора синтетического эксперимента."""

    n_users: int = 20_000
    metric_type: MetricType = "conversion"
    treatment_share: float = 0.5  # запланированная доля теста
    true_effect: float = 0.0  # относительный (доли) или абсолютный (среднее)
    # Параметры конверсии
    baseline_rate: float = 0.10
    # Параметры непрерывной метрики
    baseline_mean: float = 100.0
    sd: float = 20.0
    # Пред-период
    pre_correlation: float = 0.7
    # Поломки
    srm: bool = False
    srm_actual_share: float = 0.47  # фактическая доля теста при поломке
    covariate_imbalance: bool = False
    seed: int = 42
    covariates: tuple[str, ...] = field(default_factory=lambda: ("is_mobile", "tenure_days"))


@dataclass
class ExperimentData:
    """Сгенерированный датасет эксперимента."""

    data: pd.DataFrame
    config: ExperimentConfig
    true_effect: float
    metric_type: str

    def group_arrays(self, column: str) -> tuple[np.ndarray, np.ndarray]:
        """Вернуть значения ``column`` для контроля и теста как два массива."""
        c = self.data.loc[self.data["group"] == "control", column].to_numpy()
        t = self.data.loc[self.data["group"] == "treatment", column].to_numpy()
        return c, t

    @property
    def counts(self) -> dict[str, int]:
        return self.data["group"].value_counts().to_dict()


def generate_experiment(config: ExperimentConfig | None = None, **overrides) -> ExperimentData:
    """Сгенерировать синтетический эксперимент.

    Args:
        config: конфигурация; если ``None`` - создаётся из ``overrides``.
        **overrides: переопределения полей :class:`ExperimentConfig`.

    Returns:
        :class:`ExperimentData` с датафреймом (user_id, group, pre_metric,
        metric, [covariates]).
    """
    if config is None:
        config = ExperimentConfig(**overrides)
    elif overrides:
        config = ExperimentConfig(**{**config.__dict__, **overrides})

    rng = np.random.default_rng(config.seed)
    n = config.n_users

    # --- Назначение в группы ---------------------------------------------- #
    share = config.srm_actual_share if config.srm else config.treatment_share
    assign = rng.random(n) < share
    group = np.where(assign, "treatment", "control")
    is_treat = assign.astype(float)

    # --- Пред-периодная ковариата (стандартизованная) --------------------- #
    pre_std = rng.standard_normal(n)

    df = pd.DataFrame({"user_id": np.arange(n), "group": group})

    rho = float(np.clip(config.pre_correlation, -0.999, 0.999))

    if config.metric_type == "continuous":
        mu_x, sigma_x = config.baseline_mean, config.sd
        mu_y, sigma_y = config.baseline_mean, config.sd
        pre_metric = mu_x + sigma_x * pre_std
        noise = rng.standard_normal(n) * sigma_y * np.sqrt(1.0 - rho**2)
        metric = mu_y + rho * (sigma_y / sigma_x) * (pre_metric - mu_x) + noise
        metric = metric + is_treat * config.true_effect
        df["pre_metric"] = pre_metric
        df["metric"] = metric

    elif config.metric_type == "conversion":
        from scipy import stats as _st

        # Пробит-модель: латентная склонность, коррелирующая с пред-периодом.
        # loading a управляет силой связи пред-периода с конверсией.
        a = abs(rho)
        latent = pre_std
        # Сдвиги подобраны так, чтобы маргинальная конверсия совпадала с baseline
        # в контроле и с baseline*(1+effect) в тесте: E[Φ(m + a*latent)] = Φ(m/sqrt(1+a^2)).
        z_base = _st.norm.ppf(config.baseline_rate)
        p_treat = float(np.clip(config.baseline_rate * (1.0 + config.true_effect), 1e-6, 1 - 1e-6))
        z_treat = _st.norm.ppf(p_treat)
        scale = np.sqrt(1.0 + a**2)
        m = np.where(is_treat > 0, z_treat * scale, z_base * scale)
        prob = _st.norm.cdf(m + a * latent)
        converted = (rng.random(n) < prob).astype(int)
        # Пред-период как наблюдаемый прокси латентной склонности.
        df["pre_metric"] = latent
        df["metric"] = converted
        df["converted"] = converted
    else:
        raise ValueError("metric_type должен быть 'conversion' или 'continuous'.")

    # --- Ковариаты (с опциональным дисбалансом) --------------------------- #
    if "is_mobile" in config.covariates:
        p_mobile = np.where((df["group"] == "treatment") & config.covariate_imbalance, 0.75, 0.6)
        df["is_mobile"] = (rng.random(n) < p_mobile).astype(int)
    if "tenure_days" in config.covariates:
        shift = np.where((df["group"] == "treatment") & config.covariate_imbalance, 40.0, 0.0)
        df["tenure_days"] = np.clip(rng.gamma(2.0, 30.0, n) + shift, 0, None).round(1)

    return ExperimentData(
        data=df,
        config=config,
        true_effect=config.true_effect,
        metric_type=config.metric_type,
    )


# Спецификации трёх демонстрационных датасетов (детерминированы seed-ом).
# Используются и CLI (`make data`), и дашбордом - данные не хранятся в git, а
# воспроизводятся из этих спецификаций при первом обращении.
SAMPLE_SPECS: dict[str, dict] = {
    "experiment_conversion.csv": dict(
        metric_type="conversion",
        n_users=80_000,
        baseline_rate=0.12,
        true_effect=0.08,
        pre_correlation=0.5,
        seed_offset=0,
    ),
    "experiment_continuous.csv": dict(
        metric_type="continuous",
        n_users=20_000,
        baseline_mean=100.0,
        sd=25.0,
        true_effect=2.0,
        pre_correlation=0.8,
        seed_offset=1,
    ),
    "experiment_srm_broken.csv": dict(
        metric_type="conversion",
        n_users=40_000,
        baseline_rate=0.12,
        true_effect=0.0,
        srm=True,
        srm_actual_share=0.46,
        covariate_imbalance=True,
        seed_offset=2,
    ),
}


def ensure_sample_datasets(
    outdir: str | Path = "data", seed: int = 42, force: bool = False
) -> dict[str, Path]:
    """Создать (или переиспользовать) демонстрационные датасеты в каталоге.

    Датасеты полностью детерминированы, поэтому в репозитории их не хранят -
    они воспроизводятся здесь. Существующие файлы не перезаписываются, если
    ``force=False``.

    Args:
        outdir: каталог для CSV.
        seed: базовый seed (к нему прибавляется ``seed_offset`` из спецификации).
        force: перегенерировать, даже если файл уже существует.

    Returns:
        Словарь ``имя_файла -> путь``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for fname, spec in SAMPLE_SPECS.items():
        path = outdir / fname
        if force or not path.exists():
            spec = dict(spec)
            offset = spec.pop("seed_offset")
            exp = generate_experiment(seed=seed + offset, **spec)
            exp.data.to_csv(path, index=False)
        result[fname] = path
    return result


def main(argv: list[str] | None = None) -> None:
    """CLI: сгенерировать примеры датасетов в каталог (по умолчанию ``data``)."""
    parser = argparse.ArgumentParser(description="Генератор синтетических A/B-данных")
    parser.add_argument("--outdir", default="data", help="каталог для сохранения CSV")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    paths = ensure_sample_datasets(args.outdir, seed=args.seed, force=True)
    print(f"Сохранено {len(paths)} датасета(ов) в {Path(args.outdir).resolve()}")


if __name__ == "__main__":
    main()
