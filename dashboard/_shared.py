"""Общие помощники для страниц дашборда: тема Plotly, форматирование, данные."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from abtest.datagen import ensure_sample_datasets

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Датасеты, поставляемые с проектом (генерируются `make data`).
BUNDLED_DATASETS = {
    "Конверсия (реальный эффект +8%)": "experiment_conversion.csv",
    "Непрерывная метрика (выручка, эффект +2.0)": "experiment_continuous.csv",
    "Поломанный сплит (SRM + дисбаланс)": "experiment_srm_broken.csv",
}


def plotly_template() -> str:
    """Выбрать тему Plotly в тон текущей теме Streamlit."""
    try:
        base = st.get_option("theme.base")
    except Exception:  # noqa: BLE001
        base = "light"
    return "plotly_dark" if base == "dark" else "plotly_white"


def page_header(title: str, subtitle: str) -> None:
    """Единый заголовок страницы."""
    st.title(title)
    st.caption(subtitle)


@st.cache_data(show_spinner="Готовлю демонстрационные данные...")
def load_bundled(name: str) -> pd.DataFrame:
    """Загрузить встроенный датасет по имени.

    Датасеты не хранятся в репозитории - при первом обращении они
    детерминированно генерируются в каталог ``data/``.
    """
    ensure_sample_datasets(DATA_DIR)
    return pd.read_csv(DATA_DIR / BUNDLED_DATASETS[name])


def metric_delta_color(good: bool) -> str:
    return "normal" if good else "inverse"
