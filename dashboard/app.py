"""A/B Experimentation Platform - Streamlit-дашборд.

Запуск: ``streamlit run dashboard/app.py`` (или ``make dashboard``).

Дашборд состоит из трёх инструментов:
* Планирование - калькулятор размера выборки, мощности и MDE;
* Анализ эксперимента - загрузка данных, тесты, ДИ и вердикт;
* Симуляции - интерактивные Монте-Карло демонстрации методов.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="A/B Experimentation Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)


def home() -> None:
    """Обзорная страница."""
    st.title("A/B Experimentation Platform")
    st.markdown(
        "Библиотека и дашборд для A/B-тестов. Весь путь эксперимента: "
        "планирование, проверка дизайна, анализ, вердикт."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Планирование")
        st.write("Размер выборки и мощность для долей и средних, расчёт MDE, кривые мощности.")
    with col2:
        st.subheader("Анализ")
        st.write(
            "z-тест долей, тест Уэлча, бутстрап, Манна-Уитни, CUPED, SRM, "
            "поправки на множественность, байес и вердикт."
        )
    with col3:
        st.subheader("Симуляции")
        st.write(
            "Калибровка FPR, реальная мощность, выигрыш CUPED и вред от "
            "подглядывания - на Монте-Карло."
        )

    st.divider()
    st.markdown("""
        #### Что внутри
        - Ошибка I рода под контролем (A/A-калибровка) и корректная мощность.
        - CUPED - снижение дисперсии и рост мощности без увеличения выборки.
        - Sequential testing (mSPRT, alpha-spending) против проблемы подглядывания.
        - SRM и баланс ковариат - проверка дизайна до анализа.
        - Множественные сравнения (Bonferroni, Benjamini-Hochberg).
        - Вердикт: раскатывать / не раскатывать / нужно больше данных.

        Выберите инструмент в боковой панели.
        """)
    st.info(
        "На странице «Анализ эксперимента» есть встроенные датасеты, "
        "включая заведомо поломанный сплит - удобно посмотреть, как ловится SRM."
    )


nav = st.navigation(
    {
        "Старт": [st.Page(home, title="Обзор", default=True)],
        "Инструменты": [
            st.Page("pages/planning.py", title="Планирование"),
            st.Page("pages/analysis.py", title="Анализ эксперимента"),
            st.Page("pages/simulations.py", title="Симуляции"),
        ],
    }
)
nav.run()
