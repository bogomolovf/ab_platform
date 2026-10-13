"""Страница «Симуляции»: интерактивные Монте-Карло демонстрации методов."""

from __future__ import annotations

import streamlit as st

from _shared import page_header, plotly_template
from abtest import simulate

page_header(
    "Монте-Карло симуляции",
    "Проверка методов на данных с известной истиной: калибровка, мощность, "
    "CUPED и вред от подглядывания",
)
tmpl = plotly_template()


@st.cache_data(show_spinner=True)
def _fpr(metric_type, n_sims, n_per_group):
    return simulate.simulate_false_positive_rate(
        metric_type=metric_type, n_sims=n_sims, n_per_group=n_per_group, seed=0
    )


@st.cache_data(show_spinner=True)
def _power(n_per_group, n_sims):
    return simulate.simulate_power(n_per_group=n_per_group, n_sims=n_sims, seed=0)


@st.cache_data(show_spinner=True)
def _cuped(correlation, n_per_group, n_sims):
    return simulate.simulate_cuped(
        correlation=correlation, n_per_group=n_per_group, n_sims=n_sims, seed=0
    )


@st.cache_data(show_spinner=True)
def _peeking(n_per_group, n_looks, n_sims):
    return simulate.simulate_peeking(
        n_per_group=n_per_group, n_looks=n_looks, n_sims=n_sims, seed=0
    )


tab1, tab2, tab3, tab4 = st.tabs(["Калибровка (A/A)", "Мощность", "CUPED", "Подглядывание"])

with tab1:
    st.markdown(
        "Под нулевой гипотезой доля значимых результатов должна равняться α. "
        "Если это так - тест откалиброван и не завышает ошибку I рода."
    )
    c1, c2 = st.columns(2)
    metric_type = c1.selectbox("Метрика", ["conversion", "continuous"])
    n_sims = c2.select_slider("Число симуляций", [1000, 2000, 5000, 10000], value=5000)
    n_pg = st.slider("Размер группы", 500, 20000, 5000, 500)
    res = _fpr(metric_type, n_sims, n_pg)
    m1, m2 = st.columns(2)
    m1.metric("Фактический FPR", f"{res.fpr:.4f}", delta=f"{res.fpr - res.alpha:+.4f} к α")
    m2.metric("95% ДИ (Уилсон)", f"[{res.ci[0]:.3f}; {res.ci[1]:.3f}]")
    st.plotly_chart(simulate.fig_false_positive_rate(res, tmpl), use_container_width=True)

with tab2:
    st.markdown(
        "Эмпирическая мощность (из симуляций) должна совпадать с аналитической "
        "формулой - это подтверждает корректность калькулятора мощности."
    )
    c1, c2 = st.columns(2)
    n_pg = c1.slider("Размер группы ", 100, 3000, 500, 100)
    n_sims = c2.select_slider("Число симуляций ", [500, 1000, 2000, 5000], value=2000)
    res = _power(n_pg, n_sims)
    st.plotly_chart(simulate.fig_power(res, tmpl), use_container_width=True)

with tab3:
    st.markdown(
        "CUPED использует пред-период для снижения дисперсии. Ожидаемое снижение "
        "~= ρ^2, что напрямую повышает мощность без роста выборки."
    )
    c1, c2 = st.columns(2)
    corr = c1.slider("Корреляция с пред-периодом ρ", 0.0, 0.95, 0.7, 0.05)
    n_sims = c2.select_slider("Число симуляций  ", [500, 1000, 2000, 5000], value=2000)
    res = _cuped(corr, 800, n_sims)
    m1, m2, m3 = st.columns(3)
    m1.metric("Мощность без CUPED", f"{res.power_without:.3f}")
    m2.metric(
        "Мощность с CUPED",
        f"{res.power_with:.3f}",
        delta=f"{res.power_with - res.power_without:+.3f}",
    )
    m3.metric(
        "Снижение дисперсии",
        f"{res.var_reduction_observed:.1%}",
        delta=f"теория ρ^2={res.var_reduction_theoretical:.1%}",
        delta_color="off",
    )
    st.plotly_chart(simulate.fig_cuped(res, tmpl), use_container_width=True)

with tab4:
    st.markdown(
        "«Подглядывание» - многократная проверка значимости по мере набора данных. "
        "Наивный подход раздувает ошибку I рода в разы; mSPRT и "
        "alpha-spending удерживают её на уровне α."
    )
    c1, c2, c3 = st.columns(3)
    n_pg = c1.slider("Размер группы   ", 500, 4000, 2000, 500)
    n_looks = c2.slider("Число проверок", 5, 40, 20, 5)
    n_sims = c3.select_slider("Число симуляций   ", [1000, 2000, 3000, 5000], value=3000)
    res = _peeking(n_pg, n_looks, n_sims)
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Наивный FPR", f"{res.cumulative_fpr_naive[-1]:.3f}", delta="раздут!", delta_color="inverse"
    )
    m2.metric("mSPRT FPR", f"{res.cumulative_fpr_msprt[-1]:.3f}")
    m3.metric("Alpha-spending FPR", f"{res.cumulative_fpr_spending[-1]:.3f}")
    st.plotly_chart(simulate.fig_peeking(res, tmpl), use_container_width=True)
