"""Страница «Планирование»: размер выборки, мощность и MDE."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from _shared import page_header, plotly_template
from abtest import power

page_header(
    "Планирование эксперимента",
    "Размер выборки, мощность и минимально детектируемый эффект (MDE)",
)

metric_type = st.radio(
    "Тип метрики",
    ["Конверсия (доля)", "Непрерывная (среднее)"],
    horizontal=True,
)

col_cfg, col_out = st.columns([1, 1.3])

with col_cfg:
    st.subheader("Параметры")
    alpha = st.slider("α (уровень значимости)", 0.01, 0.10, 0.05, 0.01)
    target_power = st.slider("Мощность (1-β)", 0.5, 0.99, 0.80, 0.01)
    ratio = st.slider("Соотношение групп (тест/контроль)", 0.2, 5.0, 1.0, 0.1)
    alternative = st.selectbox("Альтернатива", ["two-sided", "larger", "smaller"], index=0)

    if metric_type == "Конверсия (доля)":
        p_control = st.number_input(
            "Базовая конверсия p0", 0.001, 0.999, 0.10, 0.005, format="%.3f"
        )
        effect_mode = st.radio(
            "Задать эффект как", ["Относительный %", "Абсолютный"], horizontal=True
        )
        if effect_mode == "Относительный %":
            rel = st.number_input("Ожидаемый lift, %", 0.1, 200.0, 10.0, 0.5)
            p_treat = p_control * (1 + rel / 100)
        else:
            p_treat = st.number_input(
                "Ожидаемая конверсия теста", 0.001, 0.999, 0.11, 0.005, format="%.3f"
            )
        p_treat = float(np.clip(p_treat, 1e-4, 1 - 1e-4))
        st.caption(f"Конверсия теста: **{p_treat:.4f}** (абс. эффект {p_treat - p_control:+.4f})")
    else:
        sd = st.number_input("Стандартное отклонение σ", 0.0001, 1e9, 20.0, 0.5, format="%.4f")
        effect = st.number_input(
            "Детектируемый эффект (абс. разница средних)", 0.0001, 1e9, 2.0, 0.1, format="%.4f"
        )

with col_out:
    st.subheader("Результат")
    try:
        if metric_type == "Конверсия (доля)":
            res = power.sample_size_proportion(
                p_control,
                p_treat,
                alpha=alpha,
                power=target_power,
                alternative=alternative,
                ratio=ratio,
            )
        else:
            res = power.sample_size_mean(
                effect,
                sd,
                alpha=alpha,
                power=target_power,
                alternative=alternative,
                ratio=ratio,
            )
        m1, m2, m3 = st.columns(3)
        m1.metric("Контроль, n", f"{res.n_control:,}")
        m2.metric("Тест, n", f"{res.n_treatment:,}")
        m3.metric("Всего", f"{res.n_total:,}")
        st.metric("Достигаемая мощность", f"{res.power:.3f}")
        with st.expander("Детали расчёта"):
            st.json({k: round(v, 6) if isinstance(v, float) else v for k, v in res.detail.items()})
    except ValueError as exc:
        st.error(f"Не удалось рассчитать: {exc}")
        res = None

st.divider()

# --- Кривые мощности ------------------------------------------------------- #
st.subheader("Кривые мощности")
tab_n, tab_e = st.tabs(["Мощность vs размер выборки", "MDE vs размер выборки"])
tmpl = plotly_template()

with tab_n:
    if metric_type == "Конверсия (доля)":
        curve = power.power_curve_proportion(
            p_control, p_treat, alpha=alpha, alternative=alternative, ratio=ratio
        )
    else:
        curve = power.power_curve_mean(
            effect, sd, alpha=alpha, alternative=alternative, ratio=ratio
        )
    fig = go.Figure()
    fig.add_scatter(
        x=curve.x, y=curve.power, mode="lines", line=dict(color="#3b82f6", width=3), name="мощность"
    )
    fig.add_hline(
        y=target_power,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text=f"цель {target_power:.0%}",
    )
    if res is not None:
        fig.add_vline(
            x=res.n_control,
            line_dash="dot",
            line_color="#10b981",
            annotation_text=f"n={res.n_control}",
        )
    fig.update_layout(
        template=tmpl,
        xaxis_title=curve.xlabel,
        yaxis_title="мощность",
        xaxis_type="log",
        yaxis_range=[0, 1.02],
        height=420,
        margin=dict(l=50, r=30, t=30, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_e:
    # MDE как функция размера выборки на группу.
    n_grid = np.unique(np.round(np.logspace(2, 5.5, 40)).astype(int))
    mdes = []
    for n in n_grid:
        try:
            if metric_type == "Конверсия (доля)":
                mdes.append(
                    power.mde_proportion(
                        p_control,
                        int(n),
                        alpha=alpha,
                        power=target_power,
                        alternative=alternative,
                        relative=True,
                    )
                    * 100
                )
            else:
                mdes.append(
                    power.mde_mean(
                        sd, int(n), alpha=alpha, power=target_power, alternative=alternative
                    )
                )
        except ValueError:
            mdes.append(np.nan)
    ylab = "MDE, относит. %" if metric_type == "Конверсия (доля)" else "MDE (абс.)"
    fig2 = go.Figure()
    fig2.add_scatter(
        x=n_grid, y=mdes, mode="lines", line=dict(color="#8b5cf6", width=3), name="MDE"
    )
    fig2.update_layout(
        template=tmpl,
        xaxis_title="Размер контроля, n",
        yaxis_title=ylab,
        xaxis_type="log",
        height=420,
        margin=dict(l=50, r=30, t=30, b=50),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Чем больше выборка, тем меньший эффект можно надёжно обнаружить.")
