"""Страница «Анализ эксперимента»: данные, тесты, ДИ, вердикт."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _shared import BUNDLED_DATASETS, load_bundled, page_header, plotly_template
from abtest import bayes, cuped, multiple, report, srm, tests

page_header(
    "Анализ эксперимента",
    "Загрузите данные или возьмите встроенный датасет - платформа посчитает "
    "метрики, доверительные интервалы и выдаст вердикт",
)

# --------------------------------------------------------------------------- #
# 1. Источник данных
# --------------------------------------------------------------------------- #
src = st.radio("Источник данных", ["Встроенный датасет", "Загрузить CSV"], horizontal=True)
if src == "Встроенный датасет":
    name = st.selectbox("Датасет", list(BUNDLED_DATASETS))
    df = load_bundled(name)
else:
    up = st.file_uploader("CSV-файл с колонкой группы и метриками", type=["csv"])
    if up is None:
        st.stop()
    df = pd.read_csv(up)

with st.expander("Предпросмотр данных", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"{len(df):,} строк, {df.shape[1]} колонок")

# --------------------------------------------------------------------------- #
# 2. Настройка колонок
# --------------------------------------------------------------------------- #
cols = list(df.columns)
NO_PRE = "- нет -"


def _default(colname: str, options: list[str]) -> int:
    return options.index(colname) if colname in options else 0


c1, c2 = st.columns(2)
with c1:
    group_col = st.selectbox("Колонка группы", cols, index=_default("group", cols))
    groups = list(pd.unique(df[group_col]))
    control_label = st.selectbox(
        "Метка контроля", groups, index=_default("control", [str(g) for g in groups])
    )
    treatment_label = st.selectbox("Метка теста", groups, index=1 if len(groups) > 1 else 0)
with c2:
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and c != group_col]
    metric_col = st.selectbox(
        "Основная метрика", numeric_cols, index=_default("metric", numeric_cols)
    )
    pre_col = st.selectbox(
        "Пред-период для CUPED (опц.)",
        [NO_PRE, *numeric_cols],
        index=_default("pre_metric", [NO_PRE, *numeric_cols]),
    )
    practical = st.number_input(
        "Порог практической значимости (абс., опц.)", 0.0, 1e9, 0.0, format="%.4f"
    )

series = df[metric_col].dropna()
is_binary = set(np.unique(series)).issubset({0, 1}) and series.nunique() <= 2
st.caption(
    f"Метрика **{metric_col}** распознана как "
    f"{'конверсионная (0/1)' if is_binary else 'непрерывная'}."
)

sub = df[df[group_col].isin([control_label, treatment_label])]
c_mask = sub[group_col] == control_label
t_mask = sub[group_col] == treatment_label

if not st.button("Проанализировать", type="primary", use_container_width=True):
    st.stop()

tmpl = plotly_template()

# --------------------------------------------------------------------------- #
# 3. Валидация дизайна: SRM
# --------------------------------------------------------------------------- #
st.header("1. Валидация дизайна")
counts = {control_label: int(c_mask.sum()), treatment_label: int(t_mask.sum())}
srm_res = srm.check_srm(counts)
sc1, sc2, sc3 = st.columns(3)
sc1.metric(f"n({control_label})", f"{counts[control_label]:,}")
sc2.metric(f"n({treatment_label})", f"{counts[treatment_label]:,}")
sc3.metric("SRM p-value", f"{srm_res.p_value:.3g}")
if srm_res.srm_detected:
    st.error(f"{srm_res.summary} - сплитование сломано, результатам нельзя доверять!")
else:
    st.success(srm_res.summary)

# Баланс ковариат
candidate_covs = [
    c
    for c in cols
    if c not in (group_col, metric_col) and c != (None if pre_col == NO_PRE else pre_col)
]
covs = st.multiselect(
    "Ковариаты для проверки баланса",
    candidate_covs,
    default=[c for c in ("is_mobile", "tenure_days") if c in candidate_covs],
)
if covs:
    bal = srm.covariate_balance(df, group_col, covs, control_label, treatment_label)
    st.dataframe(
        bal.style.format({"effect_size": "{:.3f}", "statistic": "{:.3f}", "p_value": "{:.3g}"}),
        use_container_width=True,
    )
    if not bal["balanced"].all():
        st.warning("Часть ковариат несбалансирована - возможен перекос рандомизации.")

# --------------------------------------------------------------------------- #
# 4. Анализ основной метрики
# --------------------------------------------------------------------------- #
st.header("2. Анализ основной метрики")
result = None
if is_binary:
    sc = int(sub.loc[c_mask, metric_col].sum())
    stt = int(sub.loc[t_mask, metric_col].sum())
    nc, nt = int(c_mask.sum()), int(t_mask.sum())
    result = tests.proportion_ztest(sc, nc, stt, nt)
    a1, a2, a3, a4 = st.columns(4)
    a1.metric(f"Конверсия {control_label}", f"{result.p_control:.4f}")
    a2.metric(
        f"Конверсия {treatment_label}",
        f"{result.p_treatment:.4f}",
        delta=f"{result.relative_effect:+.2%}",
    )
    a3.metric("Абс. эффект", f"{result.absolute_effect:+.4f}")
    a4.metric("p-value", f"{result.p_value:.4g}")
    st.write(
        f"95% ДИ абс. эффекта: **[{result.absolute_ci[0]:+.4f}; {result.absolute_ci[1]:+.4f}]**, "
        f"относит. lift: **{result.relative_effect:+.2%}** "
        f"(ДИ [{result.relative_ci[0]:+.2%}; {result.relative_ci[1]:+.2%}])"
    )

    # Байес
    st.subheader("Байесовский взгляд (beta-binomial)")
    b = bayes.beta_binomial(sc, nc, stt, nt, seed=0)
    bb1, bb2, bb3 = st.columns(3)
    bb1.metric("P(B > A)", f"{b.prob_b_better:.1%}")
    bb2.metric("Ожид. потери (выбрать B)", f"{b.expected_loss_choose_b:.2e}")
    bb3.metric("Ожид. потери (выбрать A)", f"{b.expected_loss_choose_a:.2e}")
    st.caption(b.recommendation)
else:
    c_vals = sub.loc[c_mask, metric_col].to_numpy(dtype=float)
    t_vals = sub.loc[t_mask, metric_col].to_numpy(dtype=float)
    result = tests.welch_ttest(c_vals, t_vals)
    a1, a2, a3, a4 = st.columns(4)
    a1.metric(f"Среднее {control_label}", f"{result.mean_control:.4f}")
    a2.metric(
        f"Среднее {treatment_label}",
        f"{result.mean_treatment:.4f}",
        delta=f"{result.relative_effect:+.2%}",
    )
    a3.metric("Абс. эффект", f"{result.absolute_effect:+.4f}")
    a4.metric("p-value (Welch)", f"{result.p_value:.4g}")
    st.write(
        f"95% ДИ: **[{result.absolute_ci[0]:+.4f}; {result.absolute_ci[1]:+.4f}]**, "
        f"t={result.t_stat:.3f}, df={result.df:.0f}"
    )

    with st.spinner("Бутстрап и непараметрика..."):
        boot = tests.bootstrap_mean_diff(c_vals, t_vals, n_boot=5000, method="bca", seed=0)
        mw = tests.mann_whitney(c_vals, t_vals)
    n1, n2 = st.columns(2)
    n1.info(f"**Бутстрап (BCa)** ДИ: [{boot.ci[0]:+.4f}; {boot.ci[1]:+.4f}]")
    n2.info(f"**Манна-Уитни**: p={mw.p_value:.4g}, rank-biserial={mw.rank_biserial:+.3f}")

    # Распределения
    fig = go.Figure()
    fig.add_histogram(
        x=c_vals,
        name=str(control_label),
        opacity=0.6,
        histnorm="probability density",
        marker_color="#9ca3af",
    )
    fig.add_histogram(
        x=t_vals,
        name=str(treatment_label),
        opacity=0.6,
        histnorm="probability density",
        marker_color="#3b82f6",
    )
    fig.update_layout(
        template=tmpl,
        barmode="overlay",
        height=360,
        xaxis_title=metric_col,
        yaxis_title="плотность",
        margin=dict(l=50, r=30, t=30, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# 5. CUPED (для непрерывной метрики с пред-периодом)
# --------------------------------------------------------------------------- #
if pre_col != NO_PRE and not is_binary:
    st.header("3. CUPED - снижение дисперсии")
    cu = cuped.cuped_ttest(
        sub.loc[c_mask, metric_col].to_numpy(float),
        sub.loc[c_mask, pre_col].to_numpy(float),
        sub.loc[t_mask, metric_col].to_numpy(float),
        sub.loc[t_mask, pre_col].to_numpy(float),
    )
    q1, q2, q3 = st.columns(3)
    q1.metric("Корреляция ρ с пред-периодом", f"{cu.correlation:+.3f}")
    q2.metric("Снижение дисперсии", f"{cu.variance_reduction:.1%}")
    q3.metric(
        "p-value: до и после",
        f"{cu.test_unadjusted.p_value:.3g} -> {cu.test_adjusted.p_value:.3g}",
    )
    before = cu.test_unadjusted.absolute_ci
    after = cu.test_adjusted.absolute_ci
    fig = go.Figure()
    fig.add_scatter(
        x=[before[0], before[1]],
        y=["до CUPED", "до CUPED"],
        mode="lines+markers",
        line=dict(color="#9ca3af", width=8),
        name="до",
    )
    fig.add_scatter(
        x=[after[0], after[1]],
        y=["с CUPED", "с CUPED"],
        mode="lines+markers",
        line=dict(color="#3b82f6", width=8),
        name="после",
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#ef4444")
    fig.update_layout(
        template=tmpl,
        height=260,
        xaxis_title="ДИ абсолютного эффекта",
        showlegend=False,
        margin=dict(l=50, r=30, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Более узкий интервал после CUPED = выше точность и мощность при том же n.")

# --------------------------------------------------------------------------- #
# 6. Множественные метрики (поправки)
# --------------------------------------------------------------------------- #
secondary = st.multiselect(
    "Вторичные метрики для одновременной проверки (поправка на множественность)",
    [c for c in numeric_cols if c != metric_col],
)
adjusted_primary = None
if secondary:
    st.header("4. Множественные сравнения")
    rows = []
    pvals = []
    all_metrics = [metric_col, *secondary]
    for mcol in all_metrics:
        s = df[mcol].dropna()
        binary = set(np.unique(s)).issubset({0, 1}) and s.nunique() <= 2
        if binary:
            sc = int(sub.loc[c_mask, mcol].sum())
            nc = int(c_mask.sum())
            stt = int(sub.loc[t_mask, mcol].sum())
            nt = int(t_mask.sum())
            r = tests.proportion_ztest(sc, nc, stt, nt)
        else:
            r = tests.welch_ttest(
                sub.loc[c_mask, mcol].to_numpy(float), sub.loc[t_mask, mcol].to_numpy(float)
            )
        rows.append({"метрика": mcol, "эффект": r.absolute_effect, "p_value": r.p_value})
        pvals.append(r.p_value)
    method = st.radio("Метод поправки", ["benjamini-hochberg", "bonferroni"], horizontal=True)
    adj = multiple.adjust_pvalues(np.array(pvals), method=method)
    table = pd.DataFrame(rows)
    table["p_adj"] = adj.adjusted
    table["значима"] = adj.reject
    st.dataframe(
        table.style.format({"эффект": "{:+.4f}", "p_value": "{:.4g}", "p_adj": "{:.4g}"}),
        use_container_width=True,
    )
    adjusted_primary = float(adj.adjusted[0])

# --------------------------------------------------------------------------- #
# 7. Вердикт
# --------------------------------------------------------------------------- #
st.header("5. Вердикт")
verdict = report.interpret(
    result,
    practical_threshold=practical if practical > 0 else None,
    srm_result=srm_res,
    adjusted_pvalue=adjusted_primary,
    n_comparisons=1 + len(secondary),
)
if srm_res.srm_detected:
    st.error(verdict.headline)
elif verdict.statistically_significant and "хуже" not in verdict.headline:
    st.success(verdict.headline)
else:
    st.warning(verdict.headline)
st.markdown(f"**Рекомендация:** {verdict.recommendation}")
if verdict.warnings:
    for w in verdict.warnings:
        st.markdown(f"- {w}")
with st.expander("Полный текст вердикта"):
    st.code(verdict.text, language="text")
