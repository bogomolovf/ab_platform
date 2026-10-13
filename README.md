<h1 align="center">A/B Experimentation Platform</h1>

<p align="center">
  Библиотека для A/B-тестов
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue">
  <img src="https://img.shields.io/badge/tests-63%20passing-brightgreen">
  <img src="https://img.shields.io/badge/lint-ruff%20%7C%20black-black">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

---

В этом проекте я написал библиотеку, которая поддерживает все основные методы, которые используются в A/B тестировании. Каждый метод проверяется симуляциями Монте-Карло и сверяется с `statsmodels` в тестах.

## Функционал

- **Планирование** - размер выборки, мощность и MDE для конверсий и непрерывных метрик, кривые мощности.
- **Проверка дизайна** - SRM и баланс групп по ковариатам.
- **Анализ** - z-тест долей, тест Уэлча, бутстрап (BCa), Манна-Уитни, доверительные интервалы, абсолютный и относительный lift.
- **CUPED** - снижение дисперсии по данным до эксперимента.
- **Sequential testing** - mSPRT и alpha-spending: можно смотреть на тест хоть каждый день и не раздувать ошибку.
- **Множественные метрики** - поправки Bonferroni и Benjamini-Hochberg'а.
- **Байес** - вероятность, что B лучше A, и ожидаемые потери.
- **Вердикт** - «раскатывать / не раскатывать / нужно больше данных» с предупреждениями о рисках.

## Стек

Python 3.11, `numpy` / `scipy` / `statsmodels` / `pandas` (ядро), `streamlit` + `plotly` (дашборд), `pytest` (тесты), `ruff` + `black` + `pre-commit` (качество), GitHub Actions (CI). Без платных сервисов.

## Запуск

```bash
make all         # установка + линт + тесты + графики симуляций
make dashboard   # открыть дашборд на http://localhost:8501
```

Нет `make` - то же самое делает `./run.sh all`. Пример из кода:

```python
from abtest import datagen, tests, cuped, report

exp = datagen.generate_experiment(metric_type="continuous", true_effect=2.0, seed=42)
c, t = exp.group_arrays("metric")
print(report.interpret(tests.welch_ttest(c, t)).text)   # готовый вердикт
```

## Структура

```
src/abtest/     ядро: power, srm, tests, cuped, sequential, multiple, bayes, datagen, simulate, report
dashboard/      Streamlit: Планирование / Анализ / Симуляции
tests/          pytest - проверка корректности статистики
notebooks/      methodology.ipynb - методология с примерами
docs/           графики симуляций и скриншоты
```

Подробный разбор каждого метода с формулами и источниками - в [METHODOLOGY.md](METHODOLOGY.md).

## Для чего был сделан проект
Прежде всего проект создавалася, чтобы разобраться, как работают основные методы A/B тестирования изнутри. Во время работы над проектом я вручную написал расчёт мощности и MDE, тесты для долей и средних, бутстрап, CUPED, проверку SRM, поправки на множественность, последовательный анализ (mSPRT, alpha-spending) и байес. Готовыми библиотеками пользовался только как фундаментом (numpy/scipy), корректность каждого метода проверял симуляциями и сверкой со statsmodels. Данный проект дал мне глубокое понимания того, как что работает и для чего это нужно, теперь при дизайне очередного экспериента я понимаю, что мне нужно в той или иной ситуации, и могу интерпретировать это понятным бизнесу языком.
