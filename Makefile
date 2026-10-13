# Makefile для платформы A/B-экспериментов.
# Основные цели: setup, test, lint, format, sim, dashboard, all.

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup:  ## Установить пакет со всеми зависимостями (editable)
	$(PIP) install -e ".[dev,dashboard]"

.PHONY: lint
lint:  ## Проверить код линтером ruff
	ruff check src tests dashboard

.PHONY: format
format:  ## Отформатировать код (black + ruff --fix)
	ruff check --fix src tests dashboard
	black src tests dashboard

.PHONY: test
test:  ## Запустить все тесты
	pytest -q

.PHONY: sim
sim:  ## Прогнать Монте-Карло симуляции и сохранить графики в docs/figures
	$(PYTHON) -m abtest.simulate --outdir docs/figures

.PHONY: data
data:  ## Сгенерировать пример синтетических данных в data/
	$(PYTHON) -m abtest.datagen --outdir data

.PHONY: dashboard
dashboard:  ## Запустить Streamlit-дашборд
	streamlit run dashboard/app.py

.PHONY: all
all: setup lint test sim  ## Полный прогон: установка -> линт -> тесты -> симуляции
	@echo "OK: установка, линт, тесты и симуляции завершены."

.PHONY: clean
clean:  ## Удалить кэши и артефакты
	rm -rf .pytest_cache .ruff_cache **/__pycache__ src/*.egg-info build dist
