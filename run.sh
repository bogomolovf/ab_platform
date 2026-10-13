#!/usr/bin/env bash
# Универсальный запускатор платформы A/B-экспериментов.
# Использование:
#   ./run.sh setup      - создать venv и установить зависимости
#   ./run.sh test       - запустить тесты
#   ./run.sh lint       - проверить код (ruff + black --check)
#   ./run.sh sim        - прогнать симуляции и сохранить графики
#   ./run.sh dashboard  - поднять Streamlit-дашборд
#   ./run.sh all        - setup + lint + test + sim
set -euo pipefail

VENV_DIR=".venv"
PYTHON_BIN="${PYTHON:-python3}"

ensure_venv() {
  if [ ! -d "$VENV_DIR" ]; then
    echo ">> Создаю виртуальное окружение в $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
}

cmd_setup() {
  ensure_venv
  python -m pip install --upgrade pip
  python -m pip install -e ".[dev,dashboard]"
  echo ">> Установка завершена."
}

cmd_test() { ensure_venv; pytest -q; }

cmd_lint() {
  ensure_venv
  ruff check src tests dashboard
  black --check src tests dashboard
}

cmd_sim() {
  ensure_venv
  python -m abtest.simulate --outdir docs/figures
  echo ">> Графики симуляций сохранены в docs/figures"
}

cmd_dashboard() { ensure_venv; streamlit run dashboard/app.py; }

cmd_all() { cmd_setup; cmd_lint; cmd_test; cmd_sim; echo ">> Готово."; }

main() {
  local target="${1:-all}"
  case "$target" in
    setup) cmd_setup ;;
    test) cmd_test ;;
    lint) cmd_lint ;;
    sim) cmd_sim ;;
    dashboard) cmd_dashboard ;;
    all) cmd_all ;;
    *) echo "Неизвестная команда: $target"; echo "Доступно: setup|test|lint|sim|dashboard|all"; exit 1 ;;
  esac
}

main "$@"
