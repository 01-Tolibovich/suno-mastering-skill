#!/usr/bin/env bash
# Запускает master.py в окружении скилла.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$HERE/_common.sh"
if [[ ! -x "$VENV_PY" ]]; then
  echo "Окружение не установлено. Запусти: bash $HERE/setup.sh" >&2
  exit 1
fi
exec "$VENV_PY" "$HERE/master.py" "$@"
