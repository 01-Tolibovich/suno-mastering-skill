#!/usr/bin/env bash
# Запускает master.py в окружении скилла.
VENV="${MASTER_VENV:-$HOME/.local/share/master-skill/venv}"
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Окружение не установлено. Запусти: bash $HERE/setup.sh" >&2
  exit 1
fi
exec "$VENV/bin/python" "$HERE/master.py" "$@"
