#!/usr/bin/env bash
# Установка окружения для скилла /master (Arch Linux, без видеокарты).
#   bash setup.sh                    — мастеринг (pedalboard, pyloudnorm, matchering, …)
#   bash setup.sh --with-separator   — плюс разделение на стемы (torch CPU + audio-separator, ~1 ГБ)
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${MASTER_VENV:-$HOME/.local/share/master-skill/venv}"
WITH_SEP=0
[[ "${1:-}" == "--with-separator" ]] && WITH_SEP=1

missing=()
for c in ffmpeg python3; do command -v "$c" >/dev/null || missing+=("$c"); done
if ((${#missing[@]})); then
  echo "Не хватает: ${missing[*]}"
  echo "Установи:  sudo pacman -S --needed ffmpeg python libsndfile xdg-utils uv"
  exit 1
fi
command -v xdg-open >/dev/null || echo "Совет: sudo pacman -S xdg-utils — чтобы A/B-страница открывалась из Claude Code."

if command -v uv >/dev/null; then
  # Python 3.12 — самая совместимая версия для numba/matchering/torch
  uv venv --allow-existing --python 3.12 "$VENV"
  PIP=(uv pip install --python "$VENV/bin/python")
else
  PYV=$(python3 -c 'import sys; print(sys.version_info[0]*100 + sys.version_info[1])')
  if ((PYV >= 314)); then
    echo "⚠ Системный Python новее 3.13 — часть аудио-библиотек может не собраться."
    echo "  Лучше поставить uv (sudo pacman -S uv) и перезапустить setup.sh — он сам возьмёт Python 3.12."
  fi
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q -U pip
  PIP=("$VENV/bin/pip" install)
fi

"${PIP[@]}" pedalboard pyloudnorm matchering soundfile numpy scipy matplotlib

if ((WITH_SEP)); then
  echo "Ставлю разделение на стемы (CPU-версия torch, без CUDA)…"
  "${PIP[@]}" torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  "${PIP[@]}" "audio-separator[cpu]"
fi

chmod +x "$SKILL_DIR/scripts/"*.sh "$SKILL_DIR/scripts/master.py"
"$VENV/bin/python" -c "import pedalboard, pyloudnorm, matchering, soundfile, scipy; print('✓ Окружение готово:', '$VENV')"
((WITH_SEP)) && { "$VENV/bin/audio-separator" --version >/dev/null 2>&1 && echo "✓ audio-separator установлен" || echo "⚠ audio-separator не запустился"; }
echo "Запуск: $SKILL_DIR/scripts/run.sh analyze трек.wav"
