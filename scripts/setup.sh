#!/usr/bin/env bash
# Установка окружения для скилла mastering-skill (Linux, macOS, Windows через Git Bash).
#   bash setup.sh                           — мастеринг (pedalboard, pyloudnorm, matchering, …), видеокарта не нужна
#   bash setup.sh --with-separator          — плюс разделение на стемы (torch + audio-separator, 1–3 ГБ);
#                                             ускорение выбирается само: NVIDIA → CUDA, Apple Silicon → MPS/CoreML, иначе CPU
#   bash setup.sh --with-separator --cpu    — принудительно без видеокарты
#   bash setup.sh --with-separator --gpu    — принудительно CUDA (NVIDIA)
# Индекс для torch с CUDA можно сменить: TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu126
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SKILL_DIR="$(dirname "$HERE")"
source "$HERE/_common.sh"

WITH_SEP=0 ACCEL=auto
for a in "$@"; do
  case "$a" in
    --with-separator) WITH_SEP=1 ;;
    --cpu) ACCEL=cpu ;;
    --gpu) ACCEL=cuda ;;
    *) echo "Неизвестный флаг: $a" >&2; exit 2 ;;
  esac
done

install_hint() {
  case $OS in
    macos)   echo "  brew install ffmpeg uv" ;;
    windows) echo "  winget install Gyan.FFmpeg astral-sh.uv   (потом перезапусти терминал)" ;;
    *)
      if   command -v pacman  >/dev/null; then echo "  sudo pacman -S --needed ffmpeg uv"
      elif command -v apt-get >/dev/null; then echo "  sudo apt install ffmpeg python3-venv && curl -LsSf https://astral.sh/uv/install.sh | sh"
      elif command -v dnf     >/dev/null; then echo "  sudo dnf install ffmpeg uv   (ffmpeg — из RPM Fusion)"
      elif command -v zypper  >/dev/null; then echo "  sudo zypper install ffmpeg uv"
      else echo "  установи ffmpeg и uv (https://docs.astral.sh/uv/) пакетным менеджером системы"
      fi ;;
  esac
}

command -v ffmpeg >/dev/null || { echo "Не хватает ffmpeg. Установи:"; install_hint; exit 1; }

if command -v uv >/dev/null; then
  # uv сам скачает Python 3.12 — самую совместимую версию для numba/matchering/torch
  uv venv --allow-existing --python 3.12 "$VENV"
  PIP=(uv pip install --python "$VENV_PY")
else
  PY=""
  for c in python3 python; do
    "$c" -c 'import sys; assert sys.version_info >= (3, 10)' 2>/dev/null && { PY=$c; break; }
  done
  [[ -n $PY ]] || { echo "Нужен Python 3.10–3.13 или uv. Установи:"; install_hint; exit 1; }
  if ! "$PY" -c 'import sys; assert sys.version_info < (3, 14)'; then
    echo "⚠ Python новее 3.13 — часть аудио-библиотек может не собраться."
    echo "  Лучше поставить uv и перезапустить setup.sh — он сам возьмёт Python 3.12:"; install_hint
  fi
  "$PY" -m venv "$VENV"
  "$VENV_PY" -m pip install -q -U pip
  PIP=("$VENV_PY" -m pip install)
fi

"${PIP[@]}" pedalboard pyloudnorm matchering soundfile numpy scipy matplotlib

if ((WITH_SEP)); then
  if [[ $ACCEL == auto ]]; then
    if [[ $OS == macos ]]; then ACCEL=mac
    elif command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then ACCEL=cuda
    else ACCEL=cpu
    fi
  fi
  case $ACCEL in
    mac)
      echo "Ставлю разделение на стемы (macOS; на Apple Silicon — ускорение MPS/CoreML)…"
      "${PIP[@]}" torch torchaudio
      "${PIP[@]}" "audio-separator[cpu]" ;;
    cuda)
      echo "Ставлю разделение на стемы с ускорением на NVIDIA (CUDA)…"
      "${PIP[@]}" torch torchaudio --index-url "${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
      "${PIP[@]}" "audio-separator[gpu]" ;;
    *)
      echo "Ставлю разделение на стемы (без видеокарты, CPU)…"
      "${PIP[@]}" torch torchaudio --index-url https://download.pytorch.org/whl/cpu
      "${PIP[@]}" "audio-separator[cpu]" ;;
  esac
fi

chmod +x "$SKILL_DIR/scripts/"*.sh "$SKILL_DIR/scripts/master.py" 2>/dev/null || true
"$VENV_PY" -c "import pedalboard, pyloudnorm, matchering, soundfile, scipy; print('✓ Окружение готово:', r'$VENV')"
if ((WITH_SEP)); then
  "$VENV_PY" -c "import audio_separator" 2>/dev/null && echo "✓ audio-separator установлен ($ACCEL)" || echo "⚠ audio-separator не установился"
fi
echo "Запуск: bash $SKILL_DIR/scripts/run.sh analyze трек.wav"
