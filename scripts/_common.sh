# Общее для setup.sh и run.sh: ОС, папка окружения, путь к Python внутри venv.
# Linux, macOS и Windows (Git Bash — в нём Claude Code выполняет команды на Windows).

case "$(uname -s)" in
  Linux*)               OS=linux ;;
  Darwin*)              OS=macos ;;
  MINGW*|MSYS*|CYGWIN*) OS=windows ;;
  *)                    OS=other ;;
esac

if [[ -n "${MASTER_VENV:-}" ]]; then
  VENV="$MASTER_VENV"
elif [[ $OS == windows && -n "${LOCALAPPDATA:-}" ]]; then
  VENV="$(cygpath -u "$LOCALAPPDATA" 2>/dev/null || echo "$LOCALAPPDATA")/master-skill/venv"
else
  VENV="$HOME/.local/share/master-skill/venv"
fi

if [[ $OS == windows ]]; then
  VENV_PY="$VENV/Scripts/python.exe"
else
  VENV_PY="$VENV/bin/python"
fi
