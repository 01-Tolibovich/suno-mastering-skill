# mastering-skill — скилл Claude Code для мастеринга Suno-треков

Работает на **Linux, macOS и Windows**, на любом железе. Мастерингу видеокарта не нужна. Для разделения трека на стемы (`--split`) видеокарта NVIDIA (CUDA) или чип Apple Silicon (M1 и новее) дают ускорение. `setup.sh` находит их сам, а если их нет, ставит версию для процессора.

## Установка

### 1. Системные программы: ffmpeg и uv

`uv` сам скачает Python 3.12, на котором аудио-библиотеки собираются надёжнее всего. Системный Python трогать не нужно.

| Система | Команда |
|---|---|
| Arch / Manjaro | `sudo pacman -S --needed ffmpeg uv` |
| Ubuntu / Debian / Mint | `sudo apt install ffmpeg` и `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Fedora | `sudo dnf install ffmpeg uv` (ffmpeg — из RPM Fusion) |
| openSUSE | `sudo zypper install ffmpeg uv` |
| macOS | `brew install ffmpeg uv` |
| Windows | `winget install Gyan.FFmpeg astral-sh.uv`, потом перезапусти терминал |

Без `uv` тоже получится: подойдёт Python 3.10–3.13. На Ubuntu/Debian для этого нужен ещё пакет `python3-venv`.

### 2. Скилл

Скилл можно положить в личные скиллы Claude Code (будет доступен во всех проектах):

```bash
git clone https://github.com/01-Tolibovich/suno-mastering-skill.git ~/.claude/skills/mastering-skill
```

Или только в один проект: склонируй в `<проект>/.claude/skills/mastering-skill`.

### 3. Окружение

```bash
bash ~/.claude/skills/mastering-skill/scripts/setup.sh                    # мастеринг, ~300 МБ
bash ~/.claude/skills/mastering-skill/scripts/setup.sh --with-separator   # + разделение на стемы, 1–3 ГБ
```

`--with-separator` сам выбирает ускорение: NVIDIA → CUDA, Mac → MPS/CoreML, иначе CPU. Выбор можно задать вручную флагами `--cpu` или `--gpu`.

Куда ставится окружение: `~/.local/share/master-skill/venv` на Linux и macOS, `%LOCALAPPDATA%\master-skill\venv` на Windows. Другую папку можно задать переменной `MASTER_VENV`.

На Windows все команды выполняются в **Git Bash**. Claude Code на Windows и так работает через него.

Если PyPI недоступен, задай зеркало: `UV_DEFAULT_INDEX=<url>` для uv или `PIP_INDEX_URL=<url>` для pip.

## Использование в Claude Code

```
/mastering-skill ~/Music/suno/song.wav
/mastering-skill ~/Music/suno/song_stems/
/mastering-skill ~/Music/suno/song.wav --ref ~/Music/refs/reference.wav
```

Или просто словами: «сделай мастеринг ~/Music/suno/song.wav для Telegram».
Claude сделает анализ и мастеринг, проверит отчёт и спектр, откроет A/B-страницу. Дальше по вашим комментариям («режет верха», «мало баса») он сделает следующую версию.

## Без Claude (напрямую)

```bash
R="bash $HOME/.claude/skills/mastering-skill/scripts/run.sh"
$R analyze song.wav                      # замеры (JSON)
$R run song.wav                          # мастеринг → song_master/v01/
$R run stems/ --preset strong            # стемы, сильная чистка
$R run song.wav --ref ref.wav --ref-mix 0.6
$R run song.wav --targets -14,-10 --air -1 --low 1
$R run song.wav --split vocals           # отделить вокал (нужен setup.sh --with-separator)
$R run song.wav --save-preset my-sound   # сохранить настройки
$R run next.wav --use my-sound           # применить к следующему треку
$R open song_master/v01/ab.html          # открыть A/B-сравнение
$R run --help                            # все флаги
```

## Что на выходе (`<трек>_master/vNN/`)

| Файл | Зачем |
|---|---|
| `master_-14LUFS.wav` | стриминги (Spotify, YouTube, Apple) |
| `master_-11LUFS.wav`, `.m4a` | Telegram, соцсети |
| `premaster.wav` | до лимитера — если захотите доделать в DAW |
| `ab.html` | A/B оригинал ↔ мастер на одинаковой громкости |
| `spectrum.png` | спектр до/после/референс |
| `report.md`, `report.json` | замеры и все настройки |

## Что скачивать из Suno

- Скачивайте **WAV**, не MP3.
- Стемы Suno (WAV) кладите в одну папку **без** полного микса и давайте эту папку.
- Нет стемов — давайте полный WAV. При желании `--split vocals` отделит вокал: без видеокарты это 5–30 минут на трек, с видеокартой обычно 1–3 минуты.

## Ограничения

- Claude не слышит, он только меряет. Решение «нравится / нет» за вами, через `ab.html`.
- Чистка ослабляет «металлический» блеск и шипящие, но AI-происхождение трека не скрывает.
- Модель разделения по умолчанию (`UVR-MDX-NET-Inst_HQ_3`) выбрана как самая быстрая из качественных. Лучший вокал даёт `--sep-model model_bs_roformer_ep_317_sdr_12.9755.ckpt`, но без видеокарты она может работать полчаса и больше.
