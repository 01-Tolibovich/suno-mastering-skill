# /master — скилл Claude Code для мастеринга Suno-треков (Arch Linux, без видеокарты)

## Установка

```bash
# 1. Системные пакеты
sudo pacman -S --needed ffmpeg python libsndfile xdg-utils uv

# 2. Скилл в личные скиллы Claude Code
mkdir -p ~/.claude/skills
unzip master-skill.zip -d ~/.claude/skills/     # получится ~/.claude/skills/master/

# 3. Окружение (~300 МБ в ~/.local/share/master-skill)
bash ~/.claude/skills/master/scripts/setup.sh
#    с разделением на стемы (torch CPU + audio-separator, ~1 ГБ):
# bash ~/.claude/skills/master/scripts/setup.sh --with-separator
```

`uv` нужен, чтобы окружение собралось на Python 3.12: у Arch системный Python самый свежий, и аудио-библиотеки (numba для Matchering) могут под него ещё не собираться.

## Использование в Claude Code

```
/master ~/Music/suno/song.wav
/master ~/Music/suno/song_stems/
/master ~/Music/suno/song.wav --ref ~/Music/refs/reference.wav
```

Или просто словами: «сделай мастеринг ~/Music/suno/song.wav для Telegram».
Claude сделает анализ, мастеринг, проверит отчёт и спектр, откроет A/B-страницу и по вашим комментариям («режет верха», «мало баса») сделает следующую версию.

## Без Claude (напрямую)

```bash
R=~/.claude/skills/master/scripts/run.sh
$R analyze song.wav                      # замеры (JSON)
$R run song.wav                          # мастеринг → song_master/v01/
$R run stems/ --preset strong            # стемы, сильная чистка
$R run song.wav --ref ref.wav --ref-mix 0.6
$R run song.wav --targets -14,-10 --air -1 --low 1
$R run song.wav --split vocals           # отделить вокал (нужен --with-separator, на CPU долго)
$R run song.wav --save-preset my-sound   # сохранить настройки
$R run next.wav --use my-sound           # применить к следующему треку
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

## Что делать из Suno

- Скачивайте **WAV**, не MP3.
- Стемы Suno (WAV) → кладите в одну папку **без** полного микса и давайте папку.
- Нет стемов — давайте полный WAV; при желании `--split vocals` отделит вокал (на CPU 5–30 мин на трек).

## Ограничения

- Claude не слышит: он меряет. Решение «нравится / нет» — за вами, через `ab.html`.
- Чистка ослабляет «металлический» блеск и шипящие, но AI-происхождение трека не скрывает.
- Разделение на стемы на CPU медленное; модель по умолчанию (`UVR-MDX-NET-Inst_HQ_3`) выбрана как самая быстрая из качественных. Лучший вокал даёт `--sep-model model_bs_roformer_ep_317_sdr_12.9755.ckpt`, но на CPU это может занять полчаса и больше.
