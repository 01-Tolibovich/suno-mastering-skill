#!/usr/bin/env python3
"""
master.py — чистка и мастеринг AI-треков (Suno/Udio). Работает на любом CPU; видеокарта нужна только для ускорения --split.

Команды:
  analyze INPUT [--ref REF]      — замеры: LUFS, True Peak, LRA, баланс частот, признаки MP3/артефактов
  run     INPUT [опции]          — чистка → (сведение стемов) → мастеринг → отчёт и A/B-страница
  open    FILE                   — открыть файл (ab.html, spectrum.png) в браузере/просмотрщике системы

INPUT — либо полный трек (WAV/FLAC/AIFF), либо папка со стемами.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import webbrowser
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import maximum_filter1d, uniform_filter1d

import pyloudnorm as pyln
import pedalboard as pb

SKILL_DIR = Path(__file__).resolve().parent.parent
PRESET_DIR = SKILL_DIR / "presets"
AUDIO_EXT = {".wav", ".flac", ".aif", ".aiff", ".mp3", ".ogg", ".m4a", ".opus"}
LOSSY_EXT = {".mp3", ".ogg", ".m4a", ".opus"}

# Значения по умолчанию для `run`. Пресет-файл и флаги CLI перекрывают их.
DEFAULTS = {
    "preset": "normal",       # сила чистки: light | normal | strong
    "targets": [-14.0, -11.0],  # LUFS: стриминги и Telegram
    "ceiling": -1.0,          # True Peak, dBTP
    "release": 150.0,         # релиз лимитера, мс
    "glue": 1.5,              # средняя компрессия шины на громких местах, dB (0 = выкл)
    "sat": 0.12,              # доля мягкой сатурации 0..1
    "width": 1.0,             # ширина стерео (side), 1.0 = как есть
    "mono_below": 120.0,      # моно ниже этой частоты, Гц (0 = выкл)
    "low": 0.0,               # полка 100 Гц, dB
    "mud": None,              # колокол 300 Гц, dB (None = по пресету)
    "presence": 0.0,          # колокол 3 кГц, dB
    "air": 0.0,               # полка 10 кГц, dB
    "deharsh": None,          # макс. подавление резкости 5.5–10 кГц, dB (None = по пресету)
    "deess": None,            # макс. де-эссинг вокала, dB (None = по пресету)
    "vocal_gain": 0.0,        # громкость вокальных стемов, dB
    "ref_mix": 1.0,           # доля подгонки под референс 0..1
    "split": "none",          # none | vocals | 4
    "sep_model": None,
    "exclude": None,
    "m4a": True,
}
STRENGTH = {"light": 0.6, "normal": 1.0, "strong": 1.6}

BANDS = [
    ("sub", 20, 60), ("bass", 60, 150), ("low_mid", 150, 500), ("mid", 500, 2000),
    ("high_mid", 2000, 5000), ("presence", 5000, 10000), ("air", 10000, 16000), ("ultra", 16000, 20000),
]


# ───────────────────────────── утилиты ─────────────────────────────

def log(msg: str) -> None:
    print(msg, flush=True)


def db2lin(db):
    return 10.0 ** (np.asarray(db) / 20.0)


def lin2db(x, floor=1e-12):
    return 20.0 * np.log10(np.maximum(np.abs(x), floor))


def run_cmd(cmd: list[str], check=True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def load(path: Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Возвращает (2, n) float64 и частоту дискретизации."""
    try:
        x, fs = sf.read(str(path), dtype="float64", always_2d=True)
    except Exception:
        with tempfile.TemporaryDirectory() as td:
            tmp = os.path.join(td, "decoded.wav")
            run_cmd(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-c:a", "pcm_f32le", tmp])
            x, fs = sf.read(tmp, dtype="float64", always_2d=True)
    x = x.T
    if x.shape[0] == 1:
        x = np.vstack([x, x])
    elif x.shape[0] > 2:
        x = x[:2]
    if sr and fs != sr:
        x = resample(x, fs, sr)
        fs = sr
    return np.ascontiguousarray(x), int(fs)


def resample(x: np.ndarray, fs: int, sr: int) -> np.ndarray:
    g = math.gcd(int(fs), int(sr))
    return signal.resample_poly(x, sr // g, fs // g, axis=1)


def save(path: Path, x: np.ndarray, sr: int, subtype="PCM_24") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0).T if subtype.startswith("PCM") else x.T, sr, subtype=subtype)


def lufs(x: np.ndarray, sr: int) -> float:
    try:
        v = pyln.Meter(sr).integrated_loudness(x.T)
    except ValueError:
        return -70.0
    return float(v) if np.isfinite(v) else -70.0


_TP_FIR: dict[int, np.ndarray] = {}


def _tp_fir(os_: int) -> np.ndarray:
    # длинный интерполирующий фильтр с полосой почти до Найквиста: короткий фильтр по умолчанию
    # занижает межсэмпловые пики на материале с яркими верхами (тарелки, шум Suno)
    if os_ not in _TP_FIR:
        _TP_FIR[os_] = signal.firwin(os_ * 32 + 1, 0.99 / os_, window=("kaiser", 9.0))
    return _TP_FIR[os_]


def os_peak_env(x: np.ndarray, os_: int = 8, chunk: int = 1 << 18, pad: int = 128) -> np.ndarray:
    """Пиковая огибающая с учётом межсэмпловых пиков (8x передискретизация), по сэмплам."""
    n = x.shape[1]
    out = np.empty(n)
    h = _tp_fir(os_)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        a, b = max(0, s - pad), min(n, e + pad)
        seg = np.abs(signal.resample_poly(x[:, a:b], os_, 1, axis=1, window=h)).max(axis=0)
        seg = seg[(s - a) * os_:(e - a) * os_].reshape(-1, os_).max(axis=1)
        out[s:e] = np.maximum(seg, np.abs(x[:, s:e]).max(axis=0))
    return out


def true_peak_db(x: np.ndarray) -> float:
    return float(lin2db(os_peak_env(x).max()))


def ffmpeg_ebur128(path: Path) -> dict:
    """Независимый замер ffmpeg: I (LUFS), LRA (LU), True Peak (dBTP)."""
    try:
        p = run_cmd(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
                     "-filter_complex", "ebur128=peak=true", "-f", "null", "-"], check=False)
    except FileNotFoundError:
        return {}
    txt = p.stderr
    tail = txt[txt.rfind("Summary:"):] if "Summary:" in txt else txt
    res = {}
    for key, rx in (("lufs", r"I:\s+(-?[\d.]+|-inf) LUFS"), ("lra", r"LRA:\s+(-?[\d.]+) LU"),
                    ("true_peak", r"Peak:\s+(-?[\d.]+|-inf) dBFS")):
        m = re.search(rx, tail)
        if m:
            res[key] = float(m.group(1)) if m.group(1) != "-inf" else -70.0
    return res


def rms_frames_db(x: np.ndarray, sr: int, win_s=0.05) -> np.ndarray:
    hop = max(1, int(win_s * sr))
    m = (x ** 2).mean(axis=0)
    k = len(m) // hop
    if k == 0:
        return np.array([-120.0])
    return 10 * np.log10(m[:k * hop].reshape(k, hop).mean(axis=1) + 1e-12)


# ───────────────────────────── обработка ─────────────────────────────

def pb_chain(x: np.ndarray, sr: int, plugins: list) -> np.ndarray:
    if not plugins:
        return x
    y = pb.Pedalboard(plugins)(x.astype(np.float32), sr)
    return y.astype(np.float64)


def dyn_band(x: np.ndarray, sr: int, lo: float, hi: float, max_db: float,
             offset_db: float = 3.0, ratio: float = 3.0, pct: float = 75.0) -> tuple[np.ndarray, dict]:
    """
    Динамическое подавление полосы (де-эссер / подавление «металлического блеска»).
    Полоса выделяется фазолинейно (filtfilt), поэтому band + rest == x бит-в-бит.
    Подавляются только моменты, когда доля энергии полосы заметно выше типичной для трека.
    """
    hi = min(hi, 0.45 * sr)
    if max_db <= 0 or lo >= hi:
        return x, {"band": [lo, hi], "max_db": 0, "mean_db": 0.0, "active_pct": 0.0}
    sos = signal.butter(2, [lo, hi], btype="bandpass", fs=sr, output="sos")
    band = signal.sosfiltfilt(sos, x, axis=1)
    rest = x - band
    w = max(3, int(0.005 * sr)) | 1
    eb = uniform_filter1d((band ** 2).mean(axis=0), w)
    ef = uniform_filter1d((x ** 2).mean(axis=0), w)
    rel = 10 * np.log10((eb + 1e-14) / (ef + 1e-14))
    active = ef > ef.max() * 1e-4
    if active.sum() < sr * 0.1:
        return x, {"band": [lo, hi], "max_db": max_db, "mean_db": 0.0, "active_pct": 0.0}
    thr = np.percentile(rel[active], pct) + offset_db
    red = np.clip((rel - thr) * (1 - 1 / ratio), 0, max_db)
    red = maximum_filter1d(red, max(1, int(0.010 * sr)))      # удержание
    red = uniform_filter1d(red, max(1, int(0.004 * sr)))      # сглаживание
    y = rest + band * db2lin(-red)
    red_a = red[active]
    return y, {"band": [lo, round(hi)], "max_db": round(max_db, 2),
               "mean_db": round(float(red_a.mean()), 2),
               "active_pct": round(float((red_a > 0.5).mean() * 100), 1)}


def auto_compress(x: np.ndarray, sr: int, target_gr: float, ratio: float,
                  attack: float, release: float) -> tuple[np.ndarray, dict]:
    """Компрессор с порогом, подобранным под нужную среднюю компрессию на громких местах."""
    if target_gr <= 0:
        return x, {"gr_db": 0.0}
    rin = rms_frames_db(x, sr, 0.4)
    loud = rin >= np.percentile(rin, 70)
    thr = float(np.percentile(rin, 90))
    y, gr = x, 0.0
    for _ in range(5):
        y = pb_chain(x, sr, [pb.Compressor(threshold_db=thr, ratio=ratio, attack_ms=attack, release_ms=release)])
        rout = rms_frames_db(y, sr, 0.4)
        n = min(len(rin), len(rout))
        gr = float(np.mean((rin[:n] - rout[:n])[loud[:n]]))
        if abs(gr - target_gr) < 0.25:
            break
        thr += (gr - target_gr) * 1.5
        thr = float(np.clip(thr, -60, 0))
    return y, {"gr_db": round(gr, 2), "threshold_db": round(thr, 1), "ratio": ratio}


def ms_stage(x: np.ndarray, sr: int, mono_below: float, width: float) -> np.ndarray:
    mid = (x[0] + x[1]) * 0.5
    side = (x[0] - x[1]) * 0.5
    if mono_below and mono_below > 0:
        sos = signal.butter(2, mono_below, btype="highpass", fs=sr, output="sos")
        side = signal.sosfiltfilt(sos, side)
    side = side * width
    return np.vstack([mid + side, mid - side])


def saturate(x: np.ndarray, amount: float, drive: float = 1.6) -> np.ndarray:
    if amount <= 0:
        return x
    wet = np.tanh(drive * x) / drive
    return x + amount * (wet - x)


def match_loudness(y: np.ndarray, ref_lufs: float, sr: int) -> np.ndarray:
    cur = lufs(y, sr)
    if cur <= -69:
        return y
    return y * db2lin(ref_lufs - cur)


def limiter(x: np.ndarray, sr: int, ceiling_db: float, release_ms: float,
            lookahead_ms: float = 5.0, block: int = 16) -> tuple[np.ndarray, dict]:
    """Брикволл-лимитер с упреждением по межсэмпловым пикам (8x)."""
    ceil = float(db2lin(ceiling_db))
    pk = os_peak_env(x)
    n = len(pk)
    nb = -(-n // block)
    pkb = np.pad(pk, (0, nb * block - n)).reshape(nb, block).max(axis=1)
    g = np.minimum(1.0, ceil / np.maximum(pkb, 1e-12))
    L = max(1, int(round(lookahead_ms * 1e-3 * sr / block)))
    W = L + 3
    h = np.lib.stride_tricks.sliding_window_view(np.pad(g, (0, W - 1), constant_values=1.0), W).min(axis=1)
    a = 1.0 - math.exp(-block / (release_ms * 1e-3 * sr))
    r = np.empty_like(h)
    prev = 1.0
    for i, t in enumerate(h):          # ~3 c на 3-минутный трек — приемлемо
        prev = t if t < prev else prev + (t - prev) * a
        r[i] = prev
    c = np.cumsum(np.concatenate([np.full(L + 1, r[0]), r]))
    s = (c[L + 1:] - c[:-(L + 1)]) / (L + 1)
    gains = np.interp(np.arange(n), np.arange(nb) * block, s)
    gr = np.maximum(0.0, -lin2db(s))
    return x * gains, {"gr_max_db": round(float(gr.max()), 2) + 0.0,
                       "gr_mean_db": round(float(gr[pkb > ceil * 0.1].mean()) if (pkb > ceil * 0.1).any() else 0.0, 2) + 0.0}


def finalize(x: np.ndarray, sr: int, target: float, ceiling: float, release: float) -> tuple[np.ndarray, dict]:
    """Подбор усиления так, чтобы после лимитера получить target LUFS и TP <= ceiling."""
    gain_db = target - lufs(x, sr)
    y, st, cur, tp = x, {}, -70.0, 0.0
    for _ in range(10):
        gain_db = min(gain_db, 30.0)
        y, st = limiter(x * db2lin(gain_db), sr, ceiling, release)
        tp = true_peak_db(y)
        if tp > ceiling + 0.02:
            y = y * db2lin(ceiling - tp - 0.02)
            tp = true_peak_db(y)
        cur = lufs(y, sr)
        if abs(cur - target) < 0.1:
            break
        gain_db += target - cur
    return y, {"lufs": round(cur, 2), "true_peak": round(tp, 2), "gain_db": round(gain_db, 2), **st}


# ───────────────────────────── стемы ─────────────────────────────

def classify(name: str) -> str:
    # audio-separator пишет «трек_(Vocals)_модель.wav» — смотрим только на метку в скобках,
    # иначе имя модели (например vocals_mel_band_roformer) спутает классификацию
    m = re.search(r"\(([^)]+)\)", name)
    n = (m.group(1) if m else name).lower()
    if n.startswith("instrumental") or n in ("no_vocals", "no vocals"):
        return "other"
    if re.search(r"vocal|vox|voice|lead|choir|acapella|вокал", n):
        return "vocals"
    if re.search(r"drum|perc|kick|snare|hat|cymbal|beat|ударн", n):
        return "drums"
    if re.search(r"bass|бас|808|sub", n):
        return "bass"
    return "other"


def stem_chain(kind: str, x: np.ndarray, sr: int, k: float, cfg: dict) -> tuple[np.ndarray, dict]:
    st: dict = {"kind": kind}
    l0 = lufs(x, sr)
    if kind == "vocals":
        x = pb_chain(x, sr, [pb.HighpassFilter(80)])
        deess = cfg["deess"] if cfg["deess"] is not None else 6.0 * k
        x, st["deess"] = dyn_band(x, sr, 5000, 9500, deess, offset_db=3.0, ratio=4.0)
        x, st["harsh"] = dyn_band(x, sr, 2500, 4500, 2.0 * k, offset_db=4.0, pct=80)
        x, st["air_artifacts"] = dyn_band(x, sr, 10000, 16000, 3.0 * k)
        x, st["comp"] = auto_compress(x, sr, 2.5, 2.5, 10, 120)
    elif kind == "drums":
        x = pb_chain(x, sr, [pb.HighpassFilter(30)])
        x, st["shimmer"] = dyn_band(x, sr, 7000, 12000, 4.0 * k)
        x, st["air_artifacts"] = dyn_band(x, sr, 12000, 17000, 2.0 * k)
        x, st["comp"] = auto_compress(x, sr, 1.5, 2.0, 25, 120)
    elif kind == "bass":
        x = pb_chain(x, sr, [pb.HighpassFilter(30), pb.LowpassFilter(8000)])
        x = ms_stage(x, sr, 200, 1.0)
        x, st["comp"] = auto_compress(x, sr, 2.5, 3.0, 20, 150)
    else:
        mud = cfg["mud"] if cfg["mud"] is not None else -1.0 * k
        x = pb_chain(x, sr, [pb.HighpassFilter(35), pb.PeakFilter(300, mud, 1.0)])
        x, st["shimmer"] = dyn_band(x, sr, 6000, 11000, 3.0 * k)
        x, st["air_artifacts"] = dyn_band(x, sr, 11000, 16000, 2.0 * k)
    if l0 > -69:
        x = match_loudness(x, l0, sr)      # сохраняем баланс, который дал Suno
    return x, st


def fullmix_cleanup(x: np.ndarray, sr: int, k: float, cfg: dict) -> tuple[np.ndarray, dict]:
    st: dict = {}
    mud = cfg["mud"] if cfg["mud"] is not None else -1.0 * k
    x = pb_chain(x, sr, [pb.HighpassFilter(25), pb.PeakFilter(300, mud, 0.9)])
    deharsh = cfg["deharsh"] if cfg["deharsh"] is not None else 3.5 * k
    x, st["deharsh"] = dyn_band(x, sr, 5500, 10000, deharsh, offset_db=3.0, ratio=3.0)
    x, st["air_artifacts"] = dyn_band(x, sr, 11000, 16000, 2.5 * k)
    return x, st


def list_stems(folder: Path, exclude: str | None) -> list[Path]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXT and p.is_file())
    if exclude:
        files = [p for p in files if not re.search(exclude, p.name, re.I)]
    return files


def load_stems(files: list[Path]) -> tuple[list[tuple[str, np.ndarray]], int]:
    sr = None
    stems = []
    for p in files:
        x, fs = load(p, sr)
        sr = sr or fs
        stems.append((p.name, x))
    n = max(s[1].shape[1] for s in stems)
    stems = [(name, np.pad(x, ((0, 0), (0, n - x.shape[1])))) for name, x in stems]
    return stems, sr


def separator_exe() -> str:
    for name in ("audio-separator", "audio-separator.exe"):
        exe = Path(sys.executable).parent / name
        if exe.exists():
            return str(exe)
    w = shutil.which("audio-separator")
    if not w:
        raise SystemExit("audio-separator не установлен. Запусти: scripts/setup.sh --with-separator")
    return w


def separate(path: Path, mode: str, outdir: Path, model: str | None) -> list[Path]:
    """Разделение через audio-separator (CUDA / Apple Silicon / CPU). Возвращает список стемов."""
    exe = separator_exe()
    default = {"vocals": "UVR-MDX-NET-Inst_HQ_3.onnx", "4": "htdemucs.yaml"}[mode]
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [exe, str(path), "--model_filename", model or default,
           "--output_dir", str(outdir), "--output_format", "WAV"]
    log(f"• Разделение на стемы ({model or default}; без видеокарты — может занять несколько минут)…")
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("Ошибка разделения:\n" + p.stderr[-2000:])
    files = sorted(outdir.glob("*.wav"))
    log(f"  готово за {time.time() - t:.0f} с: " + ", ".join(f.name for f in files))
    return files


# ───────────────────────────── анализ ─────────────────────────────

def spectrum(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    mono = x.mean(axis=0)
    nper = 8192 if sr <= 48000 else 16384
    f, p = signal.welch(mono, sr, nperseg=min(nper, len(mono)), scaling="spectrum")
    return f, p


def analyze_signal(x: np.ndarray, sr: int, path: Path | None = None) -> dict:
    info: dict = {"sample_rate": sr, "duration_s": round(x.shape[1] / sr, 1)}
    if path is not None:
        try:
            si = sf.info(str(path))
            info["format"] = f"{si.format}/{si.subtype}"
        except Exception:
            info["format"] = path.suffix.lstrip(".")
        info["lossy_source"] = path.suffix.lower() in LOSSY_EXT
    info["lufs"] = round(lufs(x, sr), 2)
    info["true_peak"] = round(true_peak_db(x), 2)
    info["sample_peak"] = round(float(lin2db(np.abs(x).max())), 2)
    info["clipped_samples"] = int((np.abs(x) >= 0.9999).sum())
    info["dc_offset"] = round(float(np.abs(x.mean(axis=1)).max()), 5)
    rf = rms_frames_db(x, sr, 0.05)
    info["plr_db"] = round(info["true_peak"] - info["lufs"], 1)
    # стерео
    c = np.corrcoef(x[0], x[1])[0, 1] if np.std(x[0]) > 0 and np.std(x[1]) > 0 else 1.0
    info["stereo_correlation"] = round(float(c), 3)
    sos = signal.butter(4, 150, btype="lowpass", fs=sr, output="sos")
    lo = signal.sosfilt(sos, x, axis=1)
    cl = np.corrcoef(lo[0], lo[1])[0, 1] if np.std(lo[0]) > 0 and np.std(lo[1]) > 0 else 1.0
    info["low_end_correlation"] = round(float(cl), 3)
    # баланс частот
    f, p = spectrum(x, sr)
    total = p[(f >= 20) & (f <= min(20000, sr / 2))].sum() + 1e-20
    info["bands_db"] = {name: round(float(10 * np.log10(p[(f >= a) & (f < min(b, sr / 2))].sum() / total + 1e-20)), 1)
                        for name, a, b in BANDS if a < sr / 2}
    # срез верхов (признак MP3/кодека)
    pdb = 10 * np.log10(p + 1e-20)
    ref_lvl = np.median(pdb[(f >= 1000) & (f <= 5000)])
    above = np.where((pdb > ref_lvl - 45) & (f > 8000))[0]
    info["hf_cutoff_hz"] = int(f[above.max()]) if len(above) else None
    # всплески резкости 5.5–10 кГц
    sosb = signal.butter(2, [5500, min(10000, 0.45 * sr)], btype="bandpass", fs=sr, output="sos")
    band = signal.sosfilt(sosb, x.mean(axis=0))
    w = int(0.01 * sr)
    eb = uniform_filter1d(band ** 2, w)
    ef = uniform_filter1d(x.mean(axis=0) ** 2, w)
    act = ef > ef.max() * 1e-4
    rel = 10 * np.log10((eb + 1e-14) / (ef + 1e-14))
    info["harsh_bursts_pct"] = round(float((rel[act] > np.percentile(rel[act], 50) + 6).mean() * 100), 1) if act.any() else 0.0
    info["loud_frames_spread_db"] = round(float(np.percentile(rf, 95) - np.percentile(rf[rf > rf.max() - 40], 10)), 1)
    return info


def warnings_for(info: dict) -> list[str]:
    w = []
    if info.get("lossy_source"):
        w.append("сжатый формат (MP3/OGG/M4A). Скачай из Suno WAV — обработка вытянет артефакты сжатия.")
    cut = info.get("hf_cutoff_hz")
    if cut and cut < 17000 and info["sample_rate"] >= 44100:
        w.append(f"Верхи обрываются около {cut} Гц — похоже на MP3-источник или сильную фильтрацию.")
    if info.get("clipped_samples", 0) > 100:
        w.append(f"В исходнике {info['clipped_samples']} клипованных сэмплов — искажения уже «вшиты».")
    if info.get("true_peak", -99) > 0:
        w.append("True Peak выше 0 dBTP — при кодировании для стримингов будут искажения.")
    if info.get("low_end_correlation", 1) < 0.5:
        w.append("Низ ниже 150 Гц сильно разъезжается по стерео — в моно бас может пропасть (поможет --mono-below).")
    if info.get("dc_offset", 0) > 0.005:
        w.append("Есть постоянная составляющая (DC offset) — её уберёт фильтр низких частот в цепочке.")
    return w


# ───────────────────────────── отчёт ─────────────────────────────

def plot_spectra(curves: list[tuple[str, np.ndarray, int]], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    styles = ["-", "-", "--"]
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=130)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for i, (label, x, sr) in enumerate(curves[:3]):
        f, p = spectrum(x, sr)
        m = (f >= 20) & (f <= min(20000, sr / 2))
        f, pdb = f[m], 10 * np.log10(p[m] + 1e-20)
        # сглаживание 1/6 октавы
        lf = np.log2(f)
        sm = np.array([pdb[np.abs(lf - v) <= 1 / 12].mean() for v in lf])
        ax.plot(f, sm, styles[i], color=colors[i], lw=2, label=label)
    ax.set_xscale("log")
    ax.set_xlim(20, 20000)
    ticks = [30, 60, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000, 16000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t // 1000}k" if t >= 1000 else str(t) for t in ticks])
    ax.grid(True, which="major", color="#e1e0d9", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#55544e", labelsize=8)
    ax.set_xlabel("Частота, Гц", color="#55544e", fontsize=9)
    ax.set_ylabel("Уровень, dB (одинаковая громкость)", color="#55544e", fontsize=9)
    ax.set_title("Спектр: до и после (выровнено по LUFS)", loc="left", fontsize=11, color="#1f1f1d")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)


AB_HTML = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>A/B — {title}</title>
<style>
:root{{--bg:#fcfcfb;--fg:#1f1f1d;--mut:#6b6a63;--line:#e1e0d9;--acc:#2a78d6;--card:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--fg:#eceae4;--mut:#9c9a92;--line:#2c2c2a;--acc:#3987e5;--card:#222220}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;padding:24px 16px}}
main{{max-width:760px;margin:0 auto}}h1{{font-size:20px;margin:0 0 4px}}p.m{{color:var(--mut);margin:0 0 20px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}}
.row{{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}}
button{{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:transparent;color:var(--fg);cursor:pointer}}
button.on{{background:var(--acc);border-color:var(--acc);color:#fff}}
audio{{width:100%}}table{{width:100%;border-collapse:collapse;font-size:14px}}
td,th{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}}th{{color:var(--mut);font-weight:500}}
kbd{{border:1px solid var(--line);border-radius:4px;padding:0 5px;font-size:12px}}
</style></head><body><main>
<h1>A/B: {title}</h1>
<p class="m">Версия {version}. Кнопки A/B переключают файл на той же секунде. <kbd>Пробел</kbd> — пауза, <kbd>Tab</kbd> — A↔B.</p>
<div class="card"><b>Честное сравнение</b> — оригинал и мастер на одинаковой громкости ({cmp_lufs} LUFS).
<div class="row" id="ab"></div><audio id="pl" controls preload="auto"></audio></div>
<div class="card"><b>Финальные файлы</b> (полная громкость)
<div class="row" id="fin"></div></div>
<div class="card"><b>Замеры</b><table>{table}</table></div>
<div class="card"><b>Спектр</b><br><img src="spectrum.png" alt="Спектр до и после" style="width:100%;height:auto"></div>
</main><script>
const ab={ab_json}, fin={fin_json};
const pl=document.getElementById('pl');let cur=null;
function mk(box,list){{list.forEach(([l,src])=>{{const b=document.createElement('button');b.textContent=l;b.onclick=()=>play(src,b);box.appendChild(b);}});}}
function play(src,btn){{const t=pl.currentTime,was=!pl.paused||cur===null;document.querySelectorAll('button').forEach(b=>b.classList.remove('on'));btn.classList.add('on');
 if(cur!==src){{pl.src=src;cur=src;pl.addEventListener('loadedmetadata',()=>{{pl.currentTime=Math.min(t,pl.duration||t);if(was)pl.play();}},{{once:true}});}}}}
mk(document.getElementById('ab'),ab);mk(document.getElementById('fin'),fin);
document.addEventListener('keydown',e=>{{if(e.code==='Space'){{e.preventDefault();pl.paused?pl.play():pl.pause();}}
 if(e.code==='Tab'){{e.preventDefault();const bs=[...document.querySelectorAll('#ab button')];const i=bs.findIndex(b=>b.classList.contains('on'));bs[(i+1)%bs.length].click();}}}});
</script></body></html>"""


def fmt(v, unit=""):
    return "—" if v is None else f"{v}{unit}"


# ───────────────────────────── пресеты ─────────────────────────────

def load_preset(name: str) -> dict:
    p = Path(name)
    if not p.exists():
        p = PRESET_DIR / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"Пресет не найден: {name} (ищу в {PRESET_DIR})")
    return json.loads(p.read_text(encoding="utf-8"))


def save_preset(name: str, cfg: dict) -> Path:
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    keep = {k: cfg[k] for k in DEFAULTS if k not in ("split", "sep_model", "exclude")}
    p = PRESET_DIR / f"{name}.json"
    p.write_text(json.dumps(keep, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# ───────────────────────────── команды ─────────────────────────────

def cmd_analyze(args) -> None:
    src = Path(args.input).expanduser()
    if src.is_dir():
        stems, sr = load_stems(list_stems(src, args.exclude))
        x = sum(s[1] for s in stems)
        res = {"input": str(src), "mode": "stems",
               "stems": {name: {"group": classify(name), "lufs": round(lufs(s, sr), 1)} for name, s in stems},
               "sum": analyze_signal(x, sr)}
        res["warnings"] = warnings_for(res["sum"])
    else:
        x, sr = load(src)
        res = {"input": str(src), "mode": "full_mix", **analyze_signal(x, sr, src)}
        res["ffmpeg_check"] = ffmpeg_ebur128(src)
        res["warnings"] = warnings_for(res)
    if args.ref:
        r, rsr = load(Path(args.ref).expanduser())
        res["reference"] = analyze_signal(r, rsr, Path(args.ref))
        a = res["sum"]["bands_db"] if res["mode"] == "stems" else res["bands_db"]
        res["vs_reference_db"] = {k: round(a[k] - res["reference"]["bands_db"].get(k, a[k]), 1) for k in a}
    print(json.dumps(res, ensure_ascii=False, indent=2))


def cmd_run(args) -> None:
    t0 = time.time()
    cfg = dict(DEFAULTS)
    if args.use:
        cfg.update(load_preset(args.use))
    for k in DEFAULTS:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    if args.no_m4a:
        cfg["m4a"] = False
    k = STRENGTH[cfg["preset"]]

    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"Не найден: {src}")
    if cfg["split"] != "none" and src.is_file():
        separator_exe()
    base = src.stem if src.is_file() else src.name
    root = Path(args.out).expanduser() if args.out else src.parent / f"{base}_master"
    root.mkdir(parents=True, exist_ok=True)
    nums = [int(m.group(1)) for d in root.glob("v*") if (m := re.fullmatch(r"v(\d+)", d.name))]
    out = root / f"v{(max(nums) + 1) if nums else 1:02d}"
    out.mkdir()
    log(f"▶ {src.name} → {out}")

    report: dict = {"input": str(src), "version": out.name, "settings": cfg, "warnings": []}

    # 1. Источник: стемы или полный микс
    stems = None
    if src.is_dir():
        files = list_stems(src, cfg["exclude"])
        if not files:
            raise SystemExit("В папке нет аудиофайлов.")
        stems, sr = load_stems(files)
        original = sum(s[1] for s in stems)
        report["mode"] = "stems"
        info_src = None
    else:
        original, sr = load(src)
        info_src = src
        report["mode"] = "full_mix"
        if cfg["split"] != "none":
            sep_files = separate(src, cfg["split"], out / "stems_separated", cfg["sep_model"])
            stems, _ = load_stems(sep_files)
            n = original.shape[1]
            stems = [(nm, np.pad(s, ((0, 0), (0, max(0, n - s.shape[1]))))[:, :n]) for nm, s in stems]
            resid = original - sum(s[1] for s in stems)
            report["separation_residual_db"] = round(lufs(resid, sr) - lufs(original, sr), 1)
            report["mode"] = f"full_mix+split_{cfg['split']}"

    info_before = analyze_signal(original, sr, info_src)
    report["before"] = info_before
    report["warnings"] += ["Исходник: " + w for w in warnings_for(info_before)]
    for w in report["warnings"]:
        log("⚠ " + w)

    # 2. Чистка
    if stems:
        log(f"• Чистка стемов ({cfg['preset']}):")
        processed, report["stems"] = [], {}
        for name, s in stems:
            kind = classify(name)
            y, st = stem_chain(kind, s, sr, k, cfg)
            if kind == "vocals" and cfg["vocal_gain"]:
                y = y * db2lin(cfg["vocal_gain"])
            processed.append(y)
            report["stems"][name] = st
            log(f"  {name:<40} → {kind}")
        mix = sum(processed)
        mix = pb_chain(mix, sr, [pb.HighpassFilter(25)])
    else:
        log(f"• Чистка полного микса ({cfg['preset']})")
        mix, report["cleanup"] = fullmix_cleanup(original, sr, k, cfg)

    # 3. Подгонка под референс (Matchering)
    if args.ref and cfg["ref_mix"] > 0:
        log("• Подгонка под референс (Matchering)…")
        try:
            import matchering as mg
            with tempfile.TemporaryDirectory() as td:
                tin, tout = os.path.join(td, "in.wav"), os.path.join(td, "out.wav")
                save(Path(tin), mix / max(1e-9, np.abs(mix).max()) * 0.7, sr, "FLOAT")
                mg.log(lambda *_: None)
                mg.process(target=tin, reference=str(Path(args.ref).expanduser()),
                           results=[mg.Result(tout, "FLOAT", use_limiter=False, normalize=False)])
                matched, msr = load(Path(tout), sr)
            n = mix.shape[1]
            matched = np.pad(matched, ((0, 0), (0, max(0, n - matched.shape[1]))))[:, :n]
            # выравниваем по времени (у Matchering бывает сдвиг на пару сэмплов) — иначе смешивание даст гребёнку
            seg = slice(n // 4, n // 4 + min(n // 2, 10 * sr))
            a_, b_ = mix.mean(axis=0)[seg], matched.mean(axis=0)[seg]
            cc = signal.correlate(b_, a_, mode="full", method="fft")
            mid = len(a_) - 1
            win = cc[mid - 2000:mid + 2001]
            lag = int(np.argmax(np.abs(win))) - 2000
            if lag:
                matched = np.roll(matched, -lag, axis=1)
                if lag > 0:
                    matched[:, -lag:] = 0
                else:
                    matched[:, :-lag] = 0
            base_l = lufs(mix, sr)
            matched = match_loudness(matched, base_l, sr)
            mix = (1 - cfg["ref_mix"]) * mix + cfg["ref_mix"] * matched
            report["reference"] = {"file": str(args.ref), "mix": cfg["ref_mix"], "aligned_by_samples": lag}
        except Exception as e:  # noqa: BLE001
            msg = f"Matchering не сработал ({e}); продолжаю без референса."
            log("⚠ " + msg)
            report["warnings"].append(msg)

    # 4. Мастер-шина
    log("• Мастер-шина: тон, компрессия, сатурация, стерео")
    tone = []
    if cfg["low"]:
        tone.append(pb.LowShelfFilter(100, cfg["low"], 0.7))
    if cfg["presence"]:
        tone.append(pb.PeakFilter(3000, cfg["presence"], 0.8))
    if cfg["air"]:
        tone.append(pb.HighShelfFilter(10000, cfg["air"], 0.7))
    if stems and cfg["mud"] is not None:
        tone.append(pb.PeakFilter(300, cfg["mud"], 0.9))
    mix = pb_chain(mix, sr, tone)
    mix, report["glue"] = auto_compress(mix, sr, cfg["glue"], 1.8, 30, 200)
    mix = mix / max(1e-9, np.abs(mix).max()) * db2lin(-3.0)
    mix = saturate(mix, cfg["sat"])
    mix = ms_stage(mix, sr, cfg["mono_below"], cfg["width"])
    mix = mix / max(1e-9, np.abs(mix).max()) * db2lin(-3.0)
    save(out / "premaster.wav", mix, sr)

    # 5. Громкость + лимитер для каждой цели
    report["masters"] = {}
    finals = []
    for tgt in cfg["targets"]:
        log(f"• Лимитер → {tgt:g} LUFS / {cfg['ceiling']:g} dBTP")
        name = f"master_{tgt:g}LUFS.wav"
        ceil_used = cfg["ceiling"]
        for _ in range(3):
            y, st = finalize(mix, sr, tgt, ceil_used, cfg["release"])
            save(out / name, y, sr)
            st["ffmpeg_check"] = ffmpeg_ebur128(out / name)
            # независимая проверка ffmpeg (ITU BS.1770): если он видит пик выше потолка — опускаем потолок
            over = st["ffmpeg_check"].get("true_peak", -99) - cfg["ceiling"]
            if over <= 0.05:
                break
            ceil_used -= over + 0.05
        st["ceiling_used"] = round(ceil_used, 2)
        if st["gr_max_db"] > 6:
            report["warnings"].append(f"{name}: лимитер давит до {st['gr_max_db']} dB — возможна потеря панча; "
                                      "попробуй цель тише или --glue выше.")
        report["masters"][name] = st
        finals.append((name, y, tgt))

    # M4A (AAC) для Telegram — из самой громкой версии
    if cfg["m4a"] and shutil.which("ffmpeg"):
        name, _, tgt = max(finals, key=lambda f: f[2])
        m4a = out / name.replace(".wav", ".m4a")
        src_wav, ceil_m4a = out / name, cfg["ceiling"]
        with tempfile.TemporaryDirectory() as td:
            for attempt in range(3):
                run_cmd(["ffmpeg", "-v", "error", "-y", "-i", str(src_wav), "-c:a", "aac", "-b:a", "256k",
                         "-movflags", "+faststart", str(m4a)])
                chk = ffmpeg_ebur128(m4a)
                over = chk.get("true_peak", -99) - cfg["ceiling"]
                if over <= 0.3 or attempt == 2:
                    break
                # кодер AAC добавил пиков — делаем для M4A отдельную версию с более низким потолком
                ceil_m4a -= over + 0.1
                y_m4a, _ = finalize(mix, sr, tgt, ceil_m4a, cfg["release"])
                src_wav = Path(td) / "for_m4a.wav"
                save(src_wav, y_m4a, sr)
        report["m4a"] = {"file": m4a.name, "ceiling_used": round(ceil_m4a, 2), **chk}
        if chk.get("true_peak", -99) > -0.3:
            report["warnings"].append("M4A после кодирования подходит к 0 dBTP — поставь --ceiling -1.5.")

    # 6. A/B: оригинал и мастер на одинаковой громкости
    main_name, main_y, _ = min(finals, key=lambda f: abs(f[2] + 14))
    cmp_l = min(info_before["lufs"], report["masters"][main_name]["lufs"]) - 0.5
    ab_dir = out / "ab"
    save(ab_dir / "A_original.wav", match_loudness(original, cmp_l, sr), sr, "FLOAT")
    save(ab_dir / "B_master.wav", match_loudness(main_y, cmp_l, sr), sr, "FLOAT")
    after = analyze_signal(main_y, sr)
    report["after"] = after

    curves = [("Оригинал", match_loudness(original, -20, sr), sr), (f"Мастер ({main_name.replace('master_', '').replace('.wav', '')})", match_loudness(main_y, -20, sr), sr)]
    if args.ref:
        r, rsr = load(Path(args.ref).expanduser())
        report["reference_analysis"] = analyze_signal(r, rsr)
        curves.append(("Референс", match_loudness(r, -20, rsr), rsr))
    try:
        plot_spectra(curves, out / "spectrum.png")
    except Exception as e:  # noqa: BLE001
        report["warnings"].append(f"График не построен: {e}")

    # 7. Отчёты
    rows = [("", "Оригинал", "Мастер"),
            ("Громкость, LUFS", info_before["lufs"], after["lufs"]),
            ("True Peak, dBTP", info_before["true_peak"], after["true_peak"]),
            ("PLR (пик − громкость), dB", info_before["plr_db"], after["plr_db"]),
            ("Всплески резкости 5.5–10 кГц, %", info_before["harsh_bursts_pct"], after["harsh_bursts_pct"]),
            ("Корреляция низа (<150 Гц)", info_before["low_end_correlation"], after["low_end_correlation"])]
    rows += [(f"Доля полосы {lo}–{hi} Гц, dB", info_before["bands_db"].get(n), after["bands_db"].get(n))
             for n, lo, hi in BANDS if n in info_before["bands_db"]]
    table = "".join(f"<tr><th>{a}</th><td>{fmt(b)}</td><td>{fmt(c)}</td></tr>" if i else
                    f"<tr><th></th><th>{b}</th><th>{c}</th></tr>" for i, (a, b, c) in enumerate(rows))
    ab = [["A · Оригинал", "ab/A_original.wav"], ["B · Мастер", "ab/B_master.wav"]]
    fin = [[n.replace("master_", "").replace("LUFS.wav", " LUFS"), n] for n, _, _ in finals]
    (out / "ab.html").write_text(AB_HTML.format(
        title=base, version=out.name, cmp_lufs=round(cmp_l, 1), table=table,
        ab_json=json.dumps(ab, ensure_ascii=False), fin_json=json.dumps(fin, ensure_ascii=False)), encoding="utf-8")

    report["elapsed_s"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    md = [f"# Мастеринг: {base} ({out.name})", "",
          f"Режим: **{report['mode']}**, пресет чистки: **{cfg['preset']}**, время: {report['elapsed_s']} с", "",
          "| | Оригинал | Мастер |", "|---|---|---|"]
    md += [f"| {a} | {fmt(b)} | {fmt(c)} |" for a, b, c in rows[1:]]
    md += ["", "## Файлы", ""]
    for n, st in report["masters"].items():
        tp_show = st.get("ffmpeg_check", {}).get("true_peak", st["true_peak"])
        md.append(f"- `{n}` — {st['lufs']} LUFS, TP {tp_show} dBTP, лимитер: макс {st['gr_max_db']} dB / "
                  f"в среднем {st['gr_mean_db']} dB")
    if "m4a" in report:
        m = report["m4a"]
        md.append(f"- `{m['file']}` — {fmt(m.get('lufs'))} LUFS, TP {fmt(m.get('true_peak'))} dBTP (AAC 256 kbps, для Telegram)")
    md += ["- `premaster.wav` — после чистки и шины, до лимитера", "- `ab.html` — A/B-сравнение, `spectrum.png` — спектр"]
    if report["warnings"]:
        md += ["", "## Предупреждения", ""] + [f"- {w}" for w in report["warnings"]]
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    if args.save_preset:
        p = save_preset(args.save_preset, cfg)
        log(f"• Настройки сохранены в пресет: {p}")

    log("")
    log("\n".join(md))
    log(f"\nПапка: {out}")


def cmd_open(args) -> None:
    p = Path(args.file).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"Нет файла: {p}")
    if not webbrowser.open(p.as_uri()):
        log(f"Не удалось открыть автоматически — открой вручную: {p}")


def main() -> None:
    # консоль Windows по умолчанию не в UTF-8 — без этого падает вывод кириллицы и значков
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Мастеринг AI-треков (Suno/Udio)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="замеры трека или папки стемов (JSON)")
    a.add_argument("input")
    a.add_argument("--ref")
    a.add_argument("--exclude", help="regex: какие файлы в папке стемов пропустить")

    o = sub.add_parser("open", help="открыть ab.html / spectrum.png в системе")
    o.add_argument("file")

    r = sub.add_parser("run", help="чистка и мастеринг")
    r.add_argument("input", help="WAV/FLAC или папка со стемами")
    r.add_argument("--ref", help="референсный трек (Matchering)")
    r.add_argument("--ref-mix", type=float, help="0..1, насколько сильно подгонять под референс (1.0)")
    r.add_argument("--preset", choices=list(STRENGTH), help="сила чистки артефактов (normal)")
    r.add_argument("--targets", type=lambda s: [float(v) for v in s.split(",")], help="LUFS через запятую (-14,-11)")
    r.add_argument("--ceiling", type=float, help="потолок True Peak, dBTP (-1.0)")
    r.add_argument("--release", type=float, help="релиз лимитера, мс (150)")
    r.add_argument("--glue", type=float, help="компрессия шины на громких местах, dB (1.5; 0 = выкл)")
    r.add_argument("--sat", type=float, help="сатурация 0..1 (0.12)")
    r.add_argument("--width", type=float, help="ширина стерео (1.0)")
    r.add_argument("--mono-below", type=float, help="моно ниже, Гц (120; 0 = выкл)")
    r.add_argument("--low", type=float, help="низ, полка 100 Гц, dB")
    r.add_argument("--mud", type=float, help="«грязь», 300 Гц, dB (по пресету: -1)")
    r.add_argument("--presence", type=float, help="присутствие, 3 кГц, dB")
    r.add_argument("--air", type=float, help="воздух, полка 10 кГц, dB")
    r.add_argument("--deharsh", type=float, help="макс. подавление резкости в миксе, dB (по пресету: 3.5)")
    r.add_argument("--deess", type=float, help="макс. де-эссинг вокала, dB (по пресету: 6)")
    r.add_argument("--vocal-gain", type=float, help="громкость вокала, dB (только стемы)")
    r.add_argument("--split", choices=["none", "vocals", "4"], help="разделить полный WAV на стемы (без видеокарты — медленно)")
    r.add_argument("--sep-model", help="модель audio-separator")
    r.add_argument("--exclude", help="regex: какие файлы в папке стемов пропустить")
    r.add_argument("--out", help="папка результатов (по умолчанию <трек>_master/)")
    r.add_argument("--no-m4a", action="store_true", help="не делать M4A")
    r.add_argument("--use", help="загрузить пресет (имя или путь к JSON)")
    r.add_argument("--save-preset", help="сохранить итоговые настройки как пресет")

    # «--targets -14,-11» argparse принимает за флаг — склеиваем в «--targets=-14,-11»
    argv = sys.argv[1:]
    for i, a in enumerate(argv[:-1]):
        if a == "--targets" and argv[i + 1].startswith("-"):
            argv[i:i + 2] = [f"--targets={argv[i + 1]}"]
            break
    args = ap.parse_args(argv)
    if args.cmd == "analyze":
        cmd_analyze(args)
    elif args.cmd == "open":
        cmd_open(args)
    else:
        # имена флагов → ключи cfg
        for src_name, dst in (("ref_mix", "ref_mix"), ("mono_below", "mono_below"), ("vocal_gain", "vocal_gain"),
                              ("sep_model", "sep_model")):
            setattr(args, dst, getattr(args, src_name, None))
        cmd_run(args)


if __name__ == "__main__":
    main()
