#!/usr/bin/env python3
"""
cassette_calibrator.py
====================================================
https://github.com/FlyingFathead/cassette-calibrator
====================================================

What this tool does:

1) gen:
   Builds a WAV you can print to cassette:
     [pre_silence] [DTMF start marker] [noise window silence] [pad]
     [DTMF countdown (default on; disable with --no-countdown)] [pad]
     [1 kHz ref tone] [pad]
     [ESS log sweep] [pad]
     [DTMF end marker] [post_silence]

2) detect:
   Scans a recorded WAV and finds start/end marker times automatically.

3) analyze:
   - finds markers in the recorded capture (no manual alignment)
   - extracts the sweep region based on known layout
   - applies linear time-warp based on marker-to-marker drift
   - optional fine-align via correlation
   - ESS deconvolution -> impulse response -> magnitude response
   - exports plots + CSV
   - measures SNR using:
       noise RMS from the dedicated noise window
       tone RMS from the middle of the 1 kHz tone
   - writes summary.json and optionally prints JSON to stdout

Dependencies:
  python3 -m pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

import os
import sys

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # py3.10-
    import tomli as tomllib  # type: ignore

import logging
LOG = logging.getLogger("cassette_calibrator")

# -------------------------
# Config loaders & helpers
# -------------------------

def _scan_argv_value(argv: List[str], opt: str) -> Optional[str]:
    """
    Find --opt VALUE or --opt=VALUE anywhere in argv.
    """
    for i, a in enumerate(argv):
        if a == opt and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(opt + "="):
            return a.split("=", 1)[1]
    return None


def deep_merge_dict(a: dict, b: dict) -> dict:
    """
    Return a new dict where b overrides a (recursive for dict values).
    """
    out = dict(a or {})
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def load_toml_config(path: Optional[str]) -> dict:
    """
    Load TOML config dict. If path is None, try a few defaults.
    """
    candidates: List[Path] = []
    if path:
        candidates.append(Path(path).expanduser())
    else:
        # Local project file first
        candidates.append(Path("cassette_calibrator.toml"))
        candidates.append(Path("cassette-calibrator.toml"))
        # XDG-ish fallback
        xdg = os.environ.get("XDG_CONFIG_HOME")
        if xdg:
            candidates.append(Path(xdg) / "cassette-calibrator" / "config.toml")
        else:
            candidates.append(Path.home() / ".config" / "cassette-calibrator" / "config.toml")

    cfg_path = next((p for p in candidates if p.exists()), None)
    if cfg_path is None:
        return {}

    try:
        with cfg_path.open("rb") as f:
            cfg = tomllib.load(f)
        if not isinstance(cfg, dict):
            return {}
        cfg["_config_path"] = str(cfg_path)
        return cfg
    except Exception as e:
        raise SystemExit(f"Failed to read TOML config '{cfg_path}': {e}")


def _normalize_marker_channel(v: str) -> str:
    s = str(v).strip().lower()
    if s in ["mono"]:
        return "mono"
    if s in ["l", "left"]:
        return "L"
    if s in ["r", "right"]:
        return "R"
    return str(v)


def flatten_cmd_defaults(cmd: str, section: dict) -> dict:
    """
    Turn a TOML section into argparse defaults (flat dict of dest->value).
    Handles nested analyze.snr.
    """
    out: dict = {}
    if not isinstance(section, dict):
        return out

    for k, v in section.items():
        if cmd == "analyze" and k == "snr" and isinstance(v, dict):
            # [analyze.snr] noise_s / tone_s -> snr_noise_s / snr_tone_s
            if "noise_s" in v:
                out["snr_noise_s"] = v["noise_s"]
            if "tone_s" in v:
                out["snr_tone_s"] = v["tone_s"]
            continue

        out[k] = v

    # Small normalizations so config can be forgiving:
    if cmd in ["detect", "analyze"] and "marker_channel" in out:
        out["marker_channel"] = _normalize_marker_channel(out["marker_channel"])
    if cmd == "detect" and "channel" in out:
        out["channel"] = _normalize_marker_channel(out["channel"])
    if cmd == "analyze" and "channels" in out:
        # Config injection bypasses argparse type=parse_channels, so normalize here.
        out["channels"] = parse_channels(str(out["channels"]))

    return out


def apply_config_to_subparser(cmd_parser: argparse.ArgumentParser, defaults: dict, *, label: str = "") -> None:
    """
    Apply defaults only for dests that exist in this parser. Warn on unknown keys.
    """
    dests = {a.dest for a in cmd_parser._actions}  # pylint: disable=protected-access
    usable = {}
    unknown = []

    for k, v in (defaults or {}).items():
        if k in dests:
            usable[k] = v
        else:
            unknown.append(k)

    if usable:
        cmd_parser.set_defaults(**usable)

    if unknown:
        where = f" ({label})" if label else ""
        print(f"[warn] Ignoring unknown config keys{where}: {', '.join(sorted(unknown))}", file=sys.stderr)


# -------------------------
# Utility
# -------------------------

def db(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), eps))

def rms(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x) + eps))

def to_float32(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    # DO NOT downmix here. Keep channels intact.
    if y.dtype == np.int16:
        return (y.astype(np.float32) / 32768.0)
    if y.dtype == np.int32:
        return (y.astype(np.float32) / 2147483648.0)
    return y.astype(np.float32)

def write_wav_int16(path: Path, sr: int, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    y = (x * 32767.0).astype(np.int16)
    wavfile.write(str(path), sr, y)

def to_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y
    return y.mean(axis=1).astype(np.float32)

def is_stereo_audio(y: np.ndarray) -> bool:
    y = np.asarray(y)
    return (y.ndim == 2 and y.shape[1] >= 2)

def read_wav(path: Path) -> Tuple[int, np.ndarray]:
    sr, y = wavfile.read(str(path))
    return sr, to_float32(y)

def pick_channel(y: np.ndarray, channel: str) -> np.ndarray:
    """
    channel: "mono", "L", "R"
    Returns 1D float32.
    """
    y = np.asarray(y, dtype=np.float32)
    c = (channel or "mono").strip().lower()

    if y.ndim == 1:
        return y

    if c in ["l", "left"]:
        return y[:, 0].astype(np.float32)
    if c in ["r", "right"]:
        idx = 1 if y.shape[1] > 1 else 0
        return y[:, idx].astype(np.float32)

    return y.mean(axis=1).astype(np.float32)

def read_wav_mono(path: Path) -> Tuple[int, np.ndarray]:
    sr, y = read_wav(path)
    return sr, to_mono(y)

def amp_from_dbfs(dbfs: float) -> float:
    # 0 dBFS -> 1.0 peak. -20 dBFS -> 0.1 peak.
    return float(10.0 ** (dbfs / 20.0))

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def json_sanitize(obj):
    """Convert NaN/Inf into None and make common NumPy types JSON-safe."""
    # NumPy scalars (np.float32 etc.)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None

    # Optional: NumPy ints/bools
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # Optional: arrays -> lists
    if isinstance(obj, np.ndarray):
        return json_sanitize(obj.tolist())

    if isinstance(obj, dict):
        return {k: json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]

    return obj

def extract(x: np.ndarray, start_s: float, dur_s: float, sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    start = int(round(start_s * sr))
    n = int(round(dur_s * sr))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    if start < 0:
        x = np.pad(x, (abs(start), 0))
        start = 0
    end = start + n
    if end > len(x):
        x = np.pad(x, (0, end - len(x)))
    return x[start:end]

def resample_to_length(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == target_len:
        return x
    if len(x) <= 1:
        return np.zeros(target_len, dtype=np.float32)

    ratio = target_len / max(len(x), 1)
    frac = Fraction(ratio).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    y = signal.resample_poly(x, up, down).astype(np.float32)
    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    return y

def parse_channels(s: str) -> str:
    s = (s or "").strip().lower()
    aliases = {
        "stereo": "stereo",
        "lr": "stereo",
        "l+r": "stereo",

        "mono": "mono",

        "l": "l",
        "left": "l",

        "r": "r",
        "right": "r",
    }
    if s not in aliases:
        raise argparse.ArgumentTypeError("invalid --channels (use mono|stereo|l|r; aliases: left/right/lr/l+r)")
    return aliases[s]

def _strip_known_opts(argv: List[str], opts: List[str]) -> List[str]:
    out = []
    i = 0
    while i < len(argv):
        a = argv[i]
        matched = False
        for opt in opts:
            if a == opt:
                i += 2  # drop opt + value
                matched = True
                break
            if a.startswith(opt + "="):
                i += 1  # drop opt=value
                matched = True
                break
        if not matched:
            out.append(a)
            i += 1
    return out

# -------------------------
# DTMF markers
# -------------------------

DTMF: Dict[str, Tuple[int, int]] = {
    "1": (697, 1209), "2": (697, 1336), "3": (697, 1477), "A": (697, 1633),
    "4": (770, 1209), "5": (770, 1336), "6": (770, 1477), "B": (770, 1633),
    "7": (852, 1209), "8": (852, 1336), "9": (852, 1477), "C": (852, 1633),
    "*": (941, 1209), "0": (941, 1336), "#": (941, 1477), "D": (941, 1633),
}
ROWS = [697, 770, 852, 941]
COLS = [1209, 1336, 1477, 1633]

# helper for dialtones
def raised_cosine_env(n: int, sr: int, ramp_ms: float) -> np.ndarray:
    env = np.ones(n, dtype=np.float32)
    r = int(round(sr * ramp_ms / 1000.0))
    if r <= 0:
        return env
    if 2 * r >= n:
        # too short tone for the ramp; fall back to full Hann
        return signal.windows.hann(n, sym=False).astype(np.float32)

    k = np.arange(1, r + 1, dtype=np.float32) / r
    ramp = 0.5 - 0.5 * np.cos(np.pi * k)  # 0 -> 1
    env[:r] = ramp
    env[-r:] = ramp[::-1]
    return env

def dtmf_tone(sym: str, sr: int, dur_s: float, amp: float, ramp_ms: float = 5.0) -> np.ndarray:
    if sym not in DTMF:
        raise ValueError(f"Unknown DTMF symbol: {sym}")
    f1, f2 = DTMF[sym]
    n = int(sr * dur_s)
    t = np.arange(n, dtype=np.float32) / sr

    y = amp * 0.5 * (np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t))
    y *= raised_cosine_env(n, sr, ramp_ms)
    return y.astype(np.float32)

def dtmf_sequence(seq: str, sr: int, tone_dur: float, gap: float, amp: float, ramp_ms: float = 5.0) -> np.ndarray:
    parts: List[np.ndarray] = []
    z_gap = np.zeros(int(sr * gap), dtype=np.float32)
    for ch in seq:
        parts.append(dtmf_tone(ch, sr, tone_dur, amp, ramp_ms=ramp_ms))
        parts.append(z_gap)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

def countdown_tokens(n: int = 10) -> List[str]:
    return [str(n)] + [str(k) for k in range(n - 1, 0, -1)]

def goertzel_power(x: np.ndarray, sr: int, f: float) -> float:
    w = 2.0 * np.pi * f / sr
    cos_w = math.cos(w)
    coeff = 2.0 * cos_w
    s0 = s1 = s2 = 0.0
    for v in x:
        s0 = float(v) + coeff * s1 - s2
        s2 = s1
        s1 = s0
    power = s1 * s1 + s2 * s2 - coeff * s1 * s2
    return power

@dataclass
class DTMFEvent:
    t: float      # window-center time (nice for plots/logs)
    t0: float     # window-start time (layout anchor)
    sym: str

def detect_dtmf_events(
    x: np.ndarray,
    sr: int,
    win_ms: float = 40.0,
    hop_ms: float = 10.0,
    thresh_ratio: float = 6.0,
    min_dbfs: float = -55.0,
    dedupe_s: float = 0.15,
    log_stats: bool = False,
) -> List[DTMFEvent]:

    x = np.asarray(x, dtype=np.float32)

    # Bandpass around DTMF region to reject LF rumble and very HF hiss
    sos = signal.butter(4, [600.0, 1800.0], btype="bandpass", fs=sr, output="sos")
    y = signal.sosfilt(sos, x)

    win = int(sr * win_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    if win < 64:
        raise ValueError("Window too short; increase win_ms.")

    # Sanitize dedupe_s (avoid negative/NaN causing weird behavior)
    try:
        dedupe_s = float(dedupe_s)
    except Exception:
        dedupe_s = 0.0
    if (not math.isfinite(dedupe_s)) or dedupe_s < 0.0:
        dedupe_s = 0.0

    events: List[DTMFEvent] = []
    last_sym: Optional[str] = None
    last_t: float = -1e9

    min_amp = amp_from_dbfs(min_dbfs)

    for i in range(0, len(y) - win, hop):
        seg = y[i:i + win]

        # quick gate before doing Goertzel
        if rms(seg) < (min_amp * 0.2):
            continue

        seg = seg * signal.windows.hann(len(seg), sym=False).astype(np.float32)

        row_p = [goertzel_power(seg, sr, f) for f in ROWS]
        col_p = [goertzel_power(seg, sr, f) for f in COLS]
        r_idx = int(np.argmax(row_p))
        c_idx = int(np.argmax(col_p))
        r_best = row_p[r_idx]
        c_best = col_p[c_idx]
        best = r_best + c_best
        others = (sum(row_p) - r_best) + (sum(col_p) - c_best) + 1e-12

        if best / others < thresh_ratio:
            continue

        r_f = ROWS[r_idx]
        c_f = COLS[c_idx]
        sym = None
        for k, (a, b) in DTMF.items():
            if a == r_f and b == c_f:
                sym = k
                break
        if sym is None:
            continue

        # Two timestamps:
        # t  = window center (nice for scatter plots)
        # t0 = window start (layout anchor for extraction/drift math)
        t = (i + 0.5 * win) / sr
        t0 = i / sr

        # dedupe based on center-time to collapse repeated frame detections
        if sym == last_sym and (t - last_t) < dedupe_s:
            continue

        events.append(DTMFEvent(t=t, t0=t0, sym=sym))
        last_sym = sym
        last_t = t

    if log_stats and events:
        dts = np.diff([e.t for e in events])
        if len(dts):
            LOG.info(
                "DTMF stats: events=%d, dedupe_s=%.3f, dt median=%.3f s, min=%.3f s, max=%.3f s",
                len(events), float(dedupe_s),
                float(np.median(dts)), float(np.min(dts)), float(np.max(dts)),
            )
        else:
            LOG.info("DTMF stats: events=%d, dedupe_s=%.3f", len(events), float(dedupe_s))

    return events

def auto_dedupe_s(args: argparse.Namespace) -> float:
    win_s = args.win_ms / 1000.0
    hop_s = args.hop_ms / 1000.0
    upper = (args.marker_tone_s + args.marker_gap_s) - 0.5 * hop_s
    base  = args.marker_tone_s + 0.5 * win_s
    raw   = max(base, win_s, 3.0 * hop_s)
    return raw if upper <= 0 else min(raw, upper)

def find_sequence(events: Sequence[DTMFEvent], seq: str, *, which: str = "t0") -> Optional[float]:
    syms = [e.sym for e in events]

    if which not in ("t", "t0"):
        raise ValueError("which must be 't' or 't0'")

    times = [getattr(e, which) for e in events]

    sub = list(seq)
    for i in range(0, len(syms) - len(sub) + 1):
        if syms[i:i + len(sub)] == sub:
            return times[i]
    return None


# -------------------------
# ESS sweep + analysis
# -------------------------

def ess_sweep(f1: float, f2: float, T: float, sr: int, amp: float) -> np.ndarray:
    n = int(T * sr)
    t = np.arange(n, dtype=np.float32) / sr
    K = T / np.log(f2 / f1)
    L = 2.0 * np.pi * f1 * K
    phase = L * (np.exp(t / K) - 1.0)
    return (amp * np.sin(phase)).astype(np.float32)


def ess_inverse_filter(f1: float, f2: float, T: float, sr: int, sweep: np.ndarray) -> np.ndarray:
    sweep = np.asarray(sweep, dtype=np.float32)
    n = len(sweep)
    k = np.log(f2 / f1) / T
    t = np.arange(n, dtype=np.float32) / sr
    w = np.exp(t * k).astype(np.float32)
    inv = (sweep[::-1] * w).astype(np.float32)
    inv = inv / (np.sum(inv * inv) + 1e-12)
    return inv


def octave_smooth(freq: np.ndarray, mag_db_arr: np.ndarray, frac: int = 12) -> np.ndarray:
    freq = np.asarray(freq)
    mag_db_arr = np.asarray(mag_db_arr)
    lf = np.log10(np.maximum(freq, 1e-12))

    hw = (np.log10(2.0) / frac) * 0.5
    lo = lf - hw
    hi = lf + hw

    idx_lo = np.searchsorted(lf, lo, side="left")
    idx_hi = np.searchsorted(lf, hi, side="right")

    c = np.concatenate([[0.0], np.cumsum(mag_db_arr)])
    denom = np.maximum(1, idx_hi - idx_lo)
    out = (c[idx_hi] - c[idx_lo]) / denom
    return out.astype(np.float32)


@dataclass
class AnalysisResult:
    freq: np.ndarray
    mag_db: np.ndarray
    mag_db_s: np.ndarray
    ir: np.ndarray
    ir_sr: int


def analyze_chain(
    ref_sweep: np.ndarray,
    rec_sweep: np.ndarray,
    sr: int,
    f1: float,
    f2: float,
    T: float,
    ir_win_s: float,
    smooth_oct: int,
    fmin: float,
    fmax: float,
) -> AnalysisResult:
    inv = ess_inverse_filter(f1, f2, T, sr, ref_sweep)
    ir_full = signal.fftconvolve(rec_sweep, inv, mode="full").astype(np.float32)

    peak_i = int(np.argmax(np.abs(ir_full)))
    win = int(max(32, round(ir_win_s * sr)))
    start = peak_i - win // 4
    if start < 0:
        start = 0
    ir = ir_full[start:start + win]
    if len(ir) < win:
        ir = np.pad(ir, (0, win - len(ir)))
    ir = ir * signal.windows.hann(len(ir), sym=False).astype(np.float32)

    nfft = int(2 ** math.ceil(math.log2(len(ir) * 8)))
    H = np.fft.rfft(ir, n=nfft)
    freq = np.fft.rfftfreq(nfft, 1.0 / sr)
    mag = db(H)

    m = (freq >= fmin) & (freq <= fmax)
    freq = freq[m]
    mag = mag[m]
    mag_s = octave_smooth(freq, mag, frac=smooth_oct) if smooth_oct > 0 else mag.copy()

    return AnalysisResult(freq=freq, mag_db=mag, mag_db_s=mag_s, ir=ir, ir_sr=sr)


# -------------------------
# Export helpers
# -------------------------

def save_csv(path: Path, freq: np.ndarray, mag_db_arr: np.ndarray, mag_db_s_arr: np.ndarray, diff_db_s: Optional[np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["freq_hz", "mag_db", "mag_db_s"]
        if diff_db_s is not None:
            header.append("diff_db_s")
        w.writerow(header)
        for i in range(len(freq)):
            row = [float(freq[i]), float(mag_db_arr[i]), float(mag_db_s_arr[i])]
            if diff_db_s is not None:
                row.append(float(diff_db_s[i]))
            w.writerow(row)


def compute_snr(
    rec: np.ndarray,
    sr: int,
    t_marker_start: float,
    marker_start: str,
    marker_tone_s: float,
    marker_gap_s: float,
    marker_dbfs: float,
    noisewin_s: float,
    pad_s: float,
    countdown: bool,
    countdown_from: int,
    tone_s: float,
    tone_hz: float,
    snr_noise_s: float,
    snr_tone_s: float,
    layout_scale: float = 1.0,
) -> Tuple[Optional[float], Dict[str, float], List[str]]:
    warnings: List[str] = []

    layout_scale = float(layout_scale)
    if not np.isfinite(layout_scale) or layout_scale <= 0:
        layout_scale = 1.0

    marker_dur_layout = len(
        dtmf_sequence(
            marker_start,
            sr,
            marker_tone_s,
            marker_gap_s,
            amp_from_dbfs(marker_dbfs),
        )
    ) / sr
    marker_dur = marker_dur_layout * layout_scale

    cd_dur_layout = 0.0
    if countdown:
        tokens = countdown_tokens(countdown_from)
        cd_audio = []
        for tok in tokens:
            cd_audio.append(
                dtmf_sequence(
                    tok,
                    sr,
                    marker_tone_s,
                    marker_gap_s,
                    amp_from_dbfs(marker_dbfs) * 0.9,
                )
            )
            cd_audio.append(np.zeros(int(sr * 0.12), dtype=np.float32))
        cd_dur_layout = (sum(len(a) for a in cd_audio) / sr) if cd_audio else 0.0
    cd_dur = cd_dur_layout * layout_scale

    # Scale layout durations into recorded-time space (because of transport drift)
    noisewin_rec = noisewin_s * layout_scale
    pad_rec = pad_s * layout_scale
    tone_s_rec = tone_s * layout_scale

    # Noise window is right after marker_start
    noise_start = t_marker_start + marker_dur
    noise_use = (snr_noise_s if snr_noise_s > 0 else noisewin_s) * layout_scale
    noise_dur = max(0.0, min(noisewin_rec, noise_use))
    noise_seg = extract(rec, noise_start, noise_dur, sr)
    if len(noise_seg) < int(0.2 * sr):
        warnings.append("Noise window too short or missing; SNR may be unavailable.")
        noise_r = float("nan")
    else:
        noise_r = rms(noise_seg)

    # Tone start:
    tone_start = t_marker_start + marker_dur + noisewin_rec + pad_rec + cd_dur + pad_rec

    # Take tone from the middle to avoid edges
    if snr_tone_s <= 0:
        # auto: measure up to 5 seconds (in recorded-time space), limited by tone length
        tone_meas_dur = max(0.0, min(5.0, tone_s_rec))
    else:
        # user value is in layout seconds -> scale into recorded-time
        tone_meas_dur = min(snr_tone_s * layout_scale, tone_s_rec)

    if tone_s_rec <= 0.2:
        warnings.append("Tone duration too short for SNR measurement.")
        tone_r = float("nan")
    else:
        mid = tone_start + (tone_s_rec * 0.5) - (tone_meas_dur * 0.5)
        tone_seg = extract(rec, mid, tone_meas_dur, sr)
        if len(tone_seg) < int(0.2 * sr):
            warnings.append("Tone segment too short/missing; SNR may be unavailable.")
            tone_r = float("nan")
        else:
            # Optionally bandpass around tone to reduce broadband hiss affecting RMS:
            nyq = 0.5 * sr
            lo = max(10.0, tone_hz * 0.8)
            hi = min(nyq - 10.0, tone_hz * 1.2)

            if lo >= hi:
                warnings.append("Tone bandpass invalid for given tone_hz/sr; using unfiltered RMS.")
                tone_r = rms(tone_seg)
            else:
                sos = signal.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
                tone_f = signal.sosfilt(sos, tone_seg)
                tone_r = rms(tone_f)

    if not (np.isfinite(noise_r) and np.isfinite(tone_r)) or noise_r <= 0:
        return None, {"noise_rms": noise_r, "tone_rms": tone_r}, warnings

    snr_db = 20.0 * math.log10(tone_r / max(noise_r, 1e-12))
    return float(snr_db), {"noise_rms": noise_r, "tone_rms": tone_r}, warnings


# -------------------------
# Commands
# -------------------------

def cmd_gen(args: argparse.Namespace) -> None:
    sr = args.sr
    out_path = Path(args.out)
    ensure_dir(out_path.parent if out_path.parent != Path(".") else Path("."))

    marker_amp = amp_from_dbfs(args.marker_dbfs)
    tone_amp = amp_from_dbfs(args.tone_dbfs)
    sweep_amp = amp_from_dbfs(args.sweep_dbfs)

    pre = np.zeros(int(sr * args.pre_s), dtype=np.float32)
    post = np.zeros(int(sr * args.post_s), dtype=np.float32)
    pad = np.zeros(int(sr * args.pad_s), dtype=np.float32)

    marker_start = dtmf_sequence(
        args.marker_start, sr, args.marker_tone_s, args.marker_gap_s, marker_amp,
        ramp_ms=args.dtmf_ramp_ms
    )
    marker_end = dtmf_sequence(
        args.marker_end, sr, args.marker_tone_s, args.marker_gap_s, marker_amp,
        ramp_ms=args.dtmf_ramp_ms
    )

    noisewin = np.zeros(int(sr * args.noisewin_s), dtype=np.float32)

    cd = np.zeros(0, dtype=np.float32)
    if args.countdown:
        tokens = countdown_tokens(args.countdown_from)
        parts: List[np.ndarray] = []
        for tok in tokens:
            parts.append(dtmf_sequence(
                tok, sr, args.marker_tone_s, args.marker_gap_s, marker_amp * 0.9,
                ramp_ms=args.dtmf_ramp_ms
            ))
            parts.append(np.zeros(int(sr * 0.12), dtype=np.float32))
        cd = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    t = np.arange(int(sr * args.tone_s), dtype=np.float32) / sr
    ref_tone = (tone_amp * np.sin(2 * np.pi * args.tone_hz * t)).astype(np.float32)

    sweep = ess_sweep(args.f1, args.f2, args.sweep_s, sr, amp=sweep_amp)

    x = np.concatenate([
        pre,
        marker_start,
        noisewin,
        pad,
        cd,
        pad,
        ref_tone,
        pad,
        sweep,
        pad,
        marker_end,
        post
    ]).astype(np.float32)

    # Keep peak under args.peak (do not normalize to 0 dBFS)
    peak = float(np.max(np.abs(x)) + 1e-12)
    if peak > args.peak:
        x = x * (args.peak / peak)

    write_wav_int16(out_path, sr, x)
    print(f"Wrote: {out_path}")
    print(f"  sr={sr}, dur={len(x)/sr:.2f}s, peak={np.max(np.abs(x)):.3f}")
    print(f"  marker_start='{args.marker_start}', marker_end='{args.marker_end}'")
    print(f"  noisewin_s={args.noisewin_s}")
    print(f"  tone={args.tone_hz}Hz for {args.tone_s}s at {args.tone_dbfs} dBFS peak")
    print(f"  sweep={args.f1}-{args.f2}Hz for {args.sweep_s}s at {args.sweep_dbfs} dBFS peak")


def cmd_detect(args: argparse.Namespace) -> None:
    sr, y = read_wav(Path(args.wav))
    x = pick_channel(y, args.channel)

    dedupe_s = args.dtmf_dedupe_s if args.dtmf_dedupe_s is not None else auto_dedupe_s(args)

    events = detect_dtmf_events(
        x, sr,
        win_ms=args.win_ms,
        hop_ms=args.hop_ms,
        thresh_ratio=args.thresh,
        min_dbfs=args.min_dbfs,
        dedupe_s=dedupe_s,
        log_stats=getattr(args, "dtmf_stats", False),
    )

    # Anchor times (layout) -- these are what you want for extraction and drift math
    t_start = find_sequence(events, args.marker_start, which="t0")
    t_end = find_sequence(events, args.marker_end, which="t0")

    # Center times (debug/plot-friendly)
    t_start_c = find_sequence(events, args.marker_start, which="t")
    t_end_c = find_sequence(events, args.marker_end, which="t")

    result = {
        "wav": str(args.wav),
        "sr": sr,
        "channel": args.channel,
        "marker_start": args.marker_start,
        "marker_end": args.marker_end,

        # Keep existing keys as anchors (the “real” times you should use)
        "t_marker_start": t_start,
        "t_marker_end": t_end,

        # Extra debug fields
        "t_marker_start_center": t_start_c,
        "t_marker_end_center": t_end_c,

        "events": len(events),
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"Detected {len(events)} DTMF events (channel={args.channel})")
    print(f"marker_start '{args.marker_start}': {t_start if t_start is not None else 'NOT FOUND'}")
    print(f"marker_end   '{args.marker_end}': {t_end if t_end is not None else 'NOT FOUND'}")

    if args.dump_events:
        for e in events:
            print(f"{e.t0:8.3f}s  (anchor)  {e.t:8.3f}s  (center)  {e.sym}")

    if args.plot:
        times = [e.t for e in events]
        plt.figure()
        plt.scatter(times, np.arange(len(times)))
        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.ylabel("Event index")
        plt.title(f"DTMF detections over time (channel={args.channel})")
        if t_start is not None:
            plt.axvline(t_start, linestyle="--")
        if t_end is not None:
            plt.axvline(t_end, linestyle="--")
        plt.show()


def cmd_analyze(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    sr_ref, ref_y = read_wav(Path(args.ref))
    sr_rec, rec_y = read_wav(Path(args.rec))
    if sr_ref != sr_rec:
        raise SystemExit(f"Sample rates differ: ref={sr_ref}, rec={sr_rec}. Resample one first.")
    sr = sr_ref

    # -------------------------
    # DTMF detection knobs (analyze)
    # -------------------------

    dedupe_s = args.dtmf_dedupe_s if args.dtmf_dedupe_s is not None else auto_dedupe_s(args)
    dtmf_log_stats = bool(getattr(args, "dtmf_stats", False))

    rec_is_stereo = is_stereo_audio(rec_y)

    # If user asked for L/R marker channel but recording is mono, force marker channel to mono
    if (not rec_is_stereo) and args.marker_channel in ["L", "R"]:
        print("[warn] Recorded capture is mono; forcing --marker-channel mono.")
        args.marker_channel = "mono"

    # Marker detection uses a selectable channel (default mono)
    rec_marker = pick_channel(rec_y, args.marker_channel)
    ref_marker = pick_channel(ref_y, args.marker_channel)

    # Detect markers in recording (marker channel)
    events = detect_dtmf_events(
        rec_marker, sr,
        win_ms=args.win_ms,
        hop_ms=args.hop_ms,
        thresh_ratio=args.thresh,
        min_dbfs=args.min_dbfs,
        dedupe_s=dedupe_s,
        log_stats=dtmf_log_stats,
    )

    # Use anchor timestamps for layout/drift math
    t_ms = find_sequence(events, args.marker_start, which="t0")
    t_me = find_sequence(events, args.marker_end, which="t0")

    if t_ms is None or t_me is None:
        raise SystemExit(
            "Could not find start/end markers in recorded file. "
            "Try lowering --min-dbfs, reducing --thresh, or increasing marker level."
        )
    if t_me <= t_ms:
        raise SystemExit("Marker end occurs before start. Wrong marker strings or bad detection.")

    # Detect marker_start in ref too (robust if user edited pre-silence)
    events_ref = detect_dtmf_events(
        ref_marker, sr,
        win_ms=args.win_ms,
        hop_ms=args.hop_ms,
        thresh_ratio=args.thresh,
        min_dbfs=args.min_dbfs,
        dedupe_s=dedupe_s,
        log_stats=dtmf_log_stats,
    )

    t_rs = find_sequence(events_ref, args.marker_start, which="t0")
    if t_rs is None:
        t_rs = args.pre_s  # fallback assumption

    # Layout constants
    marker_dur = len(dtmf_sequence(
        args.marker_start, sr, args.marker_tone_s, args.marker_gap_s, amp_from_dbfs(args.marker_dbfs)
    )) / sr

    cd_dur = 0.0
    if args.countdown:
        tokens = countdown_tokens(args.countdown_from)
        cd_audio = []
        for tok in tokens:
            cd_audio.append(dtmf_sequence(tok, sr, args.marker_tone_s, args.marker_gap_s, amp_from_dbfs(args.marker_dbfs) * 0.9))
            cd_audio.append(np.zeros(int(sr * 0.12), dtype=np.float32))
        cd_dur = (sum(len(a) for a in cd_audio) / sr) if cd_audio else 0.0

    sweep_start_offset = (
        marker_dur
        + args.noisewin_s + args.pad_s
        + cd_dur + args.pad_s
        + args.tone_s + args.pad_s
    )

    # Reference sweep always taken from mono of ref (gen output is mono anyway)
    ref_mono = pick_channel(ref_y, "mono")
    ref_sweep = extract(ref_mono, t_rs + sweep_start_offset, args.sweep_s, sr)
    target_len = len(ref_sweep)

    # Drift ratio using marker-to-marker distance (timestamps at START of marker sequence)
    expected_between = (
        marker_dur
        + args.noisewin_s + args.pad_s
        + cd_dur + args.pad_s
        + args.tone_s + args.pad_s
        + args.sweep_s + args.pad_s
    )
    actual_between = (t_me - t_ms)
    drift_ratio = expected_between / max(actual_between, 1e-9)

    if abs(drift_ratio - 1.0) > args.drift_warn:
        drift_applied = True
        rec_sweep_dur = args.sweep_s / max(drift_ratio, 1e-9)
        sweep_start_offset_rec = sweep_start_offset / max(drift_ratio, 1e-9)
        print(f"[warn] drift ratio {drift_ratio:.6f} (expected/actual) -- applying linear warp")
    else:
        drift_applied = False
        rec_sweep_dur = args.sweep_s
        sweep_start_offset_rec = sweep_start_offset

    # Loopback pre-load (optional) + marker location for loopback
    lb_y = None
    t_lbs = None
    t_lbe = None
    drift_ratio_lb = 1.0
    drift_applied_lb = False
    lb_sweep_dur = args.sweep_s
    sweep_start_offset_lb = sweep_start_offset

    # Finalize requested_mode ONCE (and keep it)
    requested_mode = args.channels  # already canonical: stereo|mono|l|r

    if not rec_is_stereo and requested_mode != "mono":
        print("[warn] Recorded capture is mono; analyzing mono.")
        requested_mode = "mono"

    if requested_mode == "stereo":
        ch_list = ["L", "R"]
    elif requested_mode == "l":
        ch_list = ["L"]
    elif requested_mode == "r":
        ch_list = ["R"]
    else:  # mono
        ch_list = ["mono"]

    analyzing_lr = ("L" in ch_list) or ("R" in ch_list)

    if args.loopback:
        sr_lb, lb_y = read_wav(Path(args.loopback))

        if sr_lb != sr:
            raise SystemExit("Loopback sample rate differs. Resample first.")

        if analyzing_lr and (not is_stereo_audio(lb_y)):
            print("[warn] Loopback file is mono; using the same loopback response for both L and R subtraction.")

        lb_marker = pick_channel(lb_y, args.marker_channel)
        events_lb = detect_dtmf_events(
            lb_marker, sr,
            win_ms=args.win_ms,
            hop_ms=args.hop_ms,
            thresh_ratio=args.thresh,
            min_dbfs=args.min_dbfs,
            dedupe_s=dedupe_s,
            log_stats=dtmf_log_stats,
        )

        t_lbs = find_sequence(events_lb, args.marker_start, which="t0")
        t_lbe = find_sequence(events_lb, args.marker_end, which="t0")

        if t_lbs is None or t_lbe is None or t_lbe <= t_lbs:
            raise SystemExit("Could not find valid markers in loopback file.")

        actual_between_lb = (t_lbe - t_lbs)
        drift_ratio_lb = expected_between / max(actual_between_lb, 1e-9)

        if abs(drift_ratio_lb - 1.0) > args.drift_warn:
            drift_applied_lb = True
            lb_sweep_dur = args.sweep_s / max(drift_ratio_lb, 1e-9)
            sweep_start_offset_lb = sweep_start_offset / max(drift_ratio_lb, 1e-9)
            print(f"[warn] loopback drift ratio {drift_ratio_lb:.6f} -- applying linear warp")
        else:
            drift_applied_lb = False
            lb_sweep_dur = args.sweep_s
            sweep_start_offset_lb = sweep_start_offset

    def out_suffix(ch: str) -> str:
        # keep legacy names if analyzing only mono
        if ch_list == ["mono"]:
            return ""
        return "_" + ch.lower()

    per_ch = {}
    results = {}      # AnalysisResult
    diffs = {}        # Optional[np.ndarray]
    snrs = {}         # Optional[float]
    snr_parts_all = {}
    snr_warnings_all = {}

    for ch in ch_list:
        rec_ch = pick_channel(rec_y, ch)
        rec_sweep_raw = extract(rec_ch, t_ms + sweep_start_offset_rec, rec_sweep_dur, sr)
        rec_sweep_warped = resample_to_length(rec_sweep_raw, target_len)

        if args.fine_align:
            corr = signal.fftconvolve(rec_sweep_warped, ref_sweep[::-1], mode="full")
            lag = int(np.argmax(corr) - (len(ref_sweep) - 1))
            if lag != 0:
                if lag > 0:
                    rec_sweep_warped = rec_sweep_warped[lag:]
                    rec_sweep_warped = np.pad(rec_sweep_warped, (0, target_len - len(rec_sweep_warped)))
                else:
                    rec_sweep_warped = np.pad(rec_sweep_warped, (abs(lag), 0))
                    rec_sweep_warped = rec_sweep_warped[:target_len]
                print(f"[info] fine-align ({ch}) lag={lag} samples ({lag/sr*1000:.2f} ms)")

        res = analyze_chain(
            ref_sweep=ref_sweep,
            rec_sweep=rec_sweep_warped,
            sr=sr,
            f1=args.f1,
            f2=args.f2,
            T=args.sweep_s,
            ir_win_s=args.ir_win_s,
            smooth_oct=args.smooth_oct,
            fmin=args.f_plot_min,
            fmax=args.f_plot_max,
        )
        results[ch] = res

        diff_db_s = None
        if lb_y is not None:
            lb_ch = pick_channel(lb_y, ch)
            lb_sweep_raw = extract(lb_ch, t_lbs + sweep_start_offset_lb, lb_sweep_dur, sr)
            lb_sweep_warped = resample_to_length(lb_sweep_raw, target_len)

            res_lb = analyze_chain(
                ref_sweep=ref_sweep,
                rec_sweep=lb_sweep_warped,
                sr=sr,
                f1=args.f1,
                f2=args.f2,
                T=args.sweep_s,
                ir_win_s=args.ir_win_s,
                smooth_oct=args.smooth_oct,
                fmin=args.f_plot_min,
                fmax=args.f_plot_max,
            )

            if len(res_lb.freq) == len(res.freq) and np.allclose(res_lb.freq, res.freq):
                diff_db_s = res.mag_db_s - res_lb.mag_db_s
            else:
                diff_db_s = res.mag_db_s - np.interp(res.freq, res_lb.freq, res_lb.mag_db_s)

        diffs[ch] = diff_db_s

        # SNR per channel
        snr_db, snr_parts, snr_warnings = compute_snr(
            rec=rec_ch,
            sr=sr,
            t_marker_start=t_ms,
            marker_start=args.marker_start,
            marker_tone_s=args.marker_tone_s,
            marker_gap_s=args.marker_gap_s,
            marker_dbfs=args.marker_dbfs,
            noisewin_s=args.noisewin_s,
            pad_s=args.pad_s,
            countdown=args.countdown,
            countdown_from=args.countdown_from,
            tone_s=args.tone_s,
            tone_hz=args.tone_hz,
            snr_noise_s=args.snr_noise_s,
            snr_tone_s=args.snr_tone_s,
            layout_scale=(1.0 / drift_ratio) if drift_applied else 1.0,
        )

        snrs[ch] = snr_db
        snr_parts_all[ch] = snr_parts
        snr_warnings_all[ch] = snr_warnings

        # Export per-channel
        save_csv(outdir / f"response{out_suffix(ch)}.csv", res.freq, res.mag_db, res.mag_db_s, diff_db_s)

        plt.figure()
        plt.semilogx(res.freq, res.mag_db_s)
        plt.grid(True, which="both")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB, smoothed)")
        plt.title(f"Cassette chain magnitude response ({ch})")
        plt.tight_layout()
        plt.savefig(outdir / f"response{out_suffix(ch)}.png", dpi=150)
        plt.close()

        if diff_db_s is not None:
            plt.figure()
            plt.semilogx(res.freq, diff_db_s)
            plt.grid(True, which="both")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Difference (dB, smoothed)")
            plt.title(f"Cassette chain minus loopback ({ch})")
            plt.tight_layout()
            plt.savefig(outdir / f"difference{out_suffix(ch)}.png", dpi=150)
            plt.close()

        if args.save_ir:
            plt.figure()
            tt = np.arange(len(res.ir), dtype=np.float32) / res.ir_sr
            plt.plot(tt, res.ir)
            plt.grid(True)
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.title(f"Windowed impulse response segment ({ch})")
            plt.tight_layout()
            plt.savefig(outdir / f"impulse{out_suffix(ch)}.png", dpi=150)
            plt.close()

        per_ch[ch] = {
            "snr_db": snr_db,
            "snr_noise_rms": snr_parts.get("noise_rms"),
            "snr_tone_rms": snr_parts.get("tone_rms"),
            "snr_warnings": snr_warnings,
            "outputs": {
                "response_png": str(outdir / f"response{out_suffix(ch)}.png"),
                "response_csv": str(outdir / f"response{out_suffix(ch)}.csv"),
                "difference_png": str(outdir / f"difference{out_suffix(ch)}.png") if diffs[ch] is not None else None,
                "impulse_png": str(outdir / f"impulse{out_suffix(ch)}.png") if args.save_ir else None,
            }
        }

    # If stereo, export L-R smoothed difference plot
    stereo_outputs = {}
    if "L" in results and "R" in results:
        resL = results["L"]
        resR = results["R"]
        if len(resL.freq) == len(resR.freq) and np.allclose(resL.freq, resR.freq):
            lr_diff = resL.mag_db_s - resR.mag_db_s
            lr_freq = resL.freq
        else:
            lr_freq = resL.freq
            lr_diff = resL.mag_db_s - np.interp(lr_freq, resR.freq, resR.mag_db_s)

        plt.figure()
        plt.semilogx(lr_freq, lr_diff)
        plt.grid(True, which="both")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("L - R (dB, smoothed)")
        plt.title("Channel mismatch: L - R")
        plt.tight_layout()
        plt.savefig(outdir / "lr_diff.png", dpi=150)
        plt.close()

        stereo_outputs["lr_diff_png"] = str(outdir / "lr_diff.png")

    summary = {
        "sr": sr,
        "ref": str(args.ref),
        "rec": str(args.rec),
        "loopback": str(args.loopback) if args.loopback else None,

        "channels_analyzed": ch_list,
        "marker_channel": args.marker_channel,

        "marker_start": args.marker_start,
        "marker_end": args.marker_end,
        "t_marker_start": t_ms,
        "t_marker_end": t_me,

        "expected_between_s": expected_between,
        "actual_between_s": actual_between,
        "drift_ratio_expected_over_actual": drift_ratio,
        "drift_applied": drift_applied,

        "loopback_drift_ratio": drift_ratio_lb if args.loopback else None,
        "loopback_drift_applied": drift_applied_lb if args.loopback else None,

        "f1": args.f1,
        "f2": args.f2,
        "sweep_s": args.sweep_s,
        "ir_win_s": args.ir_win_s,
        "smooth_oct": args.smooth_oct,

        "per_channel": per_ch,
        "stereo_outputs": stereo_outputs,

        "summary_json": str(outdir / "summary.json"),
    }

    safe_summary = json_sanitize(summary)

    (outdir / "summary.json").write_text(
        json.dumps(safe_summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    if args.json:
        print(json.dumps(safe_summary, indent=2, allow_nan=False))

    print(f"Wrote results to: {outdir}")
    if ch_list == ["mono"]:
        print("  response.png, response.csv, summary.json")
        if args.loopback:
            print("  difference.png")
        if args.save_ir:
            print("  impulse.png")
    else:
        for ch in ch_list:
            print(f"  response_{ch.lower()}.png, response_{ch.lower()}.csv")
            if args.loopback:
                print(f"  difference_{ch.lower()}.png")
            if args.save_ir:
                print(f"  impulse_{ch.lower()}.png")
        if "lr_diff_png" in stereo_outputs:
            print("  lr_diff.png")
        print("  summary.json")


# -------------------------
# CLI
# -------------------------

def build_parser() -> Tuple[argparse.ArgumentParser, Dict[str, argparse.ArgumentParser]]:
    ap = argparse.ArgumentParser(prog="cassette_calibrator.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="TOML config path (default: auto-search)")
    common.add_argument("--preset", default=None, help="Preset name under [presets.<name>.<cmd>]")

    # Create subcommands ONCE
    g = sub.add_parser("gen", help="Generate marker+noisewin+tone+sweep WAV", parents=[common])
    d = sub.add_parser("detect", help="Detect start/end markers in a recorded file", parents=[common])
    a = sub.add_parser("analyze", help="Analyze recorded sweep and export response plots/data", parents=[common])

    # -------- gen --------
    g.add_argument("--dtmf-ramp-ms", type=float, default=5.0,
                   help="DTMF attack/release ramp in ms (short = pointier; 0 = hard gate)")
    g.add_argument("--out", default="sweepcass.wav")
    g.add_argument("--sr", type=int, default=44100)
    g.add_argument("--pre-s", type=float, default=1.0)
    g.add_argument("--post-s", type=float, default=1.0)
    g.add_argument("--pad-s", type=float, default=0.30, help="silence pads between sections")
    g.add_argument("--noisewin-s", type=float, default=1.0, help="dedicated silence window for SNR measurement")
    g.add_argument("--marker-start", default="99#*")
    g.add_argument("--marker-end", default="*#99")
    g.add_argument("--marker-tone-s", type=float, default=0.22)
    g.add_argument("--marker-gap-s", type=float, default=0.04)
    g.add_argument("--marker-dbfs", type=float, default=-12.0, help="DTMF marker level (peak dBFS)")

    g.add_argument(
        "--countdown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include DTMF countdown (default: on; disable with --no-countdown)",
    )
    g.add_argument("--countdown-from", type=int, default=10)
    g.add_argument("--tone-hz", type=float, default=1000.0)
    g.add_argument("--tone-s", type=float, default=10.0)
    g.add_argument("--tone-dbfs", type=float, default=-18.0, help="ref tone level (peak dBFS)")
    g.add_argument("--f1", type=float, default=20.0)
    g.add_argument("--f2", type=float, default=20000.0)
    g.add_argument("--sweep-s", type=float, default=30.0)
    g.add_argument("--sweep-dbfs", type=float, default=-24.0, help="sweep level (peak dBFS)")
    g.add_argument("--peak", type=float, default=0.90, help="hard peak limiter for whole file")
    g.set_defaults(func=cmd_gen)

    # ------- detect -------
    d.add_argument("--channel", choices=["mono", "L", "R"], default="mono",
                   help="which channel to use for DTMF detection")
    d.add_argument("--wav", required=True)
    d.add_argument("--marker-start", default="99#*")
    d.add_argument("--marker-end", default="*#99")
    d.add_argument("--win-ms", type=float, default=40.0)
    d.add_argument("--hop-ms", type=float, default=10.0)
    d.add_argument("--thresh", type=float, default=6.0)
    d.add_argument("--min-dbfs", type=float, default=-55.0)
    d.add_argument("--dump-events", action="store_true")
    d.add_argument("--plot", action="store_true")
    d.add_argument("--json", action="store_true", help="print JSON to stdout")
    d.add_argument("--marker-tone-s", type=float, default=0.22)
    d.add_argument("--marker-gap-s", type=float, default=0.04)
    d.add_argument(
        "--dtmf-dedupe-s",
        type=float,
        default=None,
        help="merge same-symbol detections within this many seconds "
            "(default: marker_tone_s + 0.5*win_s; clamped when meaningful to avoid eating real repeats)"
    )
    d.add_argument("--dtmf-stats", action="store_true", help="print DTMF timing stats")
    d.set_defaults(func=cmd_detect)

    # ------- analyze -------
    a.add_argument("--channels", type=parse_channels, default="stereo")
    a.add_argument("--marker-channel", choices=["mono", "L", "R"], default="mono",
                   help="channel used for DTMF marker detection/layout timing")
    a.add_argument("--ref", required=True, help="reference generated WAV (the one you played out)")
    a.add_argument("--rec", required=True, help="recorded capture from cassette playback")
    a.add_argument("--loopback", default=None, help="optional loopback capture to subtract interface coloration")
    a.add_argument("--outdir", default="cassette_results")

    a.add_argument(
        "--dtmf-dedupe-s",
        type=float,
        default=None,
        help="merge same-symbol detections within this many seconds "
            "(default: marker_tone_s + 0.5*win_s; clamped when meaningful to avoid eating real repeats)"
    )
    a.add_argument("--dtmf-stats", action="store_true", help="print DTMF timing stats")

    a.add_argument("--marker-start", default="99#*")
    a.add_argument("--marker-end", default="*#99")
    a.add_argument("--marker-tone-s", type=float, default=0.22)
    a.add_argument("--marker-gap-s", type=float, default=0.04)
    a.add_argument("--marker-dbfs", type=float, default=-12.0)

    a.add_argument(
        "--countdown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="expect DTMF countdown in layout (default: on; disable with --no-countdown)",
    )
    a.add_argument("--countdown-from", type=int, default=10)

    a.add_argument("--pre-s", type=float, default=1.0)
    a.add_argument("--pad-s", type=float, default=0.30)
    a.add_argument("--noisewin-s", type=float, default=1.0)

    a.add_argument("--tone-s", type=float, default=10.0)
    a.add_argument("--tone-hz", type=float, default=1000.0)

    a.add_argument("--f1", type=float, default=20.0)
    a.add_argument("--f2", type=float, default=20000.0)
    a.add_argument("--sweep-s", type=float, default=30.0)

    a.add_argument("--ir-win-s", type=float, default=0.25)
    a.add_argument("--smooth-oct", type=int, default=12)
    a.add_argument("--f-plot-min", type=float, default=20.0)
    a.add_argument("--f-plot-max", type=float, default=20000.0)

    a.add_argument("--win-ms", type=float, default=40.0)
    a.add_argument("--hop-ms", type=float, default=10.0)
    a.add_argument("--thresh", type=float, default=6.0)
    a.add_argument("--min-dbfs", type=float, default=-55.0)

    a.add_argument("--drift-warn", type=float, default=0.002,
                   help="warn if drift ratio differs more than this")
    a.add_argument(
        "--fine-align",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply small correlation-based lag correction after warp",
    )

    a.add_argument("--snr-noise-s", type=float, default=1.0,
                   help="seconds used from noise window (<= noisewin-s)")
    a.add_argument("--snr-tone-s", type=float, default=3.0,
                   help="seconds used from middle of tone for RMS")

    # Optional improvement: allow --no-save-ir if config sets it true
    a.add_argument(
        "--save-ir",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also save impulse.png",
    )

    a.add_argument("--json", action="store_true", help="print summary JSON to stdout")
    a.set_defaults(func=cmd_analyze)

    return ap, {"gen": g, "detect": d, "analyze": a}

# ----------------------
# MAIN
# ----------------------

def main() -> None:
    argv = sys.argv[1:]

    # Grab config/preset anywhere in argv (before/after subcommand)
    cfg_path = _scan_argv_value(argv, "--config")
    preset = _scan_argv_value(argv, "--preset")

    cfg = load_toml_config(cfg_path)

    ap, cmd_parsers = build_parser()

    # Apply base sections: [gen], [detect], [analyze]
    for cmd, p in cmd_parsers.items():
        base = flatten_cmd_defaults(cmd, cfg.get(cmd, {}))
        apply_config_to_subparser(p, base, label=f"{cmd}")

    # Apply optional presets: [presets.<name>.<cmd>]
    if preset:
        presets = cfg.get("presets", {})
        pset = presets.get(preset, {}) if isinstance(presets, dict) else {}
        if not isinstance(pset, dict):
            pset = {}
        for cmd, p in cmd_parsers.items():
            override = flatten_cmd_defaults(cmd, pset.get(cmd, {}))
            apply_config_to_subparser(p, override, label=f"presets.{preset}.{cmd}")

    argv2 = _strip_known_opts(argv, ["--config", "--preset"])
    args = ap.parse_args(argv2)

    # Basic console logging (only if nothing configured yet)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # If user asked for DTMF stats, ensure INFO is visible even if env preconfigured logging
    if bool(getattr(args, "dtmf_stats", False)):
        logging.getLogger().setLevel(logging.INFO)
        LOG.setLevel(logging.INFO)

    args.func(args)

if __name__ == "__main__":
    main()
