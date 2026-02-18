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
import textwrap
import csv
import json
import math
import re
import unicodedata
from datetime import datetime, timezone
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import os
import sys

# EPS definition
EPS = 1e-12

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # py3.10-
    import tomli as tomllib  # type: ignore

# -------------------------
# Logging setup
# -------------------------

import logging
LOG = logging.getLogger("cassette_calibrator")

# -------------------------
# Version numbering; auto
# -------------------------

def _get_version() -> str:
    # 1) If installed as a package, prefer the packaged version.
    try:
        from importlib.metadata import version as pkg_version  # py3.8+
        return pkg_version("cassette-calibrator")
    except Exception:
        pass

    # 2) If running from a git checkout, use git tags.
    try:
        import subprocess
        v = subprocess.check_output(
            ["git", "describe", "--tags", "--dirty", "--always"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return v[1:] if v.startswith("v") else v
    except Exception:
        pass

    # 3) Last resort.
    return "0.0.0"

__version__ = _get_version()

# -------------------------
# Help w/ descriptions
# -------------------------

TOP_DESC = f"""\
cassette-calibrator {__version__}
By FlyingFathead / https://github.com/FlyingFathead/cassette-calibrator

Generate DTMF-marked test tapes and analyze cassette playback for:
- frequency response (ESS deconvolution)
- drift (wow/flutter-ish via time warp)
- SNR (noise window vs tone window)
"""

TOP_EPILOG = """\
Commands:

  gen
    Build a WAV you can print to cassette:
      [pre] [start marker] [noise window] [pad] [countdown] [pad]
      [1 kHz tone] [pad] [ESS sweep (+optional ticks)] [pad] [end marker] [post]

  detect
    Scan a recording and locate start/end markers (timestamps).

  analyze
    Auto-align using markers, drift-correct (linear + optional tick warp),
    deconvolve sweep -> impulse -> magnitude response, export plots + CSV + summary.json

Run: cassette_calibrator.py <command> --help  for full options.
"""

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

    Backwards-compat:
      - If config has [detect].marker_channel, treat it as [detect].channel
        (and remove marker_channel so it won't trigger "unknown key" warnings).
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

    # -----------------------------
    # Backwards-compat aliasing
    # -----------------------------
    if cmd == "detect" and "marker_channel" in out:
        # Only use it if "channel" wasn't explicitly set
        if "channel" not in out:
            out["channel"] = out["marker_channel"]
        # Always remove it so apply_config_to_subparser() won't warn
        out.pop("marker_channel", None)

    # -----------------------------
    # Small normalizations so config can be forgiving
    # -----------------------------
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

def to_float32(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)

    if y.dtype == np.uint8:
        # 8-bit PCM is unsigned
        return ((y.astype(np.float32) - 128.0) / 128.0)

    if y.dtype == np.int16:
        return (y.astype(np.float32) / 32768.0)

    if y.dtype == np.int32:
        maxabs = int(np.max(np.abs(y))) if y.size else 0
        # Heuristic: PCM24 stored in int32 usually tops out near 2^23
        if maxabs <= (2**23):
            return (y.astype(np.float32) / 8388608.0)  # 2^23
        return (y.astype(np.float32) / 2147483648.0)   # 2^31

    return y.astype(np.float32)

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

def read_wav(path: Path) -> Tuple[int, np.ndarray]:
    """
    Read WAV using scipy.io.wavfile and return (sr, y_float32).
    y is float32 in [-1, 1] (best effort), shape:
      - (n,) for mono
      - (n, ch) for multi-channel
    """
    sr, y = wavfile.read(str(path))

    y = np.asarray(y)

    # Convert to float32 full-scale
    y_f = to_float32(y)

    # If float WAV came in with >1.0 peaks (some DAWs/exporters do this),
    # scale it down so "dBFS-ish" math stays sane and marker thresholds work.
    if np.issubdtype(y_f.dtype, np.floating) and y_f.size:
        maxabs = float(np.max(np.abs(y_f)))
        if maxabs > 1.0 + 1e-6:
            y_f = (y_f / maxabs).astype(np.float32)

    return int(sr), y_f.astype(np.float32)

def read_wav_mono(path: Path) -> Tuple[int, np.ndarray]:
    sr, y = read_wav(path)
    return sr, to_mono(y)

def amp_from_dbfs(dbfs: float) -> float:
    # 0 dBFS -> 1.0 peak. -20 dBFS -> 0.1 peak.
    return float(10.0 ** (dbfs / 20.0))

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _run_id_now(now: Optional[datetime] = None) -> str:
    """
    Filesystem-friendly timestamp id, local time:
      YYYYMMDD-HHMMSS
    """
    now = now or datetime.now().astimezone()
    return now.strftime("%Y%m%d-%H%M%S")


def _slugify(s: str, maxlen: int = 48) -> str:
    """
    Turn arbitrary label into a safe ASCII-ish slug for directory names.
    Keeps: a-z 0-9 _ - .
    """
    s = (s or "").strip()
    if not s:
        return ""

    # NFKD -> ASCII best-effort (drops accents/umlauts cleanly)
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")

    s = s.lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9._-]+", "", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_-.")
    if len(s) > maxlen:
        s = s[:maxlen].rstrip("_-.")
    return s


def _unique_dir(p: Path, *, max_tries: int = 999) -> Path:
    """
    If p exists, append -NN until it doesn't.
    """
    if not p.exists():
        return p

    base = p
    for i in range(2, max_tries + 1):
        cand = base.parent / f"{base.name}-{i:02d}"
        if not cand.exists():
            return cand
    raise RuntimeError(f"Could not allocate a unique run directory under: {base.parent}")


def resolve_analyze_outdir(args: argparse.Namespace) -> Tuple[Path, dict]:
    """
    If --run-subdir is enabled (default), write into:
      <base_outdir>/<timestamp>[--<slug>]/...

    Returns:
      (resolved_outdir_path, run_meta_dict)
    """
    base_outdir = Path(str(getattr(args, "outdir", "") or "cassette_results"))
    run_name = str(getattr(args, "run_name", "") or "").strip() or None

    now_local = datetime.now().astimezone()
    now_utc = datetime.now(timezone.utc)
    run_id = _run_id_now(now_local)

    slug = _slugify(run_name or "")
    dir_name = f"{run_id}--{slug}" if slug else run_id

    run_subdir = bool(getattr(args, "run_subdir", True))

    if run_subdir:
        ensure_dir(base_outdir)
        outdir = _unique_dir(base_outdir / dir_name)
        ensure_dir(outdir)
    else:
        outdir = base_outdir
        ensure_dir(outdir)

    run_meta = {
        "id": run_id,
        "name": run_name,
        "created_at_local": now_local.isoformat(timespec="seconds"),
        "created_at_utc": now_utc.isoformat(timespec="seconds"),
        "base_outdir": str(base_outdir),
        "outdir": str(outdir),
        "run_subdir": run_subdir,
    }
    return outdir, run_meta

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

def _interp_samples_linear(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """
    Sample 1D signal x at fractional positions pos using linear interpolation.
    pos is in sample indices (0..len(x)-1-ish).
    """
    x = np.asarray(x, dtype=np.float32)
    pos = np.asarray(pos, dtype=np.float32)
    if x.size == 0:
        return np.zeros_like(pos, dtype=np.float32)

    # Pad one sample so i0+1 is always valid after clipping
    xp = np.pad(x, (0, 1), mode="edge")
    n = len(xp)

    i0 = np.floor(pos).astype(np.int64)
    i0 = np.clip(i0, 0, n - 2)
    frac = (pos - i0.astype(np.float32)).astype(np.float32)

    y0 = xp[i0]
    y1 = xp[i0 + 1]
    return (y0 * (1.0 - frac) + y1 * frac).astype(np.float32)


def warp_by_control_points(
    rec: np.ndarray,
    ref_pts: Sequence[float],
    rec_pts: Sequence[float],
    target_len: int,
) -> np.ndarray:
    """
    Piecewise-linear time warp:
      given control points mapping ref_sample_index -> rec_sample_index,
      produce output of length target_len by interpolating rec at mapped positions.

    ref_pts must be increasing; rec_pts must be nondecreasing.
    """
    rec = np.asarray(rec, dtype=np.float32)
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if rec.size <= 1:
        return np.zeros(target_len, dtype=np.float32)

    if len(ref_pts) != len(rec_pts) or len(ref_pts) < 2:
        return resample_to_length(rec, target_len)

    rp = np.asarray(ref_pts, dtype=np.float32)
    cp = np.asarray(rec_pts, dtype=np.float32)

    # Clamp into bounds
    rp = np.clip(rp, 0.0, float(target_len - 1))
    cp = np.clip(cp, 0.0, float(len(rec) - 1))

    # Enforce monotonicity + strict ref increase
    ref_clean = [float(rp[0])]
    rec_clean = [float(cp[0])]
    for r, c in zip(rp[1:], cp[1:]):
        if r <= ref_clean[-1] + 1e-6:
            continue
        if c < rec_clean[-1] - 1e-3:
            continue
        ref_clean.append(float(r))
        rec_clean.append(float(c))

    if len(ref_clean) < 2:
        return resample_to_length(rec, target_len)

    x_ref = np.arange(target_len, dtype=np.float32)
    x_rec = np.interp(
        x_ref,
        np.asarray(ref_clean, dtype=np.float32),
        np.asarray(rec_clean, dtype=np.float32),
    ).astype(np.float32)

    return _interp_samples_linear(rec, x_rec)


def match_expected_to_detected(
    expected: Sequence[Tuple[int, float]],
    detected: Sequence[float],
    tol_s: float,
) -> List[Tuple[int, float, float]]:
    """
    expected: list of (k, expected_abs_time_s)
    detected: sorted detected_abs_time_s
    returns: list of (k, expected_abs_time_s, detected_abs_time_s) in increasing time order,
             greedy 1:1 matching within tol_s.
    """
    det = list(map(float, detected))
    det.sort()

    out: List[Tuple[int, float, float]] = []
    j = 0
    tol_s = float(tol_s)
    if not math.isfinite(tol_s) or tol_s < 0.0:
        tol_s = 0.0

    for k, te in expected:
        te = float(te)

        while j < len(det) and det[j] < (te - tol_s):
            j += 1

        candidates: List[Tuple[float, float, int]] = []
        if j < len(det):
            candidates.append((abs(det[j] - te), det[j], j))
        if j > 0:
            candidates.append((abs(det[j - 1] - te), det[j - 1], j - 1))

        candidates = [c for c in candidates if c[0] <= tol_s]
        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0])
        _, td, idx = candidates[0]
        out.append((int(k), te, float(td)))
        j = idx + 1

    return out


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

def embed_dtmf_ticks_in_sweep(
    sweep: np.ndarray,
    sr: int,
    sweep_s: float,
    tick_sym: str,
    tick_interval_s: float,
    tick_offset_s: float,
    tick_tone_s: float,
    tick_amp: float,
    ramp_ms: float,
) -> np.ndarray:
    """
    Mix short DTMF ticks into the sweep at fixed times (offset + k*interval).
    This does NOT replace the sweep; it adds to it.
    """
    out = np.asarray(sweep, dtype=np.float32).copy()
    n = len(out)

    # place ticks
    t = float(tick_offset_s)
    while t < (float(sweep_s) - float(tick_tone_s)):
        i0 = int(round(t * sr))
        tone = dtmf_tone(tick_sym, sr, tick_tone_s, tick_amp, ramp_ms=ramp_ms)
        i1 = min(n, i0 + len(tone))
        if 0 <= i0 < n:
            out[i0:i1] += tone[: i1 - i0]
        t += float(tick_interval_s)

    return out.astype(np.float32)

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

    for i in range(0, len(y) - win + 1, hop):
        seg = y[i:i + win]

        # # // OLD method: quick gate before doing Goertzel
        # if rms(seg) < (min_amp * 0.2):
        #     continue

        # quick gate before doing Goertzel
        # NOTE: min_dbfs should mean what it says; don't secretly loosen it.
        if rms(seg) < min_amp:
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

        # --- two-tone sanity checks ---
        # DTMF should have BOTH a dominant ROW tone and a dominant COL tone.
        # A sweep tends to energize only one strong tone at a time -> reject that.
        r_others = (sum(row_p) - r_best) + 1e-12
        c_others = (sum(col_p) - c_best) + 1e-12
        if (r_best / r_others) < thresh_ratio:
            continue
        if (c_best / c_others) < thresh_ratio:
            continue

        # Row/col power should be roughly comparable (DTMF is two simultaneous tones).
        # Allow +/- 8 dB imbalance.
        bal_db = 10.0 * math.log10((r_best + 1e-12) / (c_best + 1e-12))
        if abs(bal_db) > 8.0:
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

def find_sequence(
    events: Sequence[DTMFEvent],
    seq: str,
    *,
    which: str = "t0",
    start_after: Optional[float] = None,
    last: bool = False,
) -> Optional[float]:
    syms = [e.sym for e in events]

    if which not in ("t", "t0"):
        raise ValueError("which must be 't' or 't0'")

    times = [float(getattr(e, which)) for e in events]

    sub = list(seq)
    found: Optional[float] = None

    for i in range(0, len(syms) - len(sub) + 1):
        if syms[i:i + len(sub)] == sub:
            t = float(times[i])
            if start_after is not None and t < float(start_after):
                continue
            found = t
            if not last:
                return found

    return found


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


def octave_smooth_linear(freq: np.ndarray, mag_lin: np.ndarray, frac: int = 12) -> np.ndarray:
    freq = np.asarray(freq)
    mag_lin = np.asarray(mag_lin, dtype=np.float32)

    lf = np.log10(np.maximum(freq, 1e-12))
    hw = (np.log10(2.0) / frac) * 0.5
    lo = lf - hw
    hi = lf + hw

    idx_lo = np.searchsorted(lf, lo, side="left")
    idx_hi = np.searchsorted(lf, hi, side="right")

    c = np.concatenate([[0.0], np.cumsum(mag_lin, dtype=np.float64)])
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
    # Frequency-domain transfer-function estimate:
    # H(f) = R(f) * conj(S(f)) / (|S(f)|^2 + eps)
    # This guarantees: if rec_sweep == ref_sweep, then H == 1 (flat), numerically.

    eps = EPS

    ref_sweep = np.asarray(ref_sweep, dtype=np.float32)
    rec_sweep = np.asarray(rec_sweep, dtype=np.float32)

    n = int(max(len(ref_sweep), len(rec_sweep)))
    if n <= 0:
        return AnalysisResult(
            freq=np.zeros(0, dtype=np.float32),
            mag_db=np.zeros(0, dtype=np.float32),
            mag_db_s=np.zeros(0, dtype=np.float32),
            ir=np.zeros(0, dtype=np.float32),
            ir_sr=sr,
        )

    nfft = int(2 ** math.ceil(math.log2(n * 2)))
    S = np.fft.rfft(ref_sweep, n=nfft)
    R = np.fft.rfft(rec_sweep, n=nfft)

    denom = (np.abs(S) ** 2).astype(np.float64)
    floor = 1e-6 * float(np.max(denom) + 1e-24)  # relative regularization
    H = (R * np.conj(S)) / (denom + floor)

    freq_full = np.fft.rfftfreq(nfft, 1.0 / sr)
    mag_lin_full = np.abs(H).astype(np.float32)

    m = (freq_full >= fmin) & (freq_full <= fmax)
    freq = freq_full[m].astype(np.float32)
    mag_lin = mag_lin_full[m].astype(np.float32)

    mag = 20.0 * np.log10(np.maximum(mag_lin, eps))

    if smooth_oct > 0:
        mag_lin_s = octave_smooth_linear(freq, mag_lin, frac=smooth_oct)
        mag_s = 20.0 * np.log10(np.maximum(mag_lin_s, eps))
    else:
        mag_s = mag.copy()

    # 1 kHz-ish normalization (keep the behavior)
    refband = (freq >= 900.0) & (freq <= 1100.0)
    if np.any(refband):
        off = float(np.median(mag[refband]))
        mag = mag - off
        mag_s = mag_s - off

    # Optional: derive an IR preview from H for impulse.png
    ir_full = np.fft.irfft(H, n=nfft).astype(np.float32)
    peak_i = int(np.argmax(np.abs(ir_full))) if ir_full.size else 0
    win = int(max(32, round(ir_win_s * sr)))
    start = max(0, peak_i - win // 4)

    ir = ir_full[start:start + win]
    if len(ir) < win:
        ir = np.pad(ir, (0, win - len(ir)))
    ir = ir * signal.windows.hann(len(ir), sym=False).astype(np.float32)

    return AnalysisResult(freq=freq, mag_db=mag, mag_db_s=mag_s, ir=ir, ir_sr=sr)


# -------------------------
# Export helpers
# -------------------------

def apply_audio_freq_ticks(ax, fmin: float, fmax: float) -> None:
    ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    ticks = [t for t in ticks if (t >= fmin and t <= fmax)]
    ax.set_xlim(fmin, fmax)
    ax.set_xticks(ticks)

    def fmt(x, _pos=None):
        if x >= 1000:
            k = x / 1000.0
            return f"{int(k)}K" if abs(k - round(k)) < 1e-9 else f"{k:.1f}K"
        return f"{int(x)}"

    ax.xaxis.set_major_formatter(FuncFormatter(fmt))
    ax.set_xlabel("Hertz/Kilohertz")

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

    if args.ticks:
        tick_amp = amp_from_dbfs(args.tick_dbfs)
        sweep = embed_dtmf_ticks_in_sweep(
            sweep=sweep,
            sr=sr,
            sweep_s=args.sweep_s,
            tick_sym=args.tick_sym,
            tick_interval_s=args.tick_interval_s,
            tick_offset_s=args.tick_offset_s,
            tick_tone_s=args.tick_tone_s,
            tick_amp=tick_amp,
            ramp_ms=args.dtmf_ramp_ms,
        )

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
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.scatter(times, np.arange(len(times)))
        ax.grid(True)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Event index")
        ax.set_title(f"DTMF detections over time (channel={args.channel})")

        if t_start is not None:
            ax.axvline(t_start, linestyle="--")
        if t_end is not None:
            ax.axvline(t_end, linestyle="--")

        plt.show()
        plt.close(fig)

def cmd_analyze(args: argparse.Namespace) -> None:
    outdir, run_meta = resolve_analyze_outdir(args)

    run_notes = str(getattr(args, "run_notes", "") or "").strip()
    run_meta["notes"] = run_notes or None

    if run_notes:
        now_local = datetime.now().astimezone()
        now_utc = datetime.now(timezone.utc)
        run_meta["notes_updated_at_local"] = now_local.isoformat(timespec="seconds")
        run_meta["notes_updated_at_utc"] = now_utc.isoformat(timespec="seconds")

    # Keep args.outdir in sync for any downstream prints/summary fields
    args.outdir = str(outdir)

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
    if t_ms is None:
        raise SystemExit(
            "Could not find start marker in recorded file. "
            "Try lowering --min-dbfs, reducing --thresh, or increasing marker level."
        )

    # Layout constants needed to sanity-gate the end marker time
    marker_dur = len(dtmf_sequence(
        args.marker_start, sr, args.marker_tone_s, args.marker_gap_s, amp_from_dbfs(args.marker_dbfs)
    )) / sr

    cd_dur = 0.0
    if args.countdown:
        tokens = countdown_tokens(args.countdown_from)
        cd_audio = []
        for tok in tokens:
            cd_audio.append(
                dtmf_sequence(
                    tok, sr, args.marker_tone_s, args.marker_gap_s,
                    amp_from_dbfs(args.marker_dbfs) * 0.9
                )
            )
            cd_audio.append(np.zeros(int(sr * 0.12), dtype=np.float32))
        cd_dur = (sum(len(a) for a in cd_audio) / sr) if cd_audio else 0.0

    expected_between = (
        marker_dur
        + args.noisewin_s + args.pad_s
        + cd_dur + args.pad_s
        + args.tone_s + args.pad_s
        + args.sweep_s + args.pad_s
    )

    # End marker should be near the end of that layout span; avoid false positives in the sweep.
    min_end = float(t_ms + 0.70 * expected_between)

    t_me = find_sequence(events, args.marker_end, which="t0", start_after=min_end, last=True)
    if t_me is None:
        # Fallback: still try "last match anywhere" instead of "first match"
        t_me = find_sequence(events, args.marker_end, which="t0", last=True)

    if t_me is None:
        raise SystemExit(
            "Could not find end marker in recorded file. "
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

    # Layout constants (REUSE earlier marker_dur + cd_dur computed above)
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
    actual_between = (t_me - t_ms)
    drift_ratio = expected_between / max(actual_between, 1e-9)

    # recorded-seconds per layout-second (always useful for ticks)
    layout_scale = 1.0 / max(drift_ratio, 1e-12)

    if abs(drift_ratio - 1.0) > args.drift_warn:
        drift_applied = True
        rec_sweep_dur = args.sweep_s * layout_scale
        sweep_start_offset_rec = sweep_start_offset * layout_scale
        print(f"[warn] drift ratio {drift_ratio:.6f} (expected/actual) -- applying linear warp")
    else:
        drift_applied = False

        # If tick-warp is enabled, still use ratio-scaled sweep window so expected tick
        # times stay close enough for matching.
        if bool(getattr(args, "ticks", False)):
            rec_sweep_dur = args.sweep_s * layout_scale
            sweep_start_offset_rec = sweep_start_offset * layout_scale
        else:
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

        # --- optional tick-based non-linear warp ---
        tick_info = {"enabled": bool(getattr(args, "ticks", False)), "used": False, "matched": 0}

        if bool(getattr(args, "ticks", False)):
            sweep_abs_start = float(t_ms + sweep_start_offset_rec)

            # Use a smaller dedupe window for ticks so we don't "eat" real repeated ticks.
            # (marker_dedupe defaults are tuned for longer marker symbols, not 60 ms ticks)
            win_s = float(args.win_ms) / 1000.0
            tick_dedupe_s = max(float(args.tick_tone_s), win_s, 3.0 * (float(args.hop_ms) / 1000.0))
            # Clamp so it can't collapse adjacent ticks
            if float(args.tick_interval_s) > 0:
                tick_dedupe_s = min(tick_dedupe_s, 0.45 * float(args.tick_interval_s))

            # Detect DTMF events inside the extracted sweep window; keep only tick_sym
            tick_events = detect_dtmf_events(
                rec_sweep_raw, sr,
                win_ms=args.win_ms,
                hop_ms=args.hop_ms,
                thresh_ratio=args.thresh,
                min_dbfs=args.min_dbfs,
                dedupe_s=tick_dedupe_s,
                log_stats=False,
            )
            det_tick_abs = [sweep_abs_start + float(e.t) for e in tick_events if e.sym == args.tick_sym]
            det_tick_abs.sort()

            # Build expected tick center times (layout timeline), then map to recorded-time using layout_scale
            expected: List[Tuple[int, float]] = []
            tick_layout_centers: List[float] = []

            k = 0
            t0 = float(args.tick_offset_s)
            while t0 < (float(args.sweep_s) - float(args.tick_tone_s)):
                t_center_layout = t0 + 0.5 * float(args.tick_tone_s)
                t_center_rec = sweep_abs_start + (t_center_layout * layout_scale)
                expected.append((k, t_center_rec))
                tick_layout_centers.append(t_center_layout)
                k += 1
                t0 += float(args.tick_interval_s)

            matches = match_expected_to_detected(expected, det_tick_abs, float(args.tick_match_tol_s))

            if len(matches) >= int(args.tick_min_matches):
                # Build control points + timing errors
                ref_pts: List[float] = [0.0]
                rec_pts: List[float] = [0.0]
                errs: List[float] = []

                for kk, te, td in matches:
                    if not (0 <= kk < len(tick_layout_centers)):
                        continue

                    t_layout = tick_layout_centers[kk]
                    ref_i = float(int(round(t_layout * sr)))
                    rec_i = float((td - sweep_abs_start) * sr)

                    if (
                        0.0 <= ref_i <= float(target_len - 1)
                        and 0.0 <= rec_i <= float(max(len(rec_sweep_raw) - 1, 0))
                    ):
                        ref_pts.append(ref_i)
                        rec_pts.append(rec_i)
                        errs.append(float(td - te))

                # Endpoints to stabilize edges
                ref_pts.append(float(target_len - 1))
                rec_pts.append(float(max(len(rec_sweep_raw) - 1, 0)))

                tick_info["matched"] = int(len(errs))  # real usable matches (not just "matches" length)

                mean_ms = float(1000.0 * np.mean(np.abs(errs))) if errs else float("inf")
                max_ms = float(1000.0 * np.max(np.abs(errs))) if errs else float("inf")

                tick_info["mean_abs_ms"] = mean_ms if math.isfinite(mean_ms) else None
                tick_info["max_abs_ms"] = max_ms if math.isfinite(max_ms) else None
                tick_info["tol_s"] = float(args.tick_match_tol_s)
                tick_info["sym"] = str(args.tick_sym)
                tick_info["interval_s"] = float(args.tick_interval_s)
                tick_info["tone_s"] = float(args.tick_tone_s)

                # Quality gate: only warp if tick timing is actually tight.
                # Otherwise you get comb-filter hell and insane spikes.
                MEAN_MAX_MS = 20.0
                MAX_MAX_MS  = 80.0

                if (mean_ms <= MEAN_MAX_MS) and (max_ms <= MAX_MAX_MS) and (tick_info["matched"] >= int(args.tick_min_matches)):
                    rec_sweep_warped = warp_by_control_points(rec_sweep_raw, ref_pts, rec_pts, target_len)
                    tick_info["used"] = True

                    print(
                        f"[info] ticks ({ch}) used matched={tick_info['matched']} "
                        f"mean_abs={mean_ms:.1f} ms max_abs={max_ms:.1f} ms"
                    )
                else:
                    rec_sweep_warped = resample_to_length(rec_sweep_raw, target_len)
                    tick_info["used"] = False

                    print(
                        f"[warn] ticks ({ch}) matched={tick_info['matched']} but timing is sloppy "
                        f"(mean_abs={mean_ms:.1f} ms, max_abs={max_ms:.1f} ms) -- skipping tick-warp"
                    )

            else:
                rec_sweep_warped = resample_to_length(rec_sweep_raw, target_len)
                tick_info["matched"] = int(len(matches))
                tick_info["used"] = False
                print(
                    f"[warn] ticks ({ch}) enabled but only matched {len(matches)} "
                    f"(need {args.tick_min_matches}); falling back to linear warp"
                )

        else:
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
            layout_scale=layout_scale if (drift_applied or bool(getattr(args, "ticks", False))) else 1.0,
        )

        snrs[ch] = snr_db
        snr_parts_all[ch] = snr_parts
        snr_warnings_all[ch] = snr_warnings

        # Export per-channel
        save_csv(outdir / f"response{out_suffix(ch)}.csv",
                res.freq, res.mag_db, res.mag_db_s, diff_db_s)

        # response plot
        fig, ax = plt.subplots()
        ax.semilogx(res.freq, res.mag_db_s)
        ax.grid(True, which="both")

        apply_audio_freq_ticks(ax, args.f_plot_min, args.f_plot_max)

        ax.set_ylabel("Magnitude (dB, smoothed)")
        ax.set_title(f"Cassette chain magnitude response ({ch})")
        fig.tight_layout()
        fig.savefig(outdir / f"response{out_suffix(ch)}.png", dpi=150)
        plt.close(fig)

        # difference plot (optional)
        if diff_db_s is not None:
            fig, ax = plt.subplots()
            ax.semilogx(res.freq, diff_db_s)
            ax.grid(True, which="both")

            apply_audio_freq_ticks(ax, args.f_plot_min, args.f_plot_max)

            ax.set_ylabel("Difference (dB, smoothed)")
            ax.set_title(f"Cassette chain minus loopback ({ch})")
            fig.tight_layout()
            fig.savefig(outdir / f"difference{out_suffix(ch)}.png", dpi=150)
            plt.close(fig)

        # impulse plot (optional)
        if args.save_ir:
            fig, ax = plt.subplots()
            tt = np.arange(len(res.ir), dtype=np.float32) / res.ir_sr
            ax.plot(tt, res.ir)
            ax.grid(True)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title(f"Windowed impulse response segment ({ch})")
            fig.tight_layout()
            fig.savefig(outdir / f"impulse{out_suffix(ch)}.png", dpi=150)
            plt.close(fig)

        per_ch[ch] = {
            "snr_db": snr_db,
            "snr_noise_rms": snr_parts.get("noise_rms"),
            "snr_tone_rms": snr_parts.get("tone_rms"),
            "snr_warnings": snr_warnings,
            "ticks": tick_info,
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

        # L/R overlay plot (smoothed)
        if bool(getattr(args, "lr_overlay", True)):
            if len(resL.freq) == len(resR.freq) and np.allclose(resL.freq, resR.freq):
                lr_freq_ov = resL.freq
                yL = resL.mag_db_s
                yR = resR.mag_db_s
            else:
                lr_freq_ov = resL.freq
                yL = resL.mag_db_s
                yR = np.interp(lr_freq_ov, resR.freq, resR.mag_db_s)

            fig, ax = plt.subplots()
            ax.semilogx(
                lr_freq_ov,
                yL,
                color=str(getattr(args, "lr_overlay_color_l", "tab:blue")),
                label="Left (L)",
            )
            ax.semilogx(
                lr_freq_ov,
                yR,
                color=str(getattr(args, "lr_overlay_color_r", "tab:orange")),
                label="Right (R)",
            )
            ax.grid(True, which="both")

            apply_audio_freq_ticks(ax, args.f_plot_min, args.f_plot_max)

            ax.set_ylabel("Magnitude (dB, smoothed)")
            ax.set_title("Cassette chain magnitude response (L/R overlay)")

            # Legend under the plot
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.14),
                ncol=2,
                frameon=False,
            )

            # Make room for the legend underneath
            fig.tight_layout(rect=[0, 0.06, 1, 1])
            fig.savefig(outdir / "response_lr_overlay.png", dpi=150)
            plt.close(fig)

            stereo_outputs["lr_overlay_png"] = str(outdir / "response_lr_overlay.png")

        if len(resL.freq) == len(resR.freq) and np.allclose(resL.freq, resR.freq):
            lr_diff = resL.mag_db_s - resR.mag_db_s
            lr_freq = resL.freq
        else:
            lr_freq = resL.freq
            lr_diff = resL.mag_db_s - np.interp(lr_freq, resR.freq, resR.mag_db_s)

        fig, ax = plt.subplots()
        ax.semilogx(lr_freq, lr_diff)
        ax.grid(True, which="both")

        apply_audio_freq_ticks(ax, args.f_plot_min, args.f_plot_max)

        ax.set_ylabel("L - R (dB, smoothed)")
        ax.set_title("Channel mismatch: L - R")
        fig.tight_layout()
        fig.savefig(outdir / "lr_diff.png", dpi=150)
        plt.close(fig)

        stereo_outputs["lr_diff_png"] = str(outdir / "lr_diff.png")

    summary = {
        "version": __version__,
        "run": run_meta,

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

        "ticks": {
            "enabled": bool(getattr(args, "ticks", False)),
            "sym": str(args.tick_sym),
            "interval_s": float(args.tick_interval_s),
            "tone_s": float(args.tick_tone_s),
            "match_tol_s": float(args.tick_match_tol_s),
            "min_matches": int(args.tick_min_matches),
        },

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
        if "lr_overlay_png" in stereo_outputs:
            print("  response_lr_overlay.png")
        print("  summary.json")


# -------------------------
# CLI
# -------------------------

def build_parser() -> Tuple[argparse.ArgumentParser, Dict[str, argparse.ArgumentParser]]:

    ap = argparse.ArgumentParser(
        prog="cassette_calibrator.py",
        description=textwrap.dedent(TOP_DESC),
        epilog=textwrap.dedent(TOP_EPILOG),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument("--version", action="version", version=f"cassette-calibrator {__version__}")

    sub = ap.add_subparsers(
        dest="cmd",
        required=True,
        title="commands",
        metavar="{gen,detect,analyze}",
    )

    # common = argparse.ArgumentParser(add_help=False)
    # common.add_argument("--config", default=None, help="TOML config path (default: auto-search)")
    # common.add_argument("--preset", default=None, help="Preset name under [presets.<name>.<cmd>]")

    ap.add_argument("--config", default=None, help="TOML config path (default: auto-search)")
    ap.add_argument("--preset", default=None, help="Preset name under [presets.<name>.<cmd>]")

    # Create subcommands ONCE
    g = sub.add_parser("gen", help="Generate marker+noisewin+tone+sweep WAV")
    d = sub.add_parser("detect", help="Detect start/end markers in a recorded file")
    a = sub.add_parser("analyze", help="Analyze recorded sweep and export response plots/data")

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

    # // ticks
    g.add_argument("--ticks", action=argparse.BooleanOptionalAction, default=False,
                help="embed periodic DTMF ticks during sweep for non-linear drift correction")
    g.add_argument("--tick-sym", default="A", help="DTMF symbol used as tick (default: A)")
    g.add_argument("--tick-interval-s", type=float, default=3.0)
    g.add_argument("--tick-tone-s", type=float, default=0.06)
    g.add_argument("--tick-dbfs", type=float, default=-18.0)
    g.add_argument("--tick-offset-s", type=float, default=1.5,
                help="first tick offset from sweep start (seconds)")


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
        "--run-name",
        default=None,
        help="Optional label for this analysis run (stored in summary.json; also used in run directory name).",
    )
    
    a.add_argument(
        "--run-notes",
        default=None,
        help="Optional long-form notes for this run (stored in summary.json).",
    )

    a.add_argument(
        "--run-subdir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write outputs into a unique timestamped subdirectory under --outdir (default: on). Use --no-run-subdir for the old flat behavior.",
    )

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

    # // ticks
    a.add_argument("--ticks", action=argparse.BooleanOptionalAction, default=False,
                help="use periodic tick-based non-linear warp if available")
    a.add_argument("--tick-sym", default="A")
    a.add_argument("--tick-interval-s", type=float, default=3.0)
    a.add_argument("--tick-tone-s", type=float, default=0.06)
    a.add_argument("--tick-dbfs", type=float, default=-18.0)   # only used for detection thresholds if you want
    a.add_argument("--tick-offset-s", type=float, default=1.5)
    a.add_argument("--tick-match-tol-s", type=float, default=0.35,
                help="max deviation (seconds) when matching expected ticks to detected ticks")
    a.add_argument("--tick-min-matches", type=int, default=4,
                help="minimum matched ticks required to enable piecewise warp")

    # Optional improvement: allow --no-save-ir if config sets it true
    a.add_argument(
        "--save-ir",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also save impulse.png",
    )

    a.add_argument("--json", action="store_true", help="print summary JSON to stdout")
    a.set_defaults(func=cmd_analyze)

    # L/R overlay plot (only meaningful when both L and R are analyzed)
    a.add_argument(
        "--lr-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="if both L and R are analyzed, also save an L/R overlay plot (default: on)",
    )
    a.add_argument(
        "--lr-overlay-color-l",
        default="tab:blue",
        help="matplotlib color for Left channel in overlay plot",
    )
    a.add_argument(
        "--lr-overlay-color-r",
        default="tab:orange",
        help="matplotlib color for Right channel in overlay plot",
    )

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

    if "--help-all" in argv:
        print(ap.format_help())
        for name in ("gen", "detect", "analyze"):
            print("\n" + "="*80 + "\n")
            print(cmd_parsers[name].format_help())
        raise SystemExit(0)

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

    if getattr(args, "cmd", None) in {"gen", "detect", "analyze"}:
        print(f"cassette-calibrator {__version__}")

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
