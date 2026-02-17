#!/usr/bin/env python3
"""
cassette_calibrator.py

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


# -------------------------
# Utility
# -------------------------

def db(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), eps))


def rms(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x) + eps))


def to_float32(y: np.ndarray) -> np.ndarray:
    if y.ndim == 2:
        y = y.mean(axis=1)
    if y.dtype == np.int16:
        return (y.astype(np.float32) / 32767.0)
    if y.dtype == np.int32:
        return (y.astype(np.float32) / 2147483647.0)
    return y.astype(np.float32)


def write_wav_int16(path: Path, sr: int, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    y = (x * 32767.0).astype(np.int16)
    wavfile.write(str(path), sr, y)


def read_wav_mono(path: Path) -> Tuple[int, np.ndarray]:
    sr, y = wavfile.read(str(path))
    return sr, to_float32(y)


def amp_from_dbfs(dbfs: float) -> float:
    # 0 dBFS -> 1.0 peak. -20 dBFS -> 0.1 peak.
    return float(10.0 ** (dbfs / 20.0))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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


def dtmf_tone(sym: str, sr: int, dur_s: float, amp: float) -> np.ndarray:
    if sym not in DTMF:
        raise ValueError(f"Unknown DTMF symbol: {sym}")
    f1, f2 = DTMF[sym]
    n = int(sr * dur_s)
    t = np.arange(n, dtype=np.float32) / sr
    w = signal.windows.hann(n, sym=False).astype(np.float32) if n >= 16 else np.ones(n, dtype=np.float32)
    y = amp * (np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)) * w * 0.5
    return y.astype(np.float32)


def dtmf_sequence(seq: str, sr: int, tone_dur: float, gap: float, amp: float) -> np.ndarray:
    parts: List[np.ndarray] = []
    z_gap = np.zeros(int(sr * gap), dtype=np.float32)
    for ch in seq:
        parts.append(dtmf_tone(ch, sr, tone_dur, amp))
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
    t: float
    sym: str


def detect_dtmf_events(
    x: np.ndarray,
    sr: int,
    win_ms: float = 40.0,
    hop_ms: float = 10.0,
    thresh_ratio: float = 6.0,
    min_dbfs: float = -55.0,
) -> List[DTMFEvent]:
    x = np.asarray(x, dtype=np.float32)

    # Bandpass around DTMF region to reject LF rumble and very HF hiss
    sos = signal.butter(4, [600.0, 1800.0], btype="bandpass", fs=sr, output="sos")
    y = signal.sosfilt(sos, x)

    win = int(sr * win_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    if win < 64:
        raise ValueError("Window too short; increase win_ms.")

    events: List[DTMFEvent] = []
    last_sym: Optional[str] = None
    last_t: float = -1e9

    min_amp = amp_from_dbfs(min_dbfs)

    for i in range(0, len(y) - win, hop):
        seg = y[i:i + win]
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

        t = i / sr
        if sym == last_sym and (t - last_t) < 0.15:
            continue

        events.append(DTMFEvent(t=t, sym=sym))
        last_sym = sym
        last_t = t

    return events


def find_sequence(events: Sequence[DTMFEvent], seq: str) -> Optional[float]:
    syms = [e.sym for e in events]
    times = [e.t for e in events]
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
    lf = np.log10(np.maximum(freq, 1e-9))
    out = np.empty_like(mag_db_arr)
    hw = (np.log10(2.0) / frac) / 2.0
    for i in range(len(freq)):
        lo, hi = lf[i] - hw, lf[i] + hw
        m = (lf >= lo) & (lf <= hi)
        out[i] = float(np.mean(mag_db_arr[m])) if np.any(m) else float(mag_db_arr[i])
    return out


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
) -> Tuple[Optional[float], Dict[str, float], List[str]]:
    warnings: List[str] = []
    marker_dur = len(dtmf_sequence(marker_start, sr, marker_tone_s, marker_gap_s, amp_from_dbfs(marker_dbfs))) / sr

    cd_dur = 0.0
    if countdown:
        tokens = countdown_tokens(countdown_from)
        cd_audio = []
        for tok in tokens:
            cd_audio.append(dtmf_sequence(tok, sr, marker_tone_s, marker_gap_s, amp_from_dbfs(marker_dbfs) * 0.9))
            cd_audio.append(np.zeros(int(sr * 0.12), dtype=np.float32))
        cd_dur = (sum(len(a) for a in cd_audio) / sr) if cd_audio else 0.0

    # Noise window is right after marker_start
    noise_start = t_marker_start + marker_dur
    noise_dur = max(0.0, min(noisewin_s, snr_noise_s if snr_noise_s > 0 else noisewin_s))
    noise_seg = extract(rec, noise_start, noise_dur, sr)
    if len(noise_seg) < int(0.2 * sr):
        warnings.append("Noise window too short or missing; SNR may be unavailable.")
        noise_r = float("nan")
    else:
        noise_r = rms(noise_seg)

    # Tone start:
    tone_start = t_marker_start + marker_dur + noisewin_s + pad_s + cd_dur + pad_s
    # Take tone from the middle to avoid edges
    if snr_tone_s <= 0:
        tone_meas_dur = max(0.0, min(5.0, tone_s))
    else:
        tone_meas_dur = min(snr_tone_s, tone_s)

    if tone_s <= 0.2:
        warnings.append("Tone duration too short for SNR measurement.")
        tone_r = float("nan")
    else:
        mid = tone_start + (tone_s * 0.5) - (tone_meas_dur * 0.5)
        tone_seg = extract(rec, mid, tone_meas_dur, sr)
        if len(tone_seg) < int(0.2 * sr):
            warnings.append("Tone segment too short/missing; SNR may be unavailable.")
            tone_r = float("nan")
        else:
            # Optionally bandpass around tone to reduce broadband hiss affecting RMS:
            sos = signal.butter(4, [tone_hz * 0.8, tone_hz * 1.2], btype="bandpass", fs=sr, output="sos")
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

    marker_start = dtmf_sequence(args.marker_start, sr, args.marker_tone_s, args.marker_gap_s, marker_amp)
    marker_end = dtmf_sequence(args.marker_end, sr, args.marker_tone_s, args.marker_gap_s, marker_amp)

    noisewin = np.zeros(int(sr * args.noisewin_s), dtype=np.float32)

    cd = np.zeros(0, dtype=np.float32)
    if args.countdown:
        tokens = countdown_tokens(args.countdown_from)
        parts: List[np.ndarray] = []
        for tok in tokens:
            parts.append(dtmf_sequence(tok, sr, args.marker_tone_s, args.marker_gap_s, marker_amp * 0.9))
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
    sr, x = read_wav_mono(Path(args.wav))
    events = detect_dtmf_events(
        x, sr,
        win_ms=args.win_ms,
        hop_ms=args.hop_ms,
        thresh_ratio=args.thresh,
        min_dbfs=args.min_dbfs,
    )
    t_start = find_sequence(events, args.marker_start)
    t_end = find_sequence(events, args.marker_end)

    result = {
        "wav": str(args.wav),
        "sr": sr,
        "marker_start": args.marker_start,
        "marker_end": args.marker_end,
        "t_marker_start": t_start,
        "t_marker_end": t_end,
        "events": len(events),
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"Detected {len(events)} DTMF events")
    print(f"marker_start '{args.marker_start}': {t_start if t_start is not None else 'NOT FOUND'}")
    print(f"marker_end   '{args.marker_end}': {t_end if t_end is not None else 'NOT FOUND'}")

    if args.dump_events:
        for e in events:
            print(f"{e.t:8.3f}s  {e.sym}")

    if args.plot:
        times = [e.t for e in events]
        plt.figure()
        plt.scatter(times, np.arange(len(times)))
        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.ylabel("Event index")
        plt.title("DTMF detections over time")
        if t_start is not None:
            plt.axvline(t_start, linestyle="--")
        if t_end is not None:
            plt.axvline(t_end, linestyle="--")
        plt.show()


def cmd_analyze(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    sr_ref, ref = read_wav_mono(Path(args.ref))
    sr_rec, rec = read_wav_mono(Path(args.rec))
    if sr_ref != sr_rec:
        raise SystemExit(f"Sample rates differ: ref={sr_ref}, rec={sr_rec}. Resample one first.")
    sr = sr_ref

    # Detect markers in recording
    events = detect_dtmf_events(
        rec, sr,
        win_ms=args.win_ms,
        hop_ms=args.hop_ms,
        thresh_ratio=args.thresh,
        min_dbfs=args.min_dbfs,
    )
    t_ms = find_sequence(events, args.marker_start)
    t_me = find_sequence(events, args.marker_end)
    if t_ms is None or t_me is None:
        raise SystemExit(
            "Could not find start/end markers in recorded file. "
            "Try lowering --min-dbfs, reducing --thresh, or increasing marker level."
        )
    if t_me <= t_ms:
        raise SystemExit("Marker end occurs before start. Wrong marker strings or bad detection.")

    # Detect marker_start in ref too (robust if user edited pre-silence)
    events_ref = detect_dtmf_events(
        ref, sr,
        win_ms=args.win_ms,
        hop_ms=args.hop_ms,
        thresh_ratio=args.thresh,
        min_dbfs=args.min_dbfs,
    )
    t_rs = find_sequence(events_ref, args.marker_start)
    if t_rs is None:
        t_rs = args.pre_s  # fallback assumption

    # Compute layout offsets
    marker_dur = len(dtmf_sequence(args.marker_start, sr, args.marker_tone_s, args.marker_gap_s, amp_from_dbfs(args.marker_dbfs))) / sr

    cd_dur = 0.0
    if args.countdown:
        tokens = countdown_tokens(args.countdown_from)
        cd_audio = []
        for tok in tokens:
            cd_audio.append(dtmf_sequence(tok, sr, args.marker_tone_s, args.marker_gap_s, amp_from_dbfs(args.marker_dbfs) * 0.9))
            cd_audio.append(np.zeros(int(sr * 0.12), dtype=np.float32))
        cd_dur = (sum(len(a) for a in cd_audio) / sr) if cd_audio else 0.0

    # After marker_start: noisewin + pad + (countdown) + pad + tone + pad => sweep
    sweep_start_offset = marker_dur + args.noisewin_s + args.pad_s + cd_dur + args.pad_s + args.tone_s + args.pad_s

    ref_sweep = extract(ref, t_rs + sweep_start_offset, args.sweep_s, sr)
    rec_sweep_raw = extract(rec, t_ms + sweep_start_offset, args.sweep_s, sr)

    # Drift ratio using marker-to-marker distance (approx; marker timestamps point to first symbol)
    marker_end_dur = len(dtmf_sequence(args.marker_end, sr, args.marker_tone_s, args.marker_gap_s, amp_from_dbfs(args.marker_dbfs))) / sr
    expected_between = (marker_dur + args.noisewin_s + args.pad_s + cd_dur + args.pad_s + args.tone_s + args.pad_s + args.sweep_s + args.pad_s + marker_end_dur)
    actual_between = (t_me - t_ms)
    drift_ratio = expected_between / max(actual_between, 1e-9)

    drift_applied = False
    if abs(drift_ratio - 1.0) > args.drift_warn:
        drift_applied = True
        print(f"[warn] drift ratio {drift_ratio:.6f} (expected/actual) -- applying linear warp")

    target_len = len(ref_sweep)
    warped_len = max(16, int(round(len(rec_sweep_raw) * drift_ratio)))
    rec_sweep_warped = resample_to_length(rec_sweep_raw, warped_len)
    rec_sweep_warped = resample_to_length(rec_sweep_warped, target_len)

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
            print(f"[info] fine-align lag={lag} samples ({lag/sr*1000:.2f} ms)")

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

    diff_db_s = None
    if args.loopback:
        sr_lb, lb = read_wav_mono(Path(args.loopback))
        if sr_lb != sr:
            raise SystemExit("Loopback sample rate differs. Resample first.")

        events_lb = detect_dtmf_events(lb, sr, win_ms=args.win_ms, hop_ms=args.hop_ms, thresh_ratio=args.thresh, min_dbfs=args.min_dbfs)
        t_lbs = find_sequence(events_lb, args.marker_start)
        if t_lbs is None:
            raise SystemExit("Could not find marker_start in loopback file.")
        lb_sweep = extract(lb, t_lbs + sweep_start_offset, args.sweep_s, sr)
        lb_sweep = resample_to_length(lb_sweep, len(ref_sweep))

        res_lb = analyze_chain(
            ref_sweep=ref_sweep,
            rec_sweep=lb_sweep,
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

    # SNR
    snr_db, snr_parts, snr_warnings = compute_snr(
        rec=rec,
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
    )

    # Export plots/data
    save_csv(outdir / "response.csv", res.freq, res.mag_db, res.mag_db_s, diff_db_s)

    plt.figure()
    plt.semilogx(res.freq, res.mag_db_s)
    plt.grid(True, which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB, smoothed)")
    plt.title("Cassette chain magnitude response")
    plt.tight_layout()
    plt.savefig(outdir / "response.png", dpi=150)

    if diff_db_s is not None:
        plt.figure()
        plt.semilogx(res.freq, diff_db_s)
        plt.grid(True, which="both")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Difference (dB, smoothed)")
        plt.title("Cassette chain minus loopback")
        plt.tight_layout()
        plt.savefig(outdir / "difference.png", dpi=150)

    if args.save_ir:
        plt.figure()
        t = np.arange(len(res.ir), dtype=np.float32) / res.ir_sr
        plt.plot(t, res.ir)
        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Windowed impulse response segment")
        plt.tight_layout()
        plt.savefig(outdir / "impulse.png", dpi=150)

    summary = {
        "sr": sr,
        "ref": str(args.ref),
        "rec": str(args.rec),
        "loopback": str(args.loopback) if args.loopback else None,

        "marker_start": args.marker_start,
        "marker_end": args.marker_end,
        "t_marker_start": t_ms,
        "t_marker_end": t_me,

        "expected_between_s": expected_between,
        "actual_between_s": actual_between,
        "drift_ratio_expected_over_actual": drift_ratio,
        "drift_applied": drift_applied,

        "snr_db": snr_db,
        "snr_noise_rms": snr_parts.get("noise_rms"),
        "snr_tone_rms": snr_parts.get("tone_rms"),
        "snr_warnings": snr_warnings,

        "f1": args.f1,
        "f2": args.f2,
        "sweep_s": args.sweep_s,
        "ir_win_s": args.ir_win_s,
        "smooth_oct": args.smooth_oct,

        "outputs": {
            "response_png": str(outdir / "response.png"),
            "response_csv": str(outdir / "response.csv"),
            "difference_png": str(outdir / "difference.png") if diff_db_s is not None else None,
            "impulse_png": str(outdir / "impulse.png") if args.save_ir else None,
            "summary_json": str(outdir / "summary.json"),
        }
    }

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(summary, indent=2))

    print(f"Wrote results to: {outdir}")
    print("  response.png, response.csv, summary.json")
    if diff_db_s is not None:
        print("  difference.png (loopback comparison)")
    if args.save_ir:
        print("  impulse.png")


# -------------------------
# CLI
# -------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cassette_calibrator.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="Generate marker+noisewin+tone+sweep WAV")
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

    d = sub.add_parser("detect", help="Detect start/end markers in a recorded file")
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
    d.set_defaults(func=cmd_detect)

    a = sub.add_parser("analyze", help="Analyze recorded sweep and export response plots/data")
    a.add_argument("--ref", required=True, help="reference generated WAV (the one you played out)")
    a.add_argument("--rec", required=True, help="recorded capture from cassette playback")
    a.add_argument("--loopback", default=None, help="optional loopback capture to subtract interface coloration")
    a.add_argument("--outdir", default="cassette_results")

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

    a.add_argument("--drift-warn", type=float, default=0.002, help="warn if drift ratio differs more than this")
    a.add_argument("--fine-align", action="store_true", help="apply small correlation-based lag correction after warp")

    # SNR knobs
    a.add_argument("--snr-noise-s", type=float, default=1.0, help="seconds used from noise window (<= noisewin-s)")
    a.add_argument("--snr-tone-s", type=float, default=3.0, help="seconds used from middle of tone for RMS")

    a.add_argument("--save-ir", action="store_true", help="also save impulse.png")
    a.add_argument("--json", action="store_true", help="print summary JSON to stdout")

    a.set_defaults(func=cmd_analyze)

    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
