#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 22050


def write_wav(path: str | Path, x: np.ndarray, sr: int = SR) -> None:
    x = np.asarray(x, dtype=np.float32)
    peak = float(np.max(np.abs(x)) + 1e-9)
    x = 0.90 * (x / peak)
    wavfile.write(str(path), sr, (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16))


def adsr(n: int, a=0.01, r=0.03, sr: int = SR) -> np.ndarray:
    env = np.ones(n, dtype=np.float32)
    na = min(int(a * sr), n // 2)
    nr = min(int(r * sr), n // 2)
    if na > 0:
        env[:na] = np.linspace(0.0, 1.0, na, endpoint=False, dtype=np.float32)
    if nr > 0:
        env[-nr:] = np.linspace(1.0, 0.0, nr, endpoint=True, dtype=np.float32)
    return env


def normalize(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(x)) + 1e-9)
    return (x / m * peak).astype(np.float32)


def voiced(dur: float, f0: float = 120.0, sr: int = SR) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n, dtype=np.float32) / sr

    # robust audible buzz
    x = (
        0.7 * signal.square(2 * np.pi * f0 * t, duty=0.35) +
        0.2 * signal.sawtooth(2 * np.pi * f0 * t, width=0.55) +
        0.1 * np.sin(2 * np.pi * f0 * t)
    ).astype(np.float32)

    # light lowpass so it's less pure hell
    b, a = signal.butter(2, 3500, btype="low", fs=sr)
    x = signal.lfilter(b, a, x).astype(np.float32)
    return x


def noise(dur: float, sr: int = SR) -> np.ndarray:
    return np.random.randn(int(dur * sr)).astype(np.float32)


def bp(x: np.ndarray, lo: float, hi: float, sr: int = SR, order: int = 2) -> np.ndarray:
    lo = max(40.0, lo)
    hi = min(hi, sr * 0.5 - 100.0)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    b, a = signal.butter(order, [lo, hi], btype="bandpass", fs=sr)
    return signal.lfilter(b, a, x).astype(np.float32)


def lp(x: np.ndarray, fc: float, sr: int = SR, order: int = 2) -> np.ndarray:
    b, a = signal.butter(order, fc, btype="low", fs=sr)
    return signal.lfilter(b, a, x).astype(np.float32)


def hp(x: np.ndarray, fc: float, sr: int = SR, order: int = 2) -> np.ndarray:
    b, a = signal.butter(order, fc, btype="high", fs=sr)
    return signal.lfilter(b, a, x).astype(np.float32)


def vowel(formants: list[tuple[float, float]], dur: float, f0: float = 120.0, sr: int = SR) -> np.ndarray:
    src = voiced(dur, f0=f0, sr=sr)
    y = np.zeros_like(src)

    # broad formants, not narrow resonators
    for fc, bw in formants:
        lo = max(80.0, fc - bw * 0.5)
        hi = fc + bw * 0.5
        y += bp(src, lo, hi, sr=sr, order=2)

    y = normalize(y, 0.9)
    y *= adsr(len(y), a=0.01, r=0.04, sr=sr)
    return y.astype(np.float32)


def consonant(kind: str, dur: float, sr: int = SR) -> np.ndarray:
    src = noise(dur, sr=sr)

    if kind == "b":
        y = 1.10 * bp(src, 180, 1400, sr=sr)
    elif kind == "d":
        y = 1.15 * bp(src, 1200, 3200, sr=sr)
    elif kind == "g":
        y = 1.00 * bp(src, 300, 2200, sr=sr)
    elif kind == "t":
        y = 1.20 * bp(src, 2500, 8000, sr=sr)
    elif kind == "s":
        y = 1.10 * bp(src, 4500, 9500, sr=sr)
    elif kind == "n":
        # nasal-ish voiced murmur
        y = 1.00 * lp(voiced(dur, f0=110, sr=sr), 1400, sr=sr)
    else:
        y = np.zeros_like(src, dtype=np.float32)

    y = normalize(y, 0.9)
    y *= adsr(len(y), a=0.002, r=0.02, sr=sr)
    return y.astype(np.float32)


def gap(dur: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(dur * sr), dtype=np.float32)


# crude vowel presets
V_EE = [(300, 180), (2300, 500), (3000, 700)]   # "ee"
V_IH = [(400, 220), (1900, 500), (2550, 700)]   # "ih"
V_EH = [(550, 260), (1700, 500), (2500, 700)]   # "eh"


def begin_test(sr: int = SR) -> np.ndarray:
    parts = [
        consonant("b", 0.032, sr),
        vowel(V_EE, 0.165, f0=122, sr=sr),
        consonant("g", 0.030, sr),
        vowel(V_IH, 0.115, f0=118, sr=sr),
        consonant("n", 0.085, sr),
        gap(0.055, sr),
        consonant("t", 0.022, sr),
        vowel(V_EH, 0.125, f0=125, sr=sr),
        consonant("s", 0.065, sr),
        consonant("t", 0.020, sr),
    ]
    y = np.concatenate(parts).astype(np.float32)

    y = bp(y, 180, 4200, sr=sr)
    y = np.tanh(1.6 * y).astype(np.float32)
    return normalize(y, 0.9)

def end_test(sr: int = SR) -> np.ndarray:
    parts = [
        vowel(V_EH, 0.135, f0=118, sr=sr),   # "eh"
        consonant("n", 0.095, sr),           # "n"
        consonant("d", 0.030, sr),           # "d"
        gap(0.065, sr),
        consonant("t", 0.024, sr),           # "t"
        vowel(V_EH, 0.125, f0=124, sr=sr),   # "eh"
        consonant("s", 0.070, sr),           # "s"
        consonant("t", 0.020, sr),           # "t"
    ]
    y = np.concatenate(parts).astype(np.float32)

    # radio uglifier
    y = bp(y, 180, 4200, sr=sr)
    y = np.tanh(1.6 * y).astype(np.float32)

    # final touch: make sure the word onset isn't too weak
    return normalize(y, 0.9)

if __name__ == "__main__":
    outdir = Path("data/voice_cues")
    outdir.mkdir(parents=True, exist_ok=True)

    a = begin_test()
    b = end_test()

    write_wav(outdir / "begin_test.wav", a)
    write_wav(outdir / "end_test.wav", b)

    print("Wrote:", outdir / "begin_test.wav")
    print("Wrote:", outdir / "end_test.wav")
