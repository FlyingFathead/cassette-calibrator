#!/usr/bin/env python3
"""
webui.py

Local-only WebUI for cassette-calibrator.
- Binds to 127.0.0.1 by default (configurable via [webui] in cassette_calibrator.toml)
- Requires cassette-calibrator's deps (e.g., matplotlib). Uses tomllib (py3.11+) or tomli (py<=3.10).
- Calls the "core program" by importing cassette_calibrator.py and invoking cmd_* funcs

Security posture:
- Rejects absolute paths and any ".." path traversal
- Only serves/browses files under the allowed WebUI root
- The allowed WebUI root is the project root by default, or a configured subdirectory when restricted mode is enabled
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import html as pyhtml
from functools import lru_cache
import io
import json
import os
import sys
import time
import csv
import math
import re
import shutil
import unicodedata
from datetime import datetime
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse, parse_qs, quote

# Force a headless matplotlib backend BEFORE importing matplotlib / cassette_calibrator
os.environ.setdefault("MPLBACKEND", "Agg")

# --- TOML-backed defaults + preset merging (WebUI "Default" == TOML base) ---
from typing import Any, Dict, Optional

import cassette_calibrator as cc  # noqa: E402
APP_VERSION = getattr(cc, "__version__", "0.0.0")
from cassette_calibrator import apply_audio_freq_ticks  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

try:
    import tomllib  # py3.11+
except Exception:
    import tomli as tomllib  # pip install tomli  (py<=3.10)


DEFAULT_CONFIG_PATH = Path(
    os.environ.get(
        "CASSETTE_CALIBRATOR_CONFIG",
        str(Path(__file__).with_name("cassette_calibrator.toml")),
    )
)

def _load_toml(path: Path) -> Dict[str, Any]:
    try:
        with path.open("rb") as f:
            return tomllib.load(f) or {}
    except FileNotFoundError:
        return {}

def _deep_get(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def _pick(d: Dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

# Current detector semantics:
# - "thresh" is a RATIO gate (best/others >= thresh), so default is ~6.0, not 0.65.
FALLBACK_DETECT_DEFAULTS = {
    "min_dbfs": -55.0,
    "thresh": 6.0,
    "marker_channel": "mono",
}

def _normalize_marker_channel(v: str) -> str:
    s = str(v).strip().lower()
    if s in ["mix", "sum", "avg", "mean"]:
        return "mono"
    if s in ["mono"]:
        return "mono"
    if s in ["l", "left"]:
        return "L"
    if s in ["r", "right"]:
        return "R"
    return "mono"

def list_detect_presets(cfg: Dict[str, Any]) -> list[str]:
    presets = cfg.get("presets", {})
    if not isinstance(presets, dict):
        return []
    out: list[str] = []
    for name, body in presets.items():
        if not isinstance(body, dict):
            continue
        # Typical structure: [presets.<name>.detect]
        if isinstance(body.get("detect"), dict):
            out.append(str(name))
        else:
            # allow "flat" presets too; we'll treat them as detect keys if used
            out.append(str(name))
    return sorted(set(out))

def get_effective_section(cfg: Dict[str, Any], section: str, preset: Optional[str]) -> Dict[str, Any]:
    base = cfg.get(section, {})
    if not isinstance(base, dict):
        base = {}

    if not preset:
        return dict(base)

    # preferred: [presets.<preset>.<section>]
    p = _deep_get(cfg, "presets", preset, section)

    # fallback/legacy: [presets.<preset>] flat keys (we treat as detect keys only)
    if p is None:
        p = _deep_get(cfg, "presets", preset)
        if not isinstance(p, dict):
            p = {}
        # if it's not nested and section != detect, ignore
        if section != "detect":
            p = {}

    if not isinstance(p, dict):
        p = {}

    merged = dict(base)
    merged.update(p)
    return merged

def get_dtmf_params(config_path: Path = DEFAULT_CONFIG_PATH, preset: Optional[str] = None) -> Dict[str, Any]:
    cfg = _load_toml(config_path)
    detect = get_effective_section(cfg, "detect", preset)

    min_dbfs = float(_pick(detect, ("min_dbfs", "dtmf_min_dbfs", "marker_min_dbfs"),
                           FALLBACK_DETECT_DEFAULTS["min_dbfs"]))
    thresh   = float(_pick(detect, ("thresh", "thresh_ratio", "dtmf_thresh", "dtmf_thresh_ratio"),
                           FALLBACK_DETECT_DEFAULTS["thresh"]))
    ch       = str(_pick(detect, ("marker_channel", "dtmf_channel", "channel"),
                         FALLBACK_DETECT_DEFAULTS["marker_channel"]))

    return {
        "min_dbfs": min_dbfs,
        "thresh": thresh,
        "marker_channel": ch,
        "config_path": str(config_path),
        "preset": preset or "",
    }
# --- end TOML-backed defaults block ---

# set up logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("cassette_calibrator.webui")

# resolve what is our root path and what is allowed for traversal
ROOT = Path(__file__).resolve().parent

WEBUI_ROOT = ROOT
WEBUI_ROOT_REL = "."
WEBUI_RESTRICT_TO_ROOT_DIR = False
WEBUI_ALLOW_PROJECT_ROOT_ACCESS = True

def _webui_is_restricted() -> bool:
    return bool(WEBUI_RESTRICT_TO_ROOT_DIR) and (not bool(WEBUI_ALLOW_PROJECT_ROOT_ACCESS))

def _dedupe_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()

    for p in paths:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)

        if key in seen:
            continue

        seen.add(key)
        out.append(p)

    return out

def _default_browser_root_rel() -> str:
    if _webui_is_restricted():
        return WEBUI_ROOT_REL
    return "data" if (ROOT / "data").exists() else "."

def _list_bases() -> list[Path]:
    if _webui_is_restricted():
        return [WEBUI_ROOT]

    bases: list[Path] = []
    if (ROOT / "data").exists():
        bases.append(ROOT / "data")
    bases.append(ROOT)
    return _dedupe_paths(bases)

def _resolve_webui_root_from_cfg(cfg: dict) -> tuple[Path, str]:
    w = cfg.get("webui", {}) if isinstance(cfg.get("webui"), dict) else {}

    restrict = bool(w.get("restrict_to_root_dir", False))
    allow_project = bool(w.get("allow_project_root_access", True))
    raw_root = str(w.get("root_dir", "data") or "data").strip()

    if (not restrict) or allow_project:
        return ROOT, "."

    if not raw_root:
        raise SystemExit("[webui].root_dir must not be empty")

    if os.path.isabs(raw_root):
        raise SystemExit("[webui].root_dir must be a relative path under the project root")

    parts = Path(raw_root).parts
    if any(part == ".." for part in parts):
        raise SystemExit("[webui].root_dir must not contain '..'")

    full = (ROOT / raw_root).resolve()
    try:
        full.relative_to(ROOT)
    except Exception as e:
        raise SystemExit("[webui].root_dir escapes project root") from e

    full.mkdir(parents=True, exist_ok=True)
    return full, str(full.relative_to(ROOT))

IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
}

MAX_LIST_ITEMS = 4000
# MAX_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1 GiB
MAX_UPLOAD_BYTES = 256 * 1024 * 1024  # 256 MiB
UPLOAD_EXTS = {".wav"}

# what keys to keep and copy during re-generation
REGEN_COPY_KEYS = (
    "fine_align",
    "drift_warn",
    "pre_s",
    "pad_s",
    "noisewin_s",
    "tone_s",
    "tone_hz",
    "f1",
    "f2",
    "sweep_s",
    "ir_win_s",
    "f_plot_min",
    "f_plot_max",
    "win_ms",
    "hop_ms",
    "marker_tone_s",
    "marker_gap_s",
    "marker_dbfs",
    "snr_noise_s",
    "snr_tone_s",
    "smooth_oct",
    "countdown",
    "countdown_from",
    "lr_overlay_color_l",
    "lr_overlay_color_r",
)

_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}

# ----------------
# helper functions
# ----------------

def _browse_root_for_scope(scope: str | None, cfg: dict) -> tuple[Path, str]:
    s = (scope or "").strip().lower()

    if s == "gen_out":
        full, rel = _ensure_gen_out_root_ready(cfg)
        return full, rel

    if _webui_is_restricted():
        return WEBUI_ROOT, WEBUI_ROOT_REL

    return ROOT, "."

def _copy_present(dst: dict, src: dict, keys: tuple[str, ...]) -> None:
    for k in keys:
        v = src.get(k)
        if v is not None:
            dst[k] = v

def _ascii_download_fallback(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name or "download"))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r'[^A-Za-z0-9._ -]+', "_", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s or "download"

def _content_disposition_attachment(name: str) -> str:
    name = str(name or "download")
    fallback = _ascii_download_fallback(name).replace("\\", "_").replace('"', "_")
    quoted = quote(name, safe="")
    return f'attachment; filename="{fallback}"; filename*=UTF-8\'\'{quoted}'

def _safe_upload_filename(name: str, *, allowed_exts: set[str]) -> str:
    raw = str(name or "")
    raw = raw.replace("\\", "/")
    base = Path(raw).name
    base = unicodedata.normalize("NFC", base)

    # Keep Unicode/apostrophes/spaces, but kill path, control and Windows-hostile chars
    base = base.replace("/", "_").replace("\\", "_")
    base = re.sub(r'[\x00-\x1f\x7f]+', "_", base)
    base = re.sub(r'[:*?"<>|]+', "_", base)
    base = re.sub(r"\s+", " ", base).strip().rstrip(". ")

    if not base or base in {".", ".."}:
        raise ValueError("invalid upload filename")

    stem_upper = Path(base).stem.upper()
    if stem_upper in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"reserved filename: {Path(base).stem}")

    ext = Path(base).suffix.lower()
    if ext not in allowed_exts:
        raise ValueError(f"unsupported file type: {ext or '(no extension)'}")

    return base

def _unique_file_path(dir_path: Path, filename: str) -> Path:
    out = dir_path / filename
    if not out.exists():
        return out

    stem = out.stem
    suf = out.suffix
    for i in range(2, 10000):
        cand = dir_path / f"{stem}-{i:02d}{suf}"
        if not cand.exists():
            return cand

    raise RuntimeError("could not allocate unique filename")

def _guess_ctype(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".png":
        return "image/png"
    if suf == ".svg":
        return "image/svg+xml; charset=utf-8"
    if suf == ".csv":
        return "text/csv; charset=utf-8"
    if suf == ".json":
        return "application/json; charset=utf-8"
    if suf == ".wav":
        return "audio/wav"
    return "application/octet-stream"

def _parse_multipart_form(content_type: str, raw: bytes) -> dict:
    if "multipart/form-data" not in (content_type or ""):
        raise ValueError("expected multipart/form-data")

    msg = BytesParser(policy=email_policy).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + raw
    )

    if not msg.is_multipart():
        raise ValueError("invalid multipart body")

    out = {"fields": {}, "files": {}}

    for part in msg.iter_parts():
        name = part.get_param("name", header="Content-Disposition")
        if not name:
            continue

        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""

        if filename is None:
            charset = part.get_content_charset() or "utf-8"
            out["fields"][name] = payload.decode(charset, errors="replace")
        else:
            out["files"][name] = {
                "filename": filename,
                "content_type": part.get_content_type(),
                "data": payload,
            }

    return out

def _validate_uploaded_wav(path: Path) -> dict:
    try:
        sr, y = cc.read_wav(path)
    except Exception as e:
        raise ValueError(f"invalid WAV file: {e}") from e

    if int(sr) <= 0:
        raise ValueError("invalid WAV file: bad sample rate")

    if not hasattr(y, "size") or int(y.size) <= 0:
        raise ValueError("invalid WAV file: no audio samples")

    frames = int(y.shape[0]) if hasattr(y, "shape") and len(y.shape) >= 1 else 0
    channels = 1 if getattr(y, "ndim", 1) == 1 else int(y.shape[1])

    if frames <= 0:
        raise ValueError("invalid WAV file: empty audio")

    dur_s = frames / float(sr)

    return {
        "sample_rate": int(sr),
        "frames": frames,
        "channels": channels,
        "duration_s": dur_s,
    }

def _parse_boolish(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value

    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False

    return default

# --------------------------------------
# generated audio helpers / directories
# --------------------------------------

def _resolve_gen_out_root_from_cfg(cfg: dict) -> tuple[Path, str]:
    w = _webui_cfg(cfg)

    restrict = bool(w.get("restrict_gen_out_to_dir", False))
    raw_root = str(w.get("gen_out_root_dir", "data/gen_test_audio") or "data/gen_test_audio").strip()

    # If not restricted, gen output can live anywhere under the normal WebUI root.
    if not restrict:
        if _webui_is_restricted():
            return WEBUI_ROOT, WEBUI_ROOT_REL
        return ROOT, "."

    if not raw_root:
        raise SystemExit("[webui].gen_out_root_dir must not be empty when restrict_gen_out_to_dir=true")

    if os.path.isabs(raw_root):
        raise SystemExit("[webui].gen_out_root_dir must be a relative path under the project root")

    if any(part == ".." for part in Path(raw_root).parts):
        raise SystemExit("[webui].gen_out_root_dir must not contain '..'")

    full = (ROOT / raw_root).resolve()
    try:
        full.relative_to(ROOT)
    except Exception as e:
        raise SystemExit("[webui].gen_out_root_dir escapes project root") from e

    # If the whole WebUI is already restricted, the gen-out dir must stay inside that root too.
    if _webui_is_restricted():
        try:
            full.relative_to(WEBUI_ROOT)
        except Exception as e:
            raise SystemExit(
                f"[webui].gen_out_root_dir must stay inside the allowed WebUI root: {WEBUI_ROOT_REL}"
            ) from e

    return full, str(full.relative_to(ROOT))

def _ensure_gen_out_root_ready(cfg: dict) -> tuple[Path, str]:
    full, rel = _resolve_gen_out_root_from_cfg(cfg)
    w = _webui_cfg(cfg)

    if bool(w.get("restrict_gen_out_to_dir", False)):
        try:
            full.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise SystemExit(
                f"Could not create [webui].gen_out_root_dir '{rel}': {e}"
            ) from e

    return full, rel

def _gen_out_base(cfg: dict) -> Path:
    full, _ = _ensure_gen_out_root_ready(cfg)
    return full

def _ensure_gen_out_rel(p: str, cfg: dict) -> str:
    rel = _rel_to_root_checked(p)
    full = (ROOT / rel).resolve()

    base = _gen_out_base(cfg)
    try:
        full.relative_to(base)
    except Exception as e:
        raise ValueError(
            f"generated test WAV path must stay under: {base.relative_to(ROOT)}"
        ) from e

    if full.suffix.lower() != ".wav":
        raise ValueError("generated test WAV must use .wav extension")

    full.parent.mkdir(parents=True, exist_ok=True)
    return str(full.relative_to(ROOT))

def _webui_warn_on_overwrite(cfg: dict) -> bool:
    w = _webui_cfg(cfg)
    return bool(w.get("warn_on_overwrite", True))

def _timestamped_nonconflicting_rel(rel_path: str, *, cfg: dict) -> str:
    rel = _ensure_gen_out_rel(rel_path, cfg)
    full = (ROOT / rel).resolve()

    parent = full.parent
    stem = full.stem
    suf = full.suffix
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    cand = parent / f"{stem}-{ts}{suf}"
    if not cand.exists():
        return str(cand.relative_to(ROOT))

    for i in range(2, 10000):
        cand = parent / f"{stem}-{ts}-{i:02d}{suf}"
        if not cand.exists():
            return str(cand.relative_to(ROOT))

    raise RuntimeError("could not allocate timestamped output filename")

# -----------------------
# DTMF / marker presets (WebUI-side)
# -----------------------

def _dtmf_from_detect_section(detect: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a detect section into WebUI DTMF preset fields.
    Works with current ratio-style thresh (best/others >= thresh).
    """
    if not isinstance(detect, dict):
        detect = {}

    min_dbfs = float(
        _pick(
            detect,
            ("min_dbfs", "dtmf_min_dbfs", "marker_min_dbfs"),
            FALLBACK_DETECT_DEFAULTS["min_dbfs"],
        )
    )
    thresh = float(
        _pick(
            detect,
            ("thresh", "thresh_ratio", "dtmf_thresh", "dtmf_thresh_ratio"),
            FALLBACK_DETECT_DEFAULTS["thresh"],
        )
    )

    ch_raw = str(
        _pick(
            detect,
            ("marker_channel", "dtmf_channel", "channel"),
            FALLBACK_DETECT_DEFAULTS["marker_channel"],
        )
    )
    ch = _normalize_marker_channel(ch_raw)  # returns "mono"/"L"/"R"

    return {"min_dbfs": min_dbfs, "thresh": thresh, "marker_channel": ch}

def _build_dtmf_presets_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build DTMF presets:
      - "default" from top-level [detect] (plus FALLBACK_DETECT_DEFAULTS)
      - "noisy_cassette", "line_in", "aggressive" derived from default
      - plus any [presets.<name>.detect] presets from the TOML
    """
    # base/default from config
    base_detect = get_effective_section(cfg, "detect", preset=None)
    dflt = _dtmf_from_detect_section(base_detect)

    presets: Dict[str, Dict[str, Any]] = {
        "default": dict(dflt),

        # derived helpers (ratio thresh: smaller = more permissive)
        "noisy_cassette": {
            "min_dbfs": dflt["min_dbfs"] - 5.0,
            "thresh": max(1.0, dflt["thresh"] - 2.0),
            "marker_channel": dflt["marker_channel"],
        },
        "line_in": {
            "min_dbfs": dflt["min_dbfs"] + 10.0,
            "thresh": dflt["thresh"] + 2.0,
            "marker_channel": dflt["marker_channel"],
        },
        "aggressive": {
            "min_dbfs": dflt["min_dbfs"] - 10.0,
            "thresh": max(1.0, dflt["thresh"] - 3.0),
            "marker_channel": dflt["marker_channel"],
        },
    }

    # overlay TOML presets: [presets.<name>.detect]
    for name in list_detect_presets(cfg):
        det = get_effective_section(cfg, "detect", preset=name)
        if isinstance(det, dict):
            presets[str(name)] = _dtmf_from_detect_section(det)

    return presets

# Safe fallback so the module never crashes even if main() changes later.
DTMF_PRESETS_FALLBACK: Dict[str, Dict[str, Any]] = {
    "default": {
        "min_dbfs": float(FALLBACK_DETECT_DEFAULTS["min_dbfs"]),
        "thresh": float(FALLBACK_DETECT_DEFAULTS["thresh"]),
        "marker_channel": str(FALLBACK_DETECT_DEFAULTS["marker_channel"]),
    },
    "noisy_cassette": {
        "min_dbfs": float(FALLBACK_DETECT_DEFAULTS["min_dbfs"]) - 5.0,
        "thresh": max(1.0, float(FALLBACK_DETECT_DEFAULTS["thresh"]) - 2.0),
        "marker_channel": str(FALLBACK_DETECT_DEFAULTS["marker_channel"]),
    },
    "line_in": {
        "min_dbfs": float(FALLBACK_DETECT_DEFAULTS["min_dbfs"]) + 10.0,
        "thresh": float(FALLBACK_DETECT_DEFAULTS["thresh"]) + 2.0,
        "marker_channel": str(FALLBACK_DETECT_DEFAULTS["marker_channel"]),
    },
    "aggressive": {
        "min_dbfs": float(FALLBACK_DETECT_DEFAULTS["min_dbfs"]) - 10.0,
        "thresh": max(1.0, float(FALLBACK_DETECT_DEFAULTS["thresh"]) - 3.0),
        "marker_channel": str(FALLBACK_DETECT_DEFAULTS["marker_channel"]),
    },
}

DTMF_PRESETS: Dict[str, Dict[str, Any]] = dict(DTMF_PRESETS_FALLBACK)

# Map our generic keys to possible argparse dest names in cassette_calibrator
DTMF_ARG_ALIASES = {
    "min_dbfs": ("min_dbfs",),
    "thresh": ("thresh",),
    "marker_channel": ("marker_channel",),
}

def _coerce_float(v, *, default=None):
    if v is None or v == "":
        return default
    try:
        x = float(v)
    except Exception:
        return default
    if not math.isfinite(x):
        return default
    return x

def _coerce_str(v, *, default=None):
    s = _opt_str(v)
    return s if s is not None else default

def _apply_dtmf_cfg(overrides: dict, dfl: dict, cfg: dict) -> None:
    """
    Apply cfg entries to overrides, but ONLY if the argparse dest exists.
    Prevents WebUI from inventing unknown args.
    """
    for k, v in (cfg or {}).items():
        if v is None:
            continue
        for dest in DTMF_ARG_ALIASES.get(k, (k,)):
            if dest in dfl:
                overrides[dest] = v
                break

def _is_marker_failure(err: str | None, log_txt: str | None) -> bool:
    s = ((err or "") + "\n" + (log_txt or "")).lower()
    # Match your known failure ("Could not find start/end markers...") and nearby phrasing
    return ("marker" in s) or ("dtmf" in s) or ("start/end" in s) or ("start/end markers" in s)

def _dtmf_candidate_configs(base_cfg: dict, preset_name: str) -> list[tuple[str, dict]]:
    """
    Returns a small ladder of candidate configs for auto-tuning.
    """
    preset = DTMF_PRESETS.get(preset_name) or DTMF_PRESETS["default"]

    cfg0 = dict(preset)
    # payload overrides win over preset
    for k, v in (base_cfg or {}).items():
        if v is not None:
            cfg0[k] = v

    cands: list[tuple[str, dict]] = []
    cands.append(("as_requested", cfg0))

    # relax min_dbfs (more sensitive)
    for delta in (-5.0, -10.0, -15.0):
        c = dict(cfg0)
        if "min_dbfs" in c and isinstance(c["min_dbfs"], (int, float)):
            c["min_dbfs"] = float(c["min_dbfs"]) + float(delta)
            cands.append((f"min_dbfs{delta:+.0f}", c))

    # relax thresh (ratio gate): smaller = more permissive
    for delta in (-1.0, -2.0, -3.0):
        c = dict(cfg0)
        if "thresh" in c and isinstance(c["thresh"], (int, float)):
            c["thresh"] = max(1.0, float(c["thresh"]) + float(delta))
            cands.append((f"thresh{delta:+.0f}", c))

    # combo step
    c = dict(cfg0)
    if isinstance(c.get("min_dbfs"), (int, float)):
        c["min_dbfs"] = float(c["min_dbfs"]) - 10.0
    if isinstance(c.get("thresh"), (int, float)):
        c["thresh"] = max(1.0, float(c["thresh"]) - 2.0)
    cands.append(("combo_-10dbfs_-2thresh", c))

    # last resort: aggressive preset
    if preset_name != "aggressive":
        aggr = DTMF_PRESETS.get("aggressive") or DTMF_PRESETS.get("default") or {}
        c = dict(aggr)
        # keep chosen channel if provided
        if "marker_channel" in cfg0:
            c["marker_channel"] = cfg0["marker_channel"]
        cands.append(("aggressive_preset", c))

    # dedupe
    seen = set()
    out = []
    for label, cfg in cands:
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append((label, cfg))
    return out

# -----------------------
# HELPERS
# -----------------------

# regen for reanalysis
def _require_existing_rel_wav(value, *, label: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{label} path is missing from saved run metadata")

    rel = _rel_to_root_checked(raw)
    full = ROOT / rel

    if not full.exists():
        raise FileNotFoundError(
            f"{label} not found: {rel}\n"
            "This saved run can only be regenerated if the original source WAV "
            "still exists under the project directory."
        )

    if not full.is_file():
        raise FileNotFoundError(f"{label} is not a file: {rel}")

    if full.suffix.lower() != ".wav":
        raise ValueError(f"{label} is not a WAV file: {rel}")

    return rel

def _channels_mode_from_summary(summary: dict) -> str:
    chs = summary.get("channels_analyzed", [])
    norm = [_norm_ch_name(str(c)) for c in chs]

    if norm == ["mono"]:
        return "mono"
    if norm == ["L"]:
        return "l"
    if norm == ["R"]:
        return "r"
    if "L" in norm and "R" in norm:
        return "stereo"

    return "stereo"

def _summary_has_impulse(summary: dict) -> bool:
    per = summary.get("per_channel", {})
    if not isinstance(per, dict):
        return False

    for item in per.values():
        if not isinstance(item, dict):
            continue
        outs = item.get("outputs", {})
        if isinstance(outs, dict) and outs.get("impulse_png"):
            return True

    return False

def _api_analyze_from_payload(payload: dict, cfg: dict) -> dict:
    ref = _require_existing_rel_wav(payload.get("ref"), label="Reference WAV")
    rec = _require_existing_rel_wav(payload.get("rec"), label="Recorded WAV")

    loopback = None
    loopback_raw = _opt_str(payload.get("loopback"))
    if loopback_raw:
        loopback = _require_existing_rel_wav(loopback_raw, label="Loopback WAV")

    form_defaults = _build_form_defaults(cfg)
    outdir_raw = _payload_str_or_default(payload, "outdir", form_defaults["an_outdir"])
    outdir = _ensure_outdir_rel(outdir_raw)

    overrides = {
        "ref": ref,
        "rec": rec,
        "outdir": outdir,
    }

    if loopback:
        overrides["loopback"] = loopback

    dfl = _cmd_argparse_defaults("analyze")

    def _set_first_dest(value, *dest_names: str) -> None:
        for dn in dest_names:
            if dn in dfl:
                overrides[dn] = value
                return

    def _maybe_str(api_key: str, *dest_names: str, normalizer=None) -> None:
        if api_key not in payload:
            return
        s = _opt_str(payload.get(api_key))
        if s is None:
            return
        if normalizer is not None:
            s = normalizer(s)
        _set_first_dest(s, *dest_names)

    def _maybe_bool(api_key: str, *dest_names: str) -> None:
        if api_key not in payload:
            return
        _set_first_dest(bool(payload.get(api_key)), *dest_names)

    def _maybe_float(api_key: str, *dest_names: str) -> None:
        if api_key not in payload:
            return
        fv = _coerce_float(payload.get(api_key), default=None)
        if fv is None:
            return
        _set_first_dest(fv, *dest_names)

    def _maybe_int(api_key: str, *dest_names: str) -> None:
        if api_key not in payload:
            return
        v = payload.get(api_key)
        if v is None or v == "":
            return
        try:
            iv = int(v)
        except Exception:
            return
        _set_first_dest(iv, *dest_names)

    _maybe_str("run_name", "run_name", "name")
    _maybe_str("run_notes", "run_notes", "notes")
    _maybe_str("channels", "channels")
    _maybe_str("marker_channel", "marker_channel", normalizer=lambda s: _normalize_marker_channel(str(s)))
    _maybe_str("marker_start", "marker_start")
    _maybe_str("marker_end", "marker_end")
    _maybe_str("tick_sym", "tick_sym")
    _maybe_str("lr_overlay_color_l", "lr_overlay_color_l")
    _maybe_str("lr_overlay_color_r", "lr_overlay_color_r")

    _maybe_bool("lock_y_axis", "lock_y_axis")
    _maybe_bool("fine_align", "fine_align")
    _maybe_bool("lr_overlay", "lr_overlay")
    _maybe_bool("save_ir", "save_ir")
    _maybe_bool("countdown", "countdown")
    _maybe_bool("ticks", "ticks")

    _maybe_float("plot_y_min", "plot_y_min")
    _maybe_float("plot_y_max", "plot_y_max")
    _maybe_float("min_dbfs", "min_dbfs")
    _maybe_float("thresh", "thresh")
    _maybe_float("drift_warn", "drift_warn")
    _maybe_float("pre_s", "pre_s")
    _maybe_float("pad_s", "pad_s")
    _maybe_float("noisewin_s", "noisewin_s")
    _maybe_float("tone_s", "tone_s")
    _maybe_float("tone_hz", "tone_hz")
    _maybe_float("f1", "f1")
    _maybe_float("f2", "f2")
    _maybe_float("sweep_s", "sweep_s")
    _maybe_float("ir_win_s", "ir_win_s")
    _maybe_float("f_plot_min", "f_plot_min")
    _maybe_float("f_plot_max", "f_plot_max")
    _maybe_float("win_ms", "win_ms")
    _maybe_float("hop_ms", "hop_ms")
    _maybe_float("marker_tone_s", "marker_tone_s")
    _maybe_float("marker_gap_s", "marker_gap_s")
    _maybe_float("marker_dbfs", "marker_dbfs")
    _maybe_float("snr_noise_s", "snr_noise_s")
    _maybe_float("snr_tone_s", "snr_tone_s")
    _maybe_float("tick_interval_s", "tick_interval_s")
    _maybe_float("tick_tone_s", "tick_tone_s")
    _maybe_float("tick_dbfs", "tick_dbfs")
    _maybe_float("tick_offset_s", "tick_offset_s")
    _maybe_float("tick_match_tol_s", "tick_match_tol_s")

    _maybe_int("smooth_oct", "smooth_oct")
    _maybe_int("countdown_from", "countdown_from")
    _maybe_int("tick_min_matches", "tick_min_matches")

    dtmf_in = payload.get("dtmf", None)
    dtmf_obj = dtmf_in if isinstance(dtmf_in, dict) else {}

    dtmf_preset = _coerce_str(dtmf_obj.get("preset"), default="default") or "default"
    dtmf_autotune = bool(dtmf_obj.get("autotune", True))

    base_dtmf_cfg = {
        "min_dbfs": _coerce_float(dtmf_obj.get("min_dbfs"), default=None),
        "thresh": _coerce_float(dtmf_obj.get("thresh"), default=None),
        "marker_channel": _coerce_str(dtmf_obj.get("marker_channel"), default=None),
    }

    if base_dtmf_cfg.get("marker_channel") is not None:
        base_dtmf_cfg["marker_channel"] = _normalize_marker_channel(base_dtmf_cfg["marker_channel"])

    replay_payload = dict(overrides)

    replay_dtmf = {
        "preset": dtmf_preset,
        "autotune": dtmf_autotune,
    }
    for k, v in base_dtmf_cfg.items():
        if v is not None:
            replay_dtmf[k] = v

    LOG.info("analyze: ref=%s rec=%s loopback=%s outdir=%s", ref, rec, loopback, outdir)

    attempts = []
    used_label = None
    used_cfg = None
    used_args_payload = None

    if dtmf_autotune:
        cand_list = _dtmf_candidate_configs(base_dtmf_cfg, dtmf_preset)
    else:
        preset = DTMF_PRESETS.get(dtmf_preset) or DTMF_PRESETS["default"]
        cfg0 = dict(preset)
        for k, v in base_dtmf_cfg.items():
            if v is not None:
                cfg0[k] = v
        cand_list = [("as_requested", cfg0)]

    ok = False
    log_txt = ""
    err = None

    for label, dtmf_cfg in cand_list:
        ov = dict(overrides)
        _apply_dtmf_cfg(ov, dfl, dtmf_cfg)

        args = _make_args("analyze", cfg, ov)
        ok, log_txt, err = _run_cc_cmd(cc.cmd_analyze, args)

        attempts.append({
            "label": label,
            "dtmf_cfg": dtmf_cfg,
            "ok": bool(ok),
            "error": err,
        })

        if ok:
            used_label = label
            used_cfg = dtmf_cfg
            used_args_payload = dict(vars(args))            
            break

        if not dtmf_autotune:
            break

        if not _is_marker_failure(err, log_txt):
            break

    if not ok:
        raise ValueError(
            json.dumps({
                "ok": False,
                "error": err,
                "log": log_txt,
                "dtmf": {
                    "preset": dtmf_preset,
                    "autotune": dtmf_autotune,
                    "attempts": attempts,
                },
            }, ensure_ascii=False)
        )

    base = ROOT / outdir
    candidates = list(base.rglob("summary.json")) if base.exists() else []
    if not candidates:
        raise RuntimeError("analyze succeeded but summary.json was not created")

    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    summary_path = candidates[0]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    plot_cfg = _summary_plot_cfg(summary, cfg)
    summary["_webui_plot_cfg"] = plot_cfg

    # Save the actual effective replay payload so regen can reuse the original analysis settings.
    # Use the successful effective args, not just the incoming WebUI overrides.
    replay_payload = dict(used_args_payload or overrides)

    # Prefer the actually-used DTMF config if autotune settled on something specific.
    if used_cfg is not None:
        replay_payload["dtmf"] = {
            "preset": dtmf_preset,
            "autotune": False,
            **{k: v for k, v in used_cfg.items() if v is not None},
        }
    else:
        replay_payload["dtmf"] = replay_dtmf

    summary["_webui_replay_payload"] = replay_payload
    _write_json_atomic_preserve_mtime(summary_path, summary)

    return {
        "ok": True,
        "summary": summary,
        "log": log_txt,
        "dtmf": {
            "preset": dtmf_preset,
            "used": used_label,
            "cfg": used_cfg,
            "attempts": attempts,
        },
    }

# fallback for legacy versions
def _summary_plot_cfg(summary: dict, cfg: dict) -> dict:
    plot = summary.get("plot", {}) if isinstance(summary, dict) else {}
    if not isinstance(plot, dict):
        plot = {}

    webui_cfg = cfg.get("webui", {}) if isinstance(cfg.get("webui"), dict) else {}
    use_cfg_fallback = bool(webui_cfg.get("legacy_run_plot_fallback_from_config", True))

    form_defaults = _build_form_defaults(cfg)

    if use_cfg_fallback:
        default_lock = bool(form_defaults.get("an_lock_y_axis", True))
        default_ymin = float(form_defaults.get("an_plot_y_min", -35.0))
        default_ymax = float(form_defaults.get("an_plot_y_max", 5.0))
    else:
        default_lock = True
        default_ymin = -35.0
        default_ymax = 5.0

    return {
        "lock_y_axis": bool(plot.get("lock_y_axis", default_lock)),
        "plot_y_min": float(plot.get("plot_y_min", default_ymin)),
        "plot_y_max": float(plot.get("plot_y_max", default_ymax)),
    }

def _payload_str_or_default(payload: dict, key: str, default: str) -> str:
    v = payload.get(key, None)
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default

def _opt_str(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        if s.lower() in ("null", "none"):
            return None
        return s
    s = str(v).strip()
    return s or None

def _is_rel_safe(p: str) -> bool:
    if p is None:
        return False
    p = str(p).strip()
    if p == "":
        return False
    if os.path.isabs(p):
        return False
    parts = Path(p).parts
    if any(part == ".." for part in parts):
        return False
    return True

def _rel_to_project_root_checked(p: str) -> str:
    p = (p or "").strip()
    if not _is_rel_safe(p):
        raise ValueError("path must be relative (no absolute paths, no '..')")

    full = (ROOT / p).resolve()
    try:
        full.relative_to(ROOT)
    except Exception as e:
        raise ValueError("path escapes project root") from e

    return str(full.relative_to(ROOT))

def _rel_to_root_checked(p: str) -> str:
    rel = _rel_to_project_root_checked(p)
    full = (ROOT / rel).resolve()
    base = WEBUI_ROOT if _webui_is_restricted() else ROOT

    try:
        full.relative_to(base)
    except Exception as e:
        if _webui_is_restricted():
            raise ValueError(f"path is outside the allowed WebUI root: {WEBUI_ROOT_REL}")
        raise ValueError("path escapes project root") from e

    return rel

def _ensure_outdir_rel(p: str) -> str:
    p = (p or "").strip()
    if not p:
        raise ValueError("outdir is required")

    p = _rel_to_root_checked(p)
    (ROOT / p).mkdir(parents=True, exist_ok=True)
    return p

def _load_cfg(cfg_path: str | None) -> dict:
    # cc.load_toml_config(None) searches relative to *cwd*.
    # WebUI should default to the TOML next to this file, so launching from
    # elsewhere (systemd/launcher/etc) still uses the same config as CLI.
    if not cfg_path:
        cfg_path = str(DEFAULT_CONFIG_PATH)
    return cc.load_toml_config(cfg_path)

def _webui_cfg(cfg: dict) -> dict:
    w = cfg.get("webui", {})
    return w if isinstance(w, dict) else {}

def _build_form_defaults(cfg: dict) -> dict:
    gen_defaults = dict(_cmd_argparse_defaults("gen"))
    analyze_defaults = dict(_cmd_argparse_defaults("analyze"))

    gen_cfg = cfg.get("gen", {}) if isinstance(cfg.get("gen"), dict) else {}
    analyze_cfg = cfg.get("analyze", {}) if isinstance(cfg.get("analyze"), dict) else {}

    try:
        gen_defaults.update(cc.flatten_cmd_defaults("gen", gen_cfg))
    except Exception:
        gen_defaults.update(gen_cfg)

    try:
        analyze_defaults.update(cc.flatten_cmd_defaults("analyze", analyze_cfg))
    except Exception:
        analyze_defaults.update(analyze_cfg)

    gen_out = str(gen_defaults.get("out") or "").strip() or "data/test_audio.wav"

    try:
        gen_out = _ensure_gen_out_rel(gen_out, cfg)
    except Exception:
        gen_root, _ = _ensure_gen_out_root_ready(cfg)
        gen_out = str((gen_root / "test_audio.wav").relative_to(ROOT))

    analyze_outdir = str(analyze_defaults.get("outdir") or "")

    return {
        "gen_out": gen_out,
        "an_ref": gen_out,
        "an_outdir": analyze_outdir,
        "an_lock_y_axis": bool(analyze_defaults.get("lock_y_axis", True)),
        "an_plot_y_min": float(analyze_defaults.get("plot_y_min", -35.0)),
        "an_plot_y_max": float(analyze_defaults.get("plot_y_max", 5.0)),
    }

def _webui_timestamped_gen_out_cfg(cfg: dict) -> dict:
    """
    WebUI-only generation filename suggestion config.

    Keeps CLI/TOML [gen].out as the seed path, but lets the WebUI
    suggest a timestamped filename by default so users do not
    accidentally overwrite an older reference WAV.
    """
    form_defaults = _build_form_defaults(cfg)
    seed = str(form_defaults.get("gen_out") or "data/test_audio.wav").strip() or "data/test_audio.wav"

    w = _webui_cfg(cfg)

    return {
        "enabled": bool(w.get("propose_timestamped_test_audio_default", True)),
        "seed": seed,
    }

# -------------------------------------------
# read the environment (for url prefixes etc)
# -------------------------------------------

def _load_simple_env(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        txt = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return out

    for raw_line in txt.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()

        if not k:
            continue

        if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
            v = v[1:-1]

        out[k] = v

    return out

def _normalize_url_prefix(v: str | None) -> str:
    s = (v or "").strip()
    if not s or s == "/":
        return ""
    if not s.startswith("/"):
        s = "/" + s
    return s.rstrip("/")

def _strip_optional_prefix(path: str, prefix: str) -> str:
    """
    Accept both prefixed and unprefixed paths.
    This lets local direct access keep working even if a reverse proxy uses a subpath.
    """
    if not prefix:
        return path

    if path == prefix:
        return "/"
    if path.startswith(prefix + "/"):
        out = path[len(prefix):]
        return out if out else "/"

    return path

def _prefix_url(prefix: str, path: str) -> str:
    """
    Join a normalized prefix with an absolute app path like '/api/runs'.
    """
    if not path.startswith("/"):
        raise ValueError("path must start with '/'")
    return f"{prefix}{path}" if prefix else path

def _run_cc_cmd(fn, *args, **kwargs) -> tuple[bool, str, str | None]:
    """
    Run a cassette_calibrator cmd_* function, capturing stdout/stderr.
    Returns: (ok, log_text, error_message_or_None)

    We treat SystemExit as a normal "user-facing" failure (400),
    because cassette_calibrator cmd_* functions use SystemExit for CLI.
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            fn(*args, **kwargs)
        return True, buf.getvalue(), None
    except SystemExit as e:
        # SystemExit.code can be int/None/str; str(e) usually contains your message.
        msg = str(e).strip()
        if not msg:
            code = getattr(e, "code", None)
            msg = str(code).strip() if code is not None else "command failed"
        return False, buf.getvalue(), msg

@lru_cache(maxsize=None)
def _cmd_argparse_defaults(cmd: str) -> dict:
    """
    Extract argparse defaults for a specific subcommand from cassette_calibrator.build_parser().
    This keeps WebUI in sync when new CLI flags are added.
    """
    _ap, cmd_parsers = cc.build_parser()
    p = cmd_parsers.get(cmd)
    if p is None:
        return {}

    out: dict = {}
    for a in getattr(p, "_actions", []):
        dest = getattr(a, "dest", None)
        if not dest or dest == "help":
            continue

        default = getattr(a, "default", None)

        # Skip argparse.SUPPRESS
        if default is argparse.SUPPRESS:
            continue

        out[dest] = default

    return out

def _make_args(cmd: str, cfg: dict, overrides: dict | None) -> SimpleNamespace:
    # 1) Start from real argparse defaults for this command
    args = dict(_cmd_argparse_defaults(cmd))

    # 2) Override with TOML section defaults
    if isinstance(cfg, dict):
        args.update(cc.flatten_cmd_defaults(cmd, cfg.get(cmd, {})))

    # 3) Override with request payload (API caller)
    if overrides:
        args.update(overrides)

    # Normalizations (because we bypass argparse parsing)
    if cmd == "analyze" and "channels" in args and isinstance(args["channels"], str):
        args["channels"] = cc.parse_channels(args["channels"])

    if cmd == "detect" and "channel" in args:
        args["channel"] = _normalize_marker_channel(str(args["channel"]))

    if cmd == "analyze" and "marker_channel" in args:
        args["marker_channel"] = _normalize_marker_channel(str(args["marker_channel"]))

    return SimpleNamespace(**args)

def _walk_pruned(base: Path):
    # Prune IGNORE_DIRS during traversal
    for root, dirs, files in os.walk(base):
        root_p = Path(root)
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        yield root_p, dirs, files

def _list_files(exts: tuple[str, ...], max_items: int = 2000) -> list[str]:
    out: list[str] = []

    for base in _list_bases():
        if not base.exists():
            continue

        for root_p, _dirs, files in _walk_pruned(base):
            for fn in files:
                p = root_p / fn
                if p.suffix.lower() in exts:
                    try:
                        rel = str(p.relative_to(ROOT))
                    except Exception:
                        continue

                    out.append(rel)
                    if len(out) >= max_items:
                        break

            if len(out) >= max_items:
                break

        if len(out) >= max_items:
            break

    return sorted(set(out))

def _list_result_dirs(max_items: int = 2000) -> list[str]:
    out: list[str] = []

    for base in _list_bases():
        if not base.exists():
            continue

        for p in base.rglob("summary.json"):
            try:
                rel = str(p.parent.relative_to(ROOT))
            except Exception:
                continue

            out.append(rel)
            if len(out) >= max_items:
                break

        if len(out) >= max_items:
            break

    return sorted(set(out))

def _list_runs(max_items: int = 500) -> list[dict]:
    runs: list[dict] = []
    seen = set()

    for base in _list_bases():
        if not base.exists():
            continue

        for p in base.rglob("summary.json"):
            try:
                run_dir = str(p.parent.relative_to(ROOT))
            except Exception:
                continue

            if run_dir in seen:
                continue
            seen.add(run_dir)

            try:
                st = p.stat()
                mtime = int(st.st_mtime)
            except Exception:
                mtime = 0

            name = None
            created = None
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    run = obj.get("run", {})
                    if isinstance(run, dict):
                        name = _opt_str(run.get("name"))
                        created = _opt_str(run.get("created_at_local")) or _opt_str(run.get("created_at_utc"))
            except Exception:
                pass

            label = (f"{name} -- {run_dir}" if name else run_dir)

            runs.append({
                "dir": run_dir,
                "mtime": mtime,
                "name": name,
                "created_at": created,
                "label": label,
            })

            if len(runs) >= max_items:
                break

        if len(runs) >= max_items:
            break

    runs.sort(key=lambda x: x.get("mtime", 0), reverse=True)
    return runs

def _browse_dir(
    rel_dir: str,
    *,
    mode: str,
    exts: list[str] | None,
    q: str | None,
    scope: str | None,
    cfg: dict,
) -> dict:
    rel_dir = (rel_dir or "").strip()
    browse_root, browse_root_rel = _browse_root_for_scope(scope, cfg)

    if rel_dir == "":
        rel_dir = browse_root_rel

    if rel_dir == "." and browse_root_rel != ".":
        rel_dir = browse_root_rel

    if rel_dir == ".":
        rel_checked = "."
        full = ROOT
    else:
        rel_checked = _rel_to_root_checked(rel_dir)
        full = (ROOT / rel_checked).resolve()

    try:
        full.relative_to(browse_root)
    except Exception as e:
        raise ValueError(f"path is outside the allowed browser root: {browse_root_rel}") from e

    if not full.exists() or not full.is_dir():
        raise ValueError(f"not a directory: {rel_dir}")

    exts_norm: list[str] = []
    if exts:
        for e in exts:
            e = (e or "").strip().lower()
            if e and not e.startswith("."):
                e = "." + e
            if e:
                exts_norm.append(e)
    exts_norm = sorted(set(exts_norm))

    qn = (q or "").strip().lower()

    entries = []
    count = 0

    for p in sorted(full.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        name = p.name
        if name in IGNORE_DIRS:
            continue

        try:
            relp = str(p.relative_to(ROOT))
        except Exception:
            continue

        if qn:
            if qn not in name.lower() and qn not in relp.lower():
                continue

        if p.is_dir():
            st = p.stat()
            entries.append({
                "name": name,
                "path": relp,
                "is_dir": True,
                "size": None,
                "mtime": int(st.st_mtime),
            })
            count += 1
        else:
            if mode != "file":
                continue
            if exts_norm and p.suffix.lower() not in exts_norm:
                continue

            st = p.stat()
            entries.append({
                "name": name,
                "path": relp,
                "is_dir": False,
                "size": int(st.st_size),
                "mtime": int(st.st_mtime),
            })
            count += 1

        if count >= MAX_LIST_ITEMS:
            break

    parent = None
    if full != browse_root:
        parent = str(full.parent.relative_to(ROOT))

    return {
        "cwd": rel_checked if rel_checked != "." else ".",
        "parent": parent,
        "root_dir": browse_root_rel,
        "at_root": (full == browse_root),
        "mode": mode,
        "exts": exts_norm,
        "q": q or "",
        "entries": entries,
        "now": int(time.time()),
    }

def _write_json_atomic_preserve_mtime(path: Path, obj: object) -> None:
    old_times = None
    try:
        st = path.stat()
        old_times = (st.st_atime, st.st_mtime)
    except Exception:
        old_times = None

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)

    # Keep mtime so the run list ordering doesn't change just because notes were edited
    if old_times:
        try:
            os.utime(path, old_times)
        except Exception:
            pass

# -----------------------
# COMPARE (runs -> high-res grid PNG)
# -----------------------

_COMPARE_METRICS = {
    # metric -> (per_channel output key prefix, xlabel, ylabel, logx default)
    "response":   ("response",   "Frequency (Hz)", "dB", True),
    "difference": ("difference", "Frequency (Hz)", "dB", True),
    "impulse":    ("impulse",    "Samples",        "Amplitude", False),
}

def _slug(s: str) -> str:
    s = (s or "").strip()
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "x"

def _load_summary_for_run_dir(rel_run_dir: str, cfg: dict | None = None) -> dict:
    rel_run_dir = _rel_to_root_checked(rel_run_dir)
    p = (ROOT / rel_run_dir / "summary.json")
    if not p.exists() or not p.is_file():
        raise ValueError(f"summary.json not found for run: {rel_run_dir}")

    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"summary.json is not an object for run: {rel_run_dir}")

    plot_cfg = _summary_plot_cfg(obj, cfg or {})
    obj["_webui_plot_cfg"] = plot_cfg
    return obj

def _run_label(rel_run_dir: str, summary: dict) -> str:
    name = None
    try:
        run = summary.get("run", {})
        if isinstance(run, dict):
            name = _opt_str(run.get("name"))
    except Exception:
        name = None
    return f"{name} -- {rel_run_dir}" if name else rel_run_dir

def _norm_ch_name(ch: str) -> str:
    s = (ch or "").strip()
    if not s:
        return s
    u = s.upper()
    if u in ("LEFT", "L"):
        return "L"
    if u in ("RIGHT", "R"):
        return "R"
    if u in ("MONO", "MIX", "SUM"):
        return "mono"
    return s

def _get_channel_outputs(summary: dict, ch: str) -> dict:
    per = summary.get("per_channel", {})
    if not isinstance(per, dict):
        return {}
    # try exact, then normalized matching
    if ch in per and isinstance(per.get(ch), dict):
        return per[ch].get("outputs", {}) if isinstance(per[ch].get("outputs", {}), dict) else {}
    # fallback: match normalized names
    want = _norm_ch_name(ch)
    for k, v in per.items():
        if _norm_ch_name(str(k)) == want and isinstance(v, dict):
            outs = v.get("outputs", {})
            return outs if isinstance(outs, dict) else {}
    return {}

def _find_metric_paths(summary: dict, metric_prefix: str, ch: str) -> tuple[str | None, str | None]:
    """
    Return (csv_rel, png_rel) for the metric+channel if available in summary.
    We look for keys like:
      response_csv / response.png variants / response_png
    """
    outs = _get_channel_outputs(summary, ch)
    if not outs:
        return (None, None)

    # common naming patterns
    csv_keys = [
        f"{metric_prefix}_csv",
        f"{metric_prefix}.csv",
        f"{metric_prefix}_out_csv",
    ]
    png_keys = [
        f"{metric_prefix}_png",
        f"{metric_prefix}.png",
        f"{metric_prefix}_out_png",
    ]

    csv_rel = None
    for k in csv_keys:
        v = outs.get(k)
        if isinstance(v, str) and v.strip():
            csv_rel = v.strip()
            break

    png_rel = None
    for k in png_keys:
        v = outs.get(k)
        if isinstance(v, str) and v.strip():
            png_rel = v.strip()
            break

    return (csv_rel, png_rel)

def _sniff_csv_dialect(sample: str):
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t")
    except Exception:
        return csv.get_dialect("excel")

def _read_xy_csv(rel_path: str) -> tuple[list[float], list[float]]:
    """
    Very forgiving CSV reader.
    - Supports comma/semicolon/tab
    - Header-aware: tries to find x/y columns by name; else uses first 2 numeric cols
    """
    rel_path = _rel_to_root_checked(rel_path)
    full = (ROOT / rel_path)
    if not full.exists() or not full.is_file():
        raise ValueError(f"CSV not found: {rel_path}")

    txt = full.read_text(encoding="utf-8", errors="replace")
    sample = txt[:4096]
    dialect = _sniff_csv_dialect(sample)
    reader = csv.reader(io.StringIO(txt), dialect)

    rows = []
    for r in reader:
        if not r:
            continue
        # strip whitespace
        r2 = [c.strip() for c in r]
        if len(r2) == 0:
            continue
        rows.append(r2)

    if not rows:
        return ([], [])

    # detect header: any non-float in first row -> treat as header
    def _is_float(s: str) -> bool:
        try:
            float(s)
            return True
        except Exception:
            return False

    header = None
    data_start = 0
    if any((c and not _is_float(c)) for c in rows[0]):
        header = [c.strip().lower() for c in rows[0]]
        data_start = 1

    xi, yi = 0, 1

    if header:
        # heuristic column selection
        def find_idx(preds: list[str]) -> int | None:
            for p in preds:
                for i, h in enumerate(header or []):
                    if p in h:
                        return i
            return None

        x_try = find_idx(["freq", "hz", "f "])
        y_try = find_idx(["db", "mag", "amp", "value", "y"])
        if x_try is not None:
            xi = x_try
        if y_try is not None and y_try != xi:
            yi = y_try
        else:
            # if y not found, choose first column != x that looks numeric in data
            yi = 1 if xi == 0 else 0

    xs: list[float] = []
    ys: list[float] = []

    for r in rows[data_start:]:
        if xi >= len(r) or yi >= len(r):
            continue
        try:
            x = float(r[xi])
            y = float(r[yi])
        except Exception:
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        xs.append(x)
        ys.append(y)

    return (xs, ys)

def _compare_render_grid(
    *,
    runs: list[dict],
    channels: list[str],
    metric: str,
    outdir: str,
    tile_w: int,
    tile_h: int,
    dpi: int,
    lock_y_axis: bool = True,
    plot_y_min: float | None = None,
    plot_y_max: float | None = None,
) -> tuple[str, str]:
    """
    Returns (out_png_rel, log_text).

    If CSVs exist for the metric, re-plot with shared axes.
    If not, tile PNGs (axes matching not guaranteed) and log a warning.

    Note: If some cells lack CSV but have PNG, we will still show the PNG,
    but it won't share axes with the plotted cells.
    """
    metric = (metric or "").strip().lower()
    if metric not in _COMPARE_METRICS:
        raise ValueError(f"unknown metric: {metric}")

    metric_prefix, xlabel, ylabel, logx_default = _COMPARE_METRICS[metric]

    log_lines: list[str] = []
    series: dict[tuple[str, int], tuple[list[float], list[float]]] = {}
    tiles: dict[tuple[str, int], str] = {}
    have_any_series = False

    # Collect series and/or image tiles
    for j, r in enumerate(runs):
        rel_dir = r["dir"]
        summ = r["summary"]

        for ch in channels:
            csv_rel, png_rel = _find_metric_paths(summ, metric_prefix, ch)

            if csv_rel:
                try:
                    xs, ys = _read_xy_csv(csv_rel)
                    if xs and ys:
                        series[(ch, j)] = (xs, ys)
                        have_any_series = True
                        continue
                    else:
                        log_lines.append(f"[warn] empty CSV for {rel_dir} ch={ch}: {csv_rel}")
                except Exception as e:
                    log_lines.append(f"[warn] failed reading CSV for {rel_dir} ch={ch}: {csv_rel} -- {e}")

            if png_rel:
                tiles[(ch, j)] = png_rel
                if have_any_series:
                    log_lines.append(f"[warn] using PNG fallback (no shared axes) for {rel_dir} ch={ch}: {png_rel}")
            else:
                log_lines.append(f"[warn] missing outputs for {rel_dir} ch={ch} metric={metric_prefix}")

    # Output path
    outdir_rel = _ensure_outdir_rel(outdir)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    ch_tag = _slug("_".join(channels))
    out_name = f"compare-{metric}-{ch_tag}-{ts}.png"
    out_png_rel = str(Path(outdir_rel) / out_name)
    out_png_full = ROOT / out_png_rel

    nrows = max(1, len(channels))
    ncols = max(1, len(runs))

    fig_w_in = (tile_w * ncols) / max(1, dpi)
    fig_h_in = (tile_h * nrows) / max(1, dpi)

    import matplotlib.pyplot as plt  # Agg backend already set
    from matplotlib.image import imread as _imread

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w_in, fig_h_in),
        dpi=dpi,
        sharex=True if have_any_series else False,
        sharey=True if have_any_series else False,
    )

    # Normalize axes to 2D
    if nrows == 1 and ncols == 1:
        axes2 = [[axes]]
    elif nrows == 1:
        axes2 = [list(axes)]
    elif ncols == 1:
        axes2 = [[ax] for ax in axes]
    else:
        axes2 = [list(row) for row in axes]

    # Compute global limits from series (only)
    x_min = x_max = y_min = y_max = None
    if have_any_series:
        xs_all: list[float] = []
        ys_all: list[float] = []

        for (_ch, _j), (xs, ys) in series.items():
            for x, y in zip(xs, ys):
                if logx_default and x <= 0:
                    continue
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                xs_all.append(x)
                ys_all.append(y)

        if xs_all:
            x_min = min(xs_all)
            x_max = max(xs_all)
            if x_min == x_max:
                x_min *= 0.9
                x_max *= 1.1

        if lock_y_axis:
            if plot_y_min is None or plot_y_max is None:
                raise ValueError("plot_y_min/plot_y_max are required when lock_y_axis is enabled")

            y0 = float(plot_y_min)
            y1 = float(plot_y_max)

            if not (math.isfinite(y0) and math.isfinite(y1)):
                raise ValueError("plot_y_min/plot_y_max must be finite")
            if y0 >= y1:
                raise ValueError("plot_y_min must be smaller than plot_y_max")

            y_min = y0
            y_max = y1
            log_lines.append(f"[info] compare: locked y-axis enabled ({y_min:.2f} .. {y_max:.2f} dB)")

        elif ys_all:
            y_min = min(ys_all)
            y_max = max(ys_all)
            if y_min == y_max:
                y_min -= 1.0
                y_max += 1.0
            log_lines.append(f"[info] compare: auto y-axis from data ({y_min:.2f} .. {y_max:.2f})")

    # Helpers: row/col labels
    def _short_run_label(r: dict) -> str:
        # try run name; else directory
        summ = r.get("summary", {})
        name = None
        try:
            run = summ.get("run", {})
            if isinstance(run, dict):
                name = _opt_str(run.get("name"))
        except Exception:
            name = None
        return name or r.get("dir", "")

    # Draw each cell
    for i, ch in enumerate(channels):
        for j, r in enumerate(runs):
            ax = axes2[i][j]

            # Column titles (top row)
            if i == 0:
                ax.set_title(_short_run_label(r), fontsize=9)

            key = (ch, j)

            if have_any_series and key in series:
                xs, ys = series[key]

                # filter non-positive x if log
                if logx_default:
                    xs2 = []
                    ys2 = []
                    for x, y in zip(xs, ys):
                        if x > 0 and math.isfinite(x) and math.isfinite(y):
                            xs2.append(x)
                            ys2.append(y)
                    xs, ys = xs2, ys2

                if xs and ys:
                    ax.plot(xs, ys, linewidth=1.1)
                else:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

                if logx_default:
                    ax.set_xscale("log")

                    # Force “audio” freq ticks/labels (20..20k, 1K/2K/5K/10K/20K)
                    fmin = float(x_min) if x_min is not None else 20.0
                    fmax = float(x_max) if x_max is not None else 20000.0
                    apply_audio_freq_ticks(ax, fmin, fmax)
                    # ax.xaxis.set_minor_formatter(NullFormatter())  # optional: hide minor tick labels

                if x_min is not None and x_max is not None:
                    ax.set_xlim(x_min, x_max)
                if y_min is not None and y_max is not None:
                    ax.set_ylim(y_min, y_max)

                ax.grid(True, alpha=0.25)
                ax.tick_params(labelsize=8)

                # Axis labels only on left/bottom edges
                if j == 0:
                    ax.set_ylabel(f"{ch} — {ylabel}", fontsize=8)
                else:
                    ax.set_ylabel("")
                if i == nrows - 1:
                    ax.set_xlabel(xlabel, fontsize=8)
                else:
                    ax.set_xlabel("")

            elif key in tiles:
                # PNG tile fallback
                try:
                    rel_png = _rel_to_root_checked(tiles[key])
                    img = _imread(str(ROOT / rel_png))
                    ax.imshow(img)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # keep some indication of row label on left-most column
                    if j == 0:
                        ax.text(
                            0.01, 0.99, f"{ch}",
                            transform=ax.transAxes,
                            ha="left", va="top",
                            fontsize=10,
                            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                        )
                except Exception as e:
                    ax.text(0.5, 0.5, f"failed to load png\n{e}", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()

    # Layout: keep your “tile geometry” mostly intact
    fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.08, wspace=0.18, hspace=0.28)

    out_png_full.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png_full, format="png")
    plt.close(fig)

    if have_any_series:
        log_lines.insert(0, "[info] compare: re-plotted from CSV (shared axes where possible)")
    else:
        log_lines.insert(0, "[info] compare: tiled PNGs (no shared axes)")

    return out_png_rel, ("\n".join(log_lines).strip() + "\n")

INDEX_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>cassette-calibrator 🖭 WebUI v__APP_VERSION__</title>
  <style>

  :root {
    --bg-top: #cfd8e3;
    --bg-bottom: #e4ebf2;
    --page-text: #1b2430;

    --card-bg: #f7fafc;
    --card-border: #b7c3cf;
    --card-shadow: 0 10px 28px rgba(24, 35, 52, 0.10);

    --footer-bg: linear-gradient(180deg, rgba(255,255,255,0.30) 0%, rgba(223,234,246,0.55) 100%);
    --footer-border: #b7c3cf;
    --footer-text: #425364;
    --footer-link: #355571;
    --footer-shadow: 0 8px 24px rgba(24, 35, 52, 0.10);
    
    --panel-bg: #eef3f8;
    --panel-border: #c7d2dd;

    --input-bg: #ffffff;
    --input-border: #aebccc;
    --input-focus: #4b6f93;

    --muted: #5f6c79;
    --muted-2: #738191;

    --accent: #44698d;
    --accent-hover: #355571;
    --accent-soft: #dce7f2;

    --log-bg: #0f141b;
    --log-fg: #d7dde5;
    --log-border: #273240;

    --img-border: #2a3440;

    --modal-bg: #f7fafc;
    --modal-border: #b7c3cf;

    --viewer-bg: #f5f8fb;
    --viewer-border: #b7c3cf;
  }

  html {
    min-height: 100%;
    background:
      linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
  }

  body {
    font-family: system-ui, sans-serif;
    margin: 20px;
    min-height: calc(100vh - 40px);
    color: var(--page-text);
    background: transparent;
  }

  h2, h3, h4 {
    color: var(--page-text);
  }

  code {
    background: rgba(255,255,255,0.35);
    padding: 1px 5px;
    border-radius: 6px;
  }

  hr {
    border: 0;
    border-top: 1px solid var(--panel-border);
    margin: 16px 0;
  }

  .row {
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
    align-items: flex-start;
  }

   .card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 14px;
    min-width: min(340px, 100%);
    max-width: 560px;
    flex: 1 1 340px;
    min-width: 0;
    box-shadow: var(--card-shadow);
    backdrop-filter: blur(2px);
    }

  label {
    display: block;
    margin-top: 8px;
    font-size: 13px;
    color: var(--page-text);
  }

  input,
  select,
  textarea {
    width: 100%;
    padding: 8px 10px;
    margin-top: 4px;
    box-sizing: border-box;
    border: 1px solid var(--input-border);
    border-radius: 9px;
    background: var(--input-bg);
    color: var(--page-text);
  }

  input:focus,
  select:focus,
  textarea:focus {
    outline: none;
    border-color: var(--input-focus);
    box-shadow: 0 0 0 3px rgba(75, 111, 147, 0.16);
  }

  details {
    margin-top: 8px;
  }

  summary {
    cursor: pointer;
    color: var(--accent);
    font-weight: 600;
  }

  button {
    margin-top: 10px;
    padding: 8px 12px;
    cursor: pointer;
    border: 1px solid #96a8ba;
    border-radius: 9px;
    background: var(--accent-soft);
    color: var(--page-text);
    font-weight: 600;
    transition: background 0.12s ease, border-color 0.12s ease, transform 0.06s ease;
  }

  button:hover {
    background: #cfddec;
    border-color: var(--accent);
  }

  button:active {
    transform: translateY(1px);
  }

  pre {
    background: var(--log-bg);
    color: var(--log-fg);
    padding: 10px;
    border-radius: 10px;
    overflow: auto;
    max-height: 260px;
    border: 1px solid var(--log-border);

    /* make long errors readable */
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    word-break: break-word;

    /* let the user expand the log box */
    resize: vertical;
    min-height: 80px;
  }

  img {
    max-width: 100%;
    border-radius: 8px;
    border: 1px solid var(--img-border);
  }

    .small {
    font-size: 12px;
    color: #000;
    }

  .grid2 {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 8px;
    align-items: end;
  }

  /* Modal file browser */
  .modal-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(8, 12, 18, 0.55);
    display: none;
    z-index: 9999;
  }

  .modal {
    position: fixed;
    inset: 40px;
    background: var(--modal-bg);
    border: 1px solid var(--modal-border);
    border-radius: 14px;
    padding: 12px;
    display: none;
    z-index: 10000;
    box-shadow: 0 14px 48px rgba(0,0,0,0.30);
  }

  .modal-header {
    display: flex;
    gap: 10px;
    align-items: center;
  }

  .modal-header .path {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    color: var(--muted);
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .modal-body {
    margin-top: 10px;
    display: grid;
    grid-template-columns: 1fr;
    gap: 8px;
    height: calc(100% - 52px);
  }

  .modal-controls {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .modal-controls input {
    margin-top: 0;
  }

  .modal-list {
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    overflow: auto;
    padding: 8px;
    height: 100%;
    background: #fbfdff;
  }

  .item {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto auto;
  gap: 10px;
  padding: 6px 8px;
  border-radius: 8px;
  align-items: center;
  }
  
  .item > div:first-child {
    min-width: 0;
  }
  
  .item .mono {
    display: block;
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: break-word;
  }
  
  .item > div:last-child {
    white-space: nowrap;
  }

  .item:hover {
    background: #eaf1f8;
  }

  .mono {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
  }

  .pill {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 999px;
    border: 1px solid var(--panel-border);
    color: var(--page-text);
    background: #f1f5f9;
  }

  .muted {
    color: var(--muted-2);
    font-size: 12px;
  }

  .runhdr {
    margin-top: 10px;
    margin-bottom: 6px;
  }

  .runhdr .title {
    font-size: 22px;
    font-weight: 700;
    margin: 0 0 4px 0;
  }

  .runhdr .meta {
    font-size: 12px;
    color: var(--muted);
    line-height: 1.35;
  }

  .runhdr .mono {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }

  textarea {
    width: 100%;
    padding: 8px 10px;
    margin-top: 4px;
    resize: vertical;
  }

  .toc {
    margin-top: 6px;
    font-size: 12px;
    color: var(--muted);
  }

  .toc a {
    color: var(--accent);
    text-decoration: none;
    border-bottom: 1px dotted #89a0b7;
  }

  .toc a:hover {
    border-bottom-style: solid;
  }

  .block {
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 10px;
    margin-top: 10px;
    background: var(--panel-bg);
  }

  .block h4 {
    margin: 0 0 8px 0;
  }

  .notes {
    white-space: pre-wrap;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    background: #f6f9fc;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid var(--panel-border);
  }

  .imgsec {
    margin-top: 14px;
    padding-top: 10px;
    border-top: 1px dashed #b9c6d3;
  }

  .imgsec h4 {
    margin: 0 0 8px 0;
  }

  /* ---- Image viewer / clickable images ---- */

  .imgwrap {
    margin-top: 10px;
  }

  .imgmeta {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    align-items: center;
  }

  .imglinks {
    display: flex;
    gap: 10px;
    align-items: center;
  }

  .imglink {
    font-size: 12px;
    color: var(--accent);
    text-decoration: none;
    border-bottom: 1px dotted #89a0b7;
  }

  .imglink:hover {
    border-bottom-style: solid;
  }

  .imgbtn {
    padding: 4px 8px;
    font-size: 12px;
    margin-top: 0;
  }

  .imgthumb {
    cursor: zoom-in;
  }

  /* Fullscreen-ish image viewer */
  .iv-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.75);
    display: none;
    z-index: 20000;
  }

  .iv {
    position: fixed;
    inset: 26px;
    background: var(--viewer-bg);
    border: 1px solid var(--viewer-border);
    border-radius: 12px;
    padding: 10px;
    display: none;
    z-index: 20001;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
  }

  .iv-head {
    display: flex;
    gap: 10px;
    align-items: center;
  }

  .iv-title {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-weight: 600;
    font-size: 13px;
  }

  .iv-actions {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .iv-meta {
    margin-top: 6px;
    font-size: 12px;
    color: var(--muted);
  }

  .iv-body {
    margin-top: 10px;
    height: calc(100% - 56px);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: auto;
    background: #111;
    border-radius: 10px;
    padding: 10px;
  }

  .iv-body img {
    max-width: 100%;
    max-height: 100%;
    border-radius: 8px;
    border: 1px solid #333;
    cursor: zoom-in;
  }

  /* “Actual” (1:1) mode: image can be larger than viewport; you pan via scrolling */
  .iv-body img.iv-actual {
    max-width: none;
    max-height: none;
    cursor: grab;
  }

  .hero {
  margin: 0 0 18px 0;
  padding: 16px 18px 14px 18px;
  border: 1px solid rgba(255,255,255,0.45);
  border-radius: 16px;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.30) 0%, rgba(255,255,255,0.12) 100%),
    linear-gradient(135deg, rgba(78, 120, 164, 0.20) 0%, rgba(255,255,255,0.10) 100%);
  box-shadow:
    0 10px 30px rgba(24, 35, 52, 0.14),
    inset 0 1px 0 rgba(255,255,255,0.45);
  backdrop-filter: blur(4px);
    }

    .hero-title {
    margin: 0;
    font-size: clamp(30px, 4vw, 44px);
    line-height: 1.06;
    font-weight: 900;
    letter-spacing: -0.03em;
    color: #f8fbff;
    text-shadow:
        0 1px 0 rgba(255,255,255,0.55),
        0 0 10px rgba(122, 170, 220, 0.30),
        0 0 22px rgba(122, 170, 220, 0.18),
        0 3px 10px rgba(28, 44, 66, 0.35);
    }

    .hero-title .brand-main {
    color: #ffffff;
    }

    .hero-title .brand-accent {
    color: #355571;
    text-shadow:
        0 1px 0 rgba(255,255,255,0.40),
        0 0 12px rgba(110, 170, 235, 0.22),
        0 2px 8px rgba(28, 44, 66, 0.28);
    }

    .hero-badge {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1px;

    margin-left: 10px;
    padding: 5px 10px 6px 10px;
    vertical-align: middle;

    font-size: 0.38em;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    line-height: 1.0;

    color: #24415c;
    background: linear-gradient(180deg, #f7fbff 0%, #dfeaf6 100%);
    border: 1px solid rgba(86, 113, 141, 0.35);
    border-radius: 999px;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.8),
        0 4px 12px rgba(40, 60, 84, 0.16);
    }

    .hero-badge-label {
    display: block;
    }

    .hero-badge-version {
    display: block;
    font-size: 0.82em;
    font-weight: 700;
    letter-spacing: 0.02em;
    opacity: 0.92;
    text-transform: none;
    }

    .hero-sub {
    margin-top: 8px;
    font-size: 13px;
    color: #3f5368;
    }

    .hero-sub code {
    background: rgba(255,255,255,0.55);
    }

   .page-footer {
    margin-top: 26px;
    padding: 14px 18px;
    border: 1px solid var(--footer-border);
    border-radius: 14px;
    background: var(--footer-bg);
    color: var(--footer-text);
    box-shadow: var(--footer-shadow);
    font-size: 12px;
  }

  .page-footer .footer-main {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 8px 16px;
  }

  .page-footer .footer-title {
    font-weight: 700;
    color: var(--page-text);
  }

  .page-footer .footer-meta {
    margin-top: 6px;
    color: var(--footer-text);
    line-height: 1.5;
  }

  .page-footer a {
    color: var(--footer-link);
    text-decoration: none;
    border-bottom: 1px dotted rgba(53, 85, 113, 0.55);
  }

  .page-footer a:hover {
    border-bottom-style: solid;
  }

.cassette-mark {
  height: 1.45em;
  width: auto;
  margin-left: 0.18em;
  vertical-align: -0.12em;

  border: none;
  border-radius: 0;
  background: transparent;
  box-shadow: none;
  display: inline-block;

  filter:
    drop-shadow(0 1px 0 rgba(255,255,255,0.35))
    drop-shadow(0 2px 6px rgba(28,44,66,0.22));
}

.op-summary,
.detect-summary {
  margin-top: 10px;
  border-radius: 10px;
  padding: 12px;
  border: 1px solid var(--panel-border);
  background: var(--panel-bg);
  box-sizing: border-box;
  overflow-x: hidden;
}

.op-summary.ok,
.detect-summary.ok {
  border-color: #2e7d32;
  background: #eaf7ec;
}

.op-summary.fail,
.detect-summary.fail {
  border-color: #b71c1c;
  background: #fdecec;
}

.op-summary.warn,
.detect-summary.warn {
  border-color: #b26a00;
  background: #fff4df;
}

.op-summary h4,
.detect-summary h4 {
  margin: 0 0 8px 0;
}

.op-summary .status-line,
.detect-summary .status-line {
  font-weight: 700;
  margin-bottom: 8px;
}

.op-summary .status-ok,
.detect-summary .status-ok {
  color: #1b5e20;
}

.op-summary .status-fail,
.detect-summary .status-fail {
  color: #8b0000;
}

.op-summary .status-warn,
.detect-summary .status-warn {
  color: #8a5300;
}

.op-summary dl,
.detect-summary dl {
  margin: 0;
  display: grid;
  grid-template-columns: max-content minmax(0, 1fr);
  gap: 6px 12px;
  align-items: start;
}

.op-summary dt,
.detect-summary dt,
.op-summary dd,
.detect-summary dd {
  min-width: 0;
}

.op-summary dt,
.detect-summary dt {
  font-weight: 600;
}

.op-summary dd,
.detect-summary dd {
  margin: 0;
  overflow-wrap: anywhere;
  word-break: break-word;
}

.file-path {
  display: block;
  min-width: 0;
  white-space: normal;
  overflow-wrap: anywhere;
  word-break: break-word;
}

.op-summary code,
.op-summary .mono,
.detect-summary code,
.detect-summary .mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  white-space: normal;
  overflow-wrap: anywhere;
  word-break: break-word;
}

.op-summary .muted,
.detect-summary .muted {
  overflow-wrap: anywhere;
  word-break: break-word;
}

.op-summary .sum-section {
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px solid var(--panel-border);
}

.op-summary .sum-section:first-of-type {
  margin-top: 0;
  padding-top: 0;
  border-top: 0;
}

.op-summary .sum-section h5 {
  margin: 0 0 8px 0;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--muted);
}

.op-summary .sum-help,
.detect-summary .sum-help {
  margin-top: 2px;
  font-size: 11px;
  line-height: 1.3;
}

@media (max-width: 560px) {
  .op-summary dl,
  .detect-summary dl {
    grid-template-columns: 1fr;
    gap: 2px 0;
  }

  .op-summary dt,
  .detect-summary dt {
    margin-top: 8px;
  }

  .op-summary dt:first-child,
  .detect-summary dt:first-child {
    margin-top: 0;
  }
}

.mb-status {
  display: none;
  margin-top: 8px;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid var(--panel-border);
  background: var(--panel-bg);
  font-size: 12px;
  line-height: 1.4;
}

.mb-status.show {
  display: block;
}

.mb-status.error {
  border-color: #b71c1c;
  background: #fdecec;
  color: #8b0000;
}

.mb-status.warn {
  border-color: #b26a00;
  background: #fff4df;
  color: #8a5300;
}

.mb-status.info {
  border-color: #2f5f8a;
  background: #eaf2fb;
  color: #24415c;
}

.ow-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(8, 12, 18, 0.72);
  display: none;
  z-index: 30000;
}

.ow-modal {
  position: fixed;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  width: min(680px, calc(100vw - 32px));
  background: var(--modal-bg);
  border: 1px solid var(--modal-border);
  border-radius: 14px;
  padding: 16px;
  display: none;
  z-index: 30001;
  box-shadow: 0 18px 56px rgba(0,0,0,0.42);
}

.ow-path {
  margin-top: 6px;
  padding: 8px 10px;
  border: 1px solid var(--panel-border);
  border-radius: 8px;
  background: #f6f9fc;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  overflow-wrap: anywhere;
  word-break: break-word;
}

.ow-actions {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
  flex-wrap: wrap;
  margin-top: 14px;
}

.ow-actions button {
  margin-top: 0;
}

.ow-btn-danger {
  background: #f8dfdf;
  border-color: #d18d8d;
}

.ow-btn-danger:hover {
  background: #f3d0d0;
  border-color: #b96e6e;
}

  </style>
</head>
<body>

    <div class="hero">
    <h1 class="hero-title">
        <span class="brand-main">cassette</span><span class="brand-accent">-calibrator</span>
        <img class="cassette-mark" src="__CASSETTE_LOGO_URL__" alt="" aria-hidden="true" />
        <span class="hero-badge">
            <span class="hero-badge-label">WEBUI</span>
            <span class="hero-badge-version">v__APP_VERSION__</span>
        </span>
    </h1>
    <div class="hero-sub">
    <i>Find out how tapehead-brained your signal path really is.</i>
    </div>
    </div>

    <div class="small">
    <p>This frontend uses <code>cassette_calibrator.toml</code> configuration defaults.</p>
    </div>

  <div class="row">
       <div class="card">
        <h3>1. Generate test audio</h3>
        <div class="small">
            <p>Creates the reference calibration WAV: DTMF markers, silence window, reference tone and ESS sweep.</p>
            <p><b>ℹ️ Use this functionality as your first step to generate the reference audio file that you will then i.e. record to your cassette/run through your signal chain and record it back from the other end.</b></p>
            <p>The test audio file only needs to be generated once, unless you need to change your audio test pattern itself. Changing the type is currently not supported via this webUI and is only available in the CLI version of this program.</p>
            <p>If you're running the same test repeatedly for A/B comparison (and esp. if you need to compare runs), ALWAYS use the same WAV that is generated here.</p>
            <p>After generating the audio, save it to your computer and use it i.e. with a separate DAW program, audio player, etc to pass it through your signal chain and record it back for analysis.</b></p>
        </div>

        <p></p>
        
        <div class="small">
        <p>The generated test audio file can be downloaded via the webUI after it has been generated. 
        Change the path/filename only if needed (i.e. if you're doing different comparison types).</p>
        
        <p><b>⚠️ NOTE: generated test audio files stay in the webUI's data folder, so in most cases you only need to run this step once.</b></p>
        </div>       

        <p></p>

        <label>Generated test WAV location if local (relative path)</label>
            
      <div class="grid2">
        <input id="gen_out" value="__GEN_OUT_DEFAULT__" />
        <button onclick="openBrowser('gen_out', {
        mode:'file',
        exts:['.wav'],
        title:'Choose output WAV',
        allowNew:true,
        allowUpload:false,
        scope:'gen_out'
        })">Browse</button>
      </div>
        <button onclick="doGen()">Generate</button>
        <div id="gen_file"></div>
        <pre id="gen_log"></pre>
        <div id="gen_summary"></div>
    </div>

    <div class="card">
        <h3>2. Detect start/stop marker sequences from audio</h3>
        <div class="small">
            <p>Scans a WAV audio file for start/end DTMF marker positions.</p>
            <p><b>ℹ️ This step is primarily intended as a fail-safe on your recorded audio file(s) to verify that they contain both the required start and stop DTMF marker sequences for synchronization.</b></p>
            <p><b>⚠️ NOTE: if the required start and stop markers are NOT found during the marker detection phase, you (likely) cannot proceed with the analysis and should re-check your recorded audio file.</b></p>
            <p>Running this step before the main analysis (3.) is highly recommended.</p>            
        </div>
        <label>Input WAV (click "Browse" to select file)</label>      
      <div class="grid2">
        <input id="detect_wav" placeholder="data/recorded.wav" />
        <button onclick="openBrowser('detect_wav', {mode:'file', exts:['.wav'], title:'Choose WAV'})">Browse</button>
      </div>
      <button onclick="doDetect()">Detect markers</button>
      <pre id="detect_out"></pre>
      <div id="detect_summary"></div>

    </div>

    <div class="card">
    <h3>3. Analyze the audio; A/B comparison</h3>
    
    <div class="small">
    <p>Compares the generated reference WAV against recorded playback and writes plots, CSVs and <code>summary.json</code>.</p>

    <p><b>ℹ️ This is the main analysis step.</b> You are comparing the original reference test WAV from step 1 against the recorded WAV that came back through your signal path.</p>
    <p><b>Important:</b> this software does <i>not</i> measure "just the cassette" in isolation unless your whole setup is designed for that on purpose. In normal use, it measures the <b>entire signal chain</b>: your playback device/interface, D/A conversion, cables, connectors, deck input stage, record level, tape path, playback stage, output stage, return cabling, A/D conversion, and any noise or level problems added along the way.</p>
    <p>In other words: <b>what goes in vs. what comes back out</b>. The result is the sum of the whole path, not just one part of it.</p>
    <p>Typical example: the reference WAV leaves your audio interface at a certain output level, gets converted from digital to analog, travels through your cabling into the deck, gets recorded at some chosen record level, and is then captured back to the computer either directly from the deck electronics or from actual tape playback. Every one of those stages can change the result.</p>
    <p>⚠️ <b><i>Tip:</i> write down your output levels, record levels, deck settings, tape type, NR/Dolby state, and anything else relevant in the run notes for every test. If you change settings and do not document them, you are basically sabotaging your own comparison data.</b></p>
    <p>With analog gear, the game is not "can degradation be avoided completely?" -- it usually can't. The real question is <b>how much</b> the signal changes, <b>where</b> it changes, and <b>which part of the chain is doing the damage</b>.</p>
    <p>That means looking at things like frequency response changes, noise floor / SNR, channel imbalance, stereo mismatch, phase behaviour, distortion, and any loss of high-frequency content. Some loss may come from the tape and deck itself, but bad cables, bad grounding, poor gain staging, cheap converters, extra adapters, and unnecessary signal steps can also make things worse.</p>
    <p>So start simple: use the cleanest possible setup, the best practical cabling, sensible levels, and as few extra stages as possible. Then test changes one at a time. If your setup is already a spaghetti nightmare, your measurements will faithfully report that nightmare back to you.</p>
    <p>⛓️ A signal chain is only as good as its weakest link... and all the other crap right next to it.</p>
    <p><b>Note:</b> playback of the test WAV and recording of the return signal are handled outside this WebUI. You can use a DAW, or any other reliable audio playback/recording software. However, <b>make sure</b> that the said setup does <b>not</b> resample, add effects, apply "enhancements", or mix in system sounds.</p>
    </div>
      
      <label>🎢 <b>Run name</b> (optional)</label>
      <input id="an_name" placeholder="e.g.: this cassette => that deck" />

      <label>🗒 <b>Run notes</b> (optional)</label>
      <textarea id="an_notes" rows="7" placeholder="Long notes: deck, tape, settings, azimuth tweaks, Dolby/NR, weirdness..."></textarea>
      
      <label>🅰️ <b>Reference WAV</b> (= original test reference audio)</label>
      <div class="grid2">
        <input id="an_ref" value="__AN_REF_DEFAULT__" />
        <button onclick="openBrowser('an_ref', {mode:'file', exts:['.wav'], title:'Choose reference WAV'})">Browse</button>
      </div>

      <label>🅱️ <b>Recorded WAV</b> (= your recording of the original test reference audio)</label>
      <div class="grid2">
        <input id="an_rec" placeholder="data/recorded.wav" />
        <button onclick="openBrowser('an_rec', {mode:'file', exts:['.wav'], title:'Choose recorded WAV to use in comparison against the reference'})">Browse</button>
      </div>

        <div class="block">
        <h4>Marker detection tuning (DTMF)</h4>
        <div class="small">
            Controls how aggressively the start/end marker finder listens for DTMF in noisy recordings.
        </div>        

        <div class="row" style="align-items:flex-end;">
          <div style="flex:1; min-width:220px;">
            <label>Preset</label>
            <select id="dtmf_preset">
              <option value="default">Default</option>
              <option value="noisy_cassette">Noisy cassette</option>
              <option value="line_in">Line-in</option>
              <option value="aggressive">Aggressive</option>
            </select>
          </div>

          <div style="min-width:200px;">
            <label style="display:flex; gap:10px; align-items:center; margin-top:8px;">
              <input id="dtmf_autotune" type="checkbox" checked style="width:auto; margin:0;" />
              Auto-tune on failure
            </label>
          </div>
        </div>

        <details>
          <summary>Advanced</summary>

          <div class="row" style="margin-top:8px;">
            <div style="flex:1; min-width:180px;">
              <label>min_dbfs</label>
              <input id="dtmf_min_dbfs" type="number" step="0.5" value="-55.0" />
              <div class="small">More negative = more sensitive.</div>
            </div>

            <div style="flex:1; min-width:180px;">
              <label>thresh</label>
              <input id="dtmf_thresh" type="number" step="0.01" value="6.0" />
              <div class="small">Lower = more permissive.</div>
            </div>

            <div style="flex:1; min-width:180px;">
              <label>marker_channel</label>
                <select id="dtmf_marker_channel">
                  <option value="mono">mix (L+R)</option>
                  <option value="L">L</option>
                  <option value="R">R</option>
                </select>
            </div>
          </div>
        </details>
      </div>
      
      <label>Loopback WAV (optional; not used in most cases; leave empty by default)</label>
      <div class="grid2">
        <input id="an_lb" placeholder="(none)" />
        <button onclick="openBrowser('an_lb', {mode:'file', exts:['.wav'], title:'Choose loopback WAV'})">Browse</button>
      </div>

      <div class="block">
        <h4>Response plot y-axis</h4>
        <div class="small">
          Locks the y-axis for response plots and the stereo L/R overlay plot, so comparisons do not lie by autoscaling.
        </div>

        <label style="display:flex; gap:10px; align-items:center; margin-top:10px;">
          <input id="an_lock_y_axis" type="checkbox" __AN_LOCK_Y_AXIS_CHECKED__ style="width:auto; margin:0;" />
          Lock y-axis
        </label>

        <div class="row" style="align-items:flex-end; margin-top:8px;">
          <div style="flex:1; min-width:160px;">
            <label>Min dB</label>
            <input id="an_plot_y_min" type="number" step="0.5" value="__AN_PLOT_Y_MIN_DEFAULT__" />
          </div>

          <div style="flex:1; min-width:160px;">
            <label>Max dB</label>
            <input id="an_plot_y_max" type="number" step="0.5" value="__AN_PLOT_Y_MAX_DEFAULT__" />
          </div>
        </div>
      </div>      

      <label>Output directory (relative path; don't change unless for some specific purpose)</label>
      <div class="grid2">
        <input id="an_outdir" value="__AN_OUTDIR_DEFAULT__" />
        <button onclick="openBrowser('an_outdir', {mode:'dir', title:'Choose or create output directory', allowNew:true})">Browse</button>
      </div>

      <hr></hr>
      
      <div class="small">
      <p>Click on the "Analyze" button below when you are ready to start.</p>
      <p>Note that the analysis may take some time.</p>
      </div>      

      <button onclick="doAnalyze()">Analyze</button>

      <div id="an_header"></div>

      <div id="an_json">
        <pre id="an_log"></pre>
      </div>

      <div id="an_imgs"></div>

    </div>
  </div>

    <div class="card">
    <h3>4. View previously saved analysis runs</h3>
    <div class="small">
        Browse previous analysis outputs, inspect notes, images and summary.json, and edit stored notes.
    </div>

    <button onclick="refreshRuns()">Refresh list</button>
    <label>Pick a run</label>
    <select id="runs_sel" style="width:100%; padding:6px; margin-top:4px;">
    <option value="">-- pick a run --</option>
    </select>

    <button onclick="loadSelectedRun()">Load run</button>
    <div id="runs_header"></div>

    <div class="block">
      <h4>Regenerate selected run</h4>
      <div class="small">
        Re-runs the selected analysis using the saved source WAV paths from summary.json.
        A new regenerated run will be created, with "-- regen" appended to the run name.
      </div>

      <label style="display:flex; gap:10px; align-items:center; margin-top:10px;">
        <input id="runs_lock_y_axis" type="checkbox" checked style="width:auto; margin:0;" />
        Lock y-axis
      </label>

      <div class="row" style="align-items:flex-end; margin-top:8px;">
        <div style="flex:1; min-width:160px;">
          <label>Min dB</label>
          <input id="runs_plot_y_min" type="number" step="0.5" value="__AN_PLOT_Y_MIN_DEFAULT__" />
        </div>

        <div style="flex:1; min-width:160px;">
          <label>Max dB</label>
          <input id="runs_plot_y_max" type="number" step="0.5" value="__AN_PLOT_Y_MAX_DEFAULT__" />
        </div>
      </div>

      <button onclick="rerunSelectedRun()">Regenerate run</button>
    </div>
    
    <div style="margin-top:8px;">
        <button id="runs_notes_edit_btn" onclick="runsEditNotes()" style="display:none">Edit notes</button>
    </div>

    <div id="runs_notes_ctl" class="block" style="display:none">
        <h4>Edit notes</h4>
        <textarea id="runs_notes_text" rows="7"
        placeholder="Deck, tape, settings, azimuth tweaks, Dolby/NR, weirdness..."></textarea>
        <div style="display:flex; gap:8px; margin-top:8px;">
        <button onclick="runsSaveNotes()">Save</button>
        <button onclick="runsCancelNotes()">Cancel</button>
        </div>
        <div id="runs_notes_status" class="small"></div>
    </div>

    <div id="runs_json">
        <pre id="runs_log"></pre>
    </div>

    <div id="runs_imgs"></div>
    </div>

  <div class="card">
  <h3>5. Compare between previous analysis runs</h3>
  <div class="small">
    Builds one comparison figure from multiple saved runs. Uses CSV data when available so axes stay aligned.
  </div>

      <button onclick="refreshCompareRuns()">Refresh runs</button>

      <div class="row" style="margin-top:8px;">
        <div style="flex:1; min-width:260px;">
          <label>Available runs</label>
          <select id="cmp_avail" multiple size="10" style="width:100%; padding:6px; margin-top:4px;"></select>
          <div style="display:flex; gap:8px; margin-top:8px;">
            <button onclick="cmpAdd()">Add →</button>
          </div>
        </div>

        <div style="flex:1; min-width:260px;">
          <label>Compare list (order = columns)</label>
          <select id="cmp_list" multiple size="10" style="width:100%; padding:6px; margin-top:4px;"></select>
          <div style="display:flex; gap:8px; margin-top:8px; flex-wrap:wrap;">
            <button onclick="cmpUp()">Up</button>
            <button onclick="cmpDown()">Down</button>
            <button onclick="cmpRemove()">Remove</button>
            <button onclick="cmpClear()">Clear</button>
          </div>
        </div>
      </div>

      <div class="row" style="align-items:flex-end; margin-top:10px;">
        <div style="flex:1; min-width:200px;">
          <label>Metric</label>
          <select id="cmp_metric">
            <option value="response">Response</option>
            <option value="difference">Difference</option>
            <option value="impulse">Impulse</option>
          </select>
        </div>

        <div style="flex:1; min-width:200px;">
          <label>Channels</label>
          <select id="cmp_channels">
            <option value="auto">Auto (union)</option>
            <option value="L">L</option>
            <option value="R">R</option>
            <option value="LR">L+R</option>
            <option value="mono">mono</option>
          </select>
        </div>
      </div>

      <div class="row" style="align-items:flex-end; margin-top:10px;">
        <div style="flex:1; min-width:160px;">
          <label>Tile width (px)</label>
          <input id="cmp_tile_w" type="number" value="1100" />
        </div>
        <div style="flex:1; min-width:160px;">
          <label>Tile height (px)</label>
          <input id="cmp_tile_h" type="number" value="650" />
        </div>
        <div style="flex:1; min-width:160px;">
          <label>DPI</label>
          <input id="cmp_dpi" type="number" value="150" />
        </div>
      </div>

        <div class="block">
        <h4>Plot y-axis</h4>
        <div class="small">
          Locks the y-axis for compare plots so run-to-run comparisons do not lie by autoscaling.
        </div>

        <label style="display:flex; gap:10px; align-items:center; margin-top:10px;">
          <input id="cmp_lock_y_axis" type="checkbox" checked style="width:auto; margin:0;" />
          Lock y-axis
        </label>

        <div class="row" style="align-items:flex-end; margin-top:8px;">
          <div style="flex:1; min-width:160px;">
            <label>Min dB</label>
            <input id="cmp_plot_y_min" type="number" step="0.5" value="__CMP_PLOT_Y_MIN_DEFAULT__" />
          </div>

          <div style="flex:1; min-width:160px;">
            <label>Max dB</label>
            <input id="cmp_plot_y_max" type="number" step="0.5" value="__CMP_PLOT_Y_MAX_DEFAULT__" />
          </div>
        </div>
      </div>

      <button onclick="doCompare()">Render compare grid</button>

      <pre id="cmp_log"></pre>
      <div id="cmp_img"></div>
    </div>

  <!-- </div> -->

  <!-- Footer -->  

  <footer class="page-footer">
    <div class="footer-main">
      <div class="footer-title">cassette-calibrator (WebUI) by FlyingFathead</div>
      <div>
  <a href="https://github.com/FlyingFathead" target="_blank" rel="noopener noreferrer">
        Copyright © 2026 FlyingFathead
    </a>
    </div>
    </div>
        <div class="footer-meta">
        Project homepage on GitHub:
        <a href="https://github.com/FlyingFathead/cassette-calibrator" target="_blank" rel="noopener noreferrer">
            github.com/FlyingFathead/cassette-calibrator
        </a>
        <br />
        Version: <code>__APP_VERSION__</code>
        </div>
  </footer>

  <!-- Image viewer (fullscreen-ish) -->
  <div id="iv_backdrop" class="iv-backdrop" onclick="closeImgViewer()"></div>
  <div id="iv" class="iv" role="dialog" aria-modal="true">
    <div class="iv-head">
      <div id="iv_title" class="iv-title"></div>
      <div class="iv-actions">
        <a id="iv_open" class="imglink" href="#" target="_blank" rel="noopener">Open in new tab ↗</a>
        <button id="iv_fitbtn" class="imgbtn" onclick="ivToggleFit()">Fit</button>
        <button class="imgbtn" onclick="closeImgViewer()">Close</button>
      </div>
    </div>
    <div id="iv_meta" class="iv-meta"></div>
    <div class="iv-body">
      <img id="iv_img" src="" alt="" />
    </div>
  </div>

  <!-- Modal file browser -->
  <div id="mb_backdrop" class="modal-backdrop" onclick="closeBrowser()"></div>
  <div id="mb" class="modal" role="dialog" aria-modal="true">
    <div class="modal-header">
      <div id="mb_title" style="font-weight:600;">Browse</div>
      <div id="mb_path" class="path"></div>
      <button onclick="closeBrowser()">Close</button>
    </div>

    <div class="modal-body">

        <div class="modal-controls">
          <button id="mb_up_btn" onclick="mbUp()">Up</button>
          <button onclick="mbHome()">Home</button>
          <input id="mb_q" placeholder="filter..." oninput="mbRefresh()" />
          <span id="mb_hint" class="muted"></span>
        </div>

        <div id="mb_status" class="mb-status" role="alert" aria-live="polite"></div>        

        <div id="mb_upload_row" class="block" style="display:none; margin-top:8px;">
        <h4 style="margin:0 0 8px 0;">Upload WAV</h4>
        <div class="small">Upload a WAV from your browser into the currently open directory.</div>
        <input id="mb_upload_file" type="file" accept=".wav,audio/wav" />
        <div style="display:flex; gap:8px; margin-top:8px;">
            <button onclick="mbUploadWav()">Upload</button>
        </div>
        <div id="mb_upload_status" class="small" style="margin-top:8px;"></div>
        </div>

        <div id="mb_list" class="modal-list"></div>

    </div>
  </div>

  <!-- Overwrite warning modal -->
  <div id="ow_backdrop" class="ow-backdrop" onclick="owCancel()"></div>
  <div id="ow_modal" class="ow-modal" role="dialog" aria-modal="true" aria-labelledby="ow_title">
    <h3 id="ow_title" style="margin:0 0 10px 0;">Overwrite warning</h3>

    <div class="small" style="margin-bottom:10px;">
      The target file already exists. Continuing with the same path will overwrite it.
    </div>

    <label style="margin-top:0;">Existing path</label>
    <div id="ow_path" class="ow-path"></div>

    <label>Auto-rename suggestion</label>
    <div id="ow_suggested_path" class="ow-path"></div>

    <div class="ow-actions">
      <button onclick="owCancel()">Cancel</button>
      <button onclick="owChooseAutorename()">Auto-rename</button>
      <button class="ow-btn-danger" onclick="owChooseOverwrite()">Overwrite</button>
    </div>
  </div>  

<script>

const DTMF_PRESETS = __DTMF_PRESETS_JSON__;
const URL_PREFIX = __URL_PREFIX_JSON__;
const WEBUI_TS_GEN_CFG = __GEN_OUT_TS_CFG_JSON__;

function prefixedUrl(u) {
  const s = String(u || "");
  if (!URL_PREFIX) return s;
  if (/^https?:\/\//i.test(s)) return s;
  if (s === URL_PREFIX || s.startsWith(URL_PREFIX + "/")) return s;
  if (s.startsWith("/")) return URL_PREFIX + s;
  return URL_PREFIX + "/" + s;
}

function initDtmfPresetSelect() {
  const sel = document.getElementById("dtmf_preset");
  if (!sel) return;

  const keep = ["default", "noisy_cassette", "line_in", "aggressive"];
  const labels = {
    default: "Default",
    noisy_cassette: "Noisy cassette",
    line_in: "Line-in",
    aggressive: "Aggressive"
  };

  // start fresh
  sel.innerHTML = "";

  // built-ins first
  for (const k of keep) {
    if (!DTMF_PRESETS[k]) continue;
    const o = document.createElement("option");
    o.value = k;
    o.textContent = labels[k] || k;
    sel.appendChild(o);
  }

  // then TOML/user presets
  const extra = Object.keys(DTMF_PRESETS || {})
    .filter(k => !keep.includes(k))
    .sort((a,b) => a.localeCompare(b));

  for (const k of extra) {
    const o = document.createElement("option");
    o.value = k;
    o.textContent = k; // keep raw preset name
    sel.appendChild(o);
  }
}

function applyDtmfPreset(name) {
  const p = (DTMF_PRESETS && DTMF_PRESETS[name]) ? DTMF_PRESETS[name] : (DTMF_PRESETS["default"] || {});
  if (p.min_dbfs !== undefined) document.getElementById("dtmf_min_dbfs").value = p.min_dbfs;
  if (p.thresh !== undefined) document.getElementById("dtmf_thresh").value = p.thresh;
  if (p.marker_channel !== undefined) document.getElementById("dtmf_marker_channel").value = p.marker_channel;
}

function readDtmfCfg() {
  return {
    preset: document.getElementById("dtmf_preset").value,
    autotune: document.getElementById("dtmf_autotune").checked,
    min_dbfs: parseFloat(document.getElementById("dtmf_min_dbfs").value),
    thresh: parseFloat(document.getElementById("dtmf_thresh").value),
    marker_channel: document.getElementById("dtmf_marker_channel").value
  };
}

function syncAnalyzeYAxisUi() {
  const locked = document.getElementById("an_lock_y_axis").checked;
  document.getElementById("an_plot_y_min").disabled = !locked;
  document.getElementById("an_plot_y_max").disabled = !locked;
}

function syncCompareYAxisUi() {
  const locked = document.getElementById("cmp_lock_y_axis").checked;
  document.getElementById("cmp_plot_y_min").disabled = !locked;
  document.getElementById("cmp_plot_y_max").disabled = !locked;
}

function mbSetStatus(msg, kind = "info") {
  const el = document.getElementById("mb_status");
  if (!el) return;

  const text = String(msg || "").trim();
  if (!text) {
    el.textContent = "";
    el.className = "mb-status";
    return;
  }

  el.textContent = text;
  el.className = `mb-status show ${kind}`;
}

function mbClearStatus() {
  mbSetStatus("", "info");
}

let MB = {
  open: false,
  targetId: null,
  mode: "file",
  exts: [],
  cwd: "",
  title: "Browse",
  allowNew: false,
  allowUpload: true,
  scope: ""
};

let OW = {
  open: false,
  info: null,
  onDecision: null
};

let RUNS_CTX = {
  dir: "",
  summary: null
};

let IV = {
  open: false,
  fit: true,
  path: ""
};

let CMP = {
  runs: [] // array of {dir,label}
};

function _selValues(selectEl) {
  return Array.from(selectEl.selectedOptions || []).map(o => o.value);
}

function _moveOption(selectFrom, selectTo, value, text) {
  // avoid duplicates
  for (const o of (selectTo.options || [])) {
    if (o.value === value) return;
  }
  const opt = document.createElement("option");
  opt.value = value;
  opt.textContent = text || value;
  selectTo.appendChild(opt);
}

function fmtTimeSec(v) {
  if (v === null || v === undefined || v === "") return "not found";
  const n = Number(v);
  if (!Number.isFinite(n)) return "not found";
  return `${n.toFixed(3)} s`;
}

function parseGenLog(log) {
  const s = String(log || "");
  const out = {
    out_path: null,
    sr: null,
    dur_s: null,

    peak_sample_amplitude: null,
    peak_dbfs: null,

    marker_start: null,
    marker_end: null,

    noisewin_s: null,

    tone_hz: null,
    tone_s: null,
    tone_dbfs: null,

    sweep_f1: null,
    sweep_f2: null,
    sweep_s: null,
    sweep_dbfs: null,

    spoken_cues_mode: null,
    spoken_cues_status: null,
    spoken_cues_pad_s: null,
    spoken_cues_dbfs: null,

    raw: s
  };

  let m;

  m = s.match(/^\s*Wrote:\s+(.+)$/m);
  if (m) out.out_path = m[1].trim();

  // Old legacy line:
  // sr=44100, dur=56.62s, peak=0.251
  m = s.match(/^\s*sr\s*=\s*([0-9]+)\s*,\s*dur\s*=\s*([0-9.]+)s(?:\s*,\s*peak\s*=\s*([0-9.]+))?\s*$/m);
  if (m) {
    out.sr = Number(m[1]);
    out.dur_s = Number(m[2]);
    if (m[3] !== undefined) out.peak_sample_amplitude = Number(m[3]);
  }

  // Newer explicit lines:
  // peak_sample_amplitude=0.251167 (...)
  // audio_peak_level_dbfs=-12.00 dBFS
  m = s.match(/^\s*peak_sample_amplitude\s*=\s*([0-9.]+)\b.*$/mi);
  if (m) out.peak_sample_amplitude = Number(m[1]);

  m = s.match(/^\s*audio_peak_level_dbfs\s*=\s*(-?[0-9.]+)\s*dBFS\s*$/mi);
  if (m) out.peak_dbfs = Number(m[1]);

  // Human-readable variants:
  // Peak sample amplitude  0.251167 (normalized linear amplitude; 1.0 = full scale)
  m = s.match(/^\s*Peak sample amplitude\s+([0-9.]+)\b.*$/mi);
  if (m) out.peak_sample_amplitude = Number(m[1]);

  // Audio peak level (dBFS)  -12.00 dBFS
  m = s.match(/^\s*Audio peak level(?:\s*\(dBFS\))?\s+(-?[0-9.]+)\s*dBFS\s*$/mi);
  if (m) out.peak_dbfs = Number(m[1]);

  // Legacy ambiguous format:
  // [File] Sample peak       0.251 FS (-12.00 dBFS)
  m = s.match(/^\s*(?:\[File\]\s*)?Sample peak\s+([0-9.]+)\s+FS\s*\(\s*(-?[0-9.]+)\s*dBFS\s*\)\s*$/mi);
  if (m) {
    out.peak_sample_amplitude = Number(m[1]);
    out.peak_dbfs = Number(m[2]);
  }

  m = s.match(/^\s*marker_start='([^']*)'\s*,\s*marker_end='([^']*)'\s*$/m);
  if (m) {
    out.marker_start = m[1];
    out.marker_end = m[2];
  }

  m = s.match(/^\s*noisewin_s\s*=\s*([0-9.]+)\s*$/m);
  if (m) out.noisewin_s = Number(m[1]);

  m = s.match(/^\s*tone\s*=\s*([0-9.]+)Hz\s+for\s+([0-9.]+)s\s+at\s+(-?[0-9.]+)\s+dBFS peak\s*$/m);
  if (m) {
    out.tone_hz = Number(m[1]);
    out.tone_s = Number(m[2]);
    out.tone_dbfs = Number(m[3]);
  }

  m = s.match(/^\s*sweep\s*=\s*([0-9.]+)-([0-9.]+)Hz\s+for\s+([0-9.]+)s\s+at\s+(-?[0-9.]+)\s+dBFS peak\s*$/m);
  if (m) {
    out.sweep_f1 = Number(m[1]);
    out.sweep_f2 = Number(m[2]);
    out.sweep_s = Number(m[3]);
    out.sweep_dbfs = Number(m[4]);
  }

  m = s.match(/^\s*spoken_cues=(on|off|requested)(?:,\s*status=(.*?))?(?:,\s*pad_s=([0-9.]+))?(?:,\s*level=(-?[0-9.]+)\s+dBFS peak)?\s*$/m);
  if (m) {
    out.spoken_cues_mode = m[1];
    out.spoken_cues_status = m[2] ? m[2].trim() : null;
    out.spoken_cues_pad_s = m[3] !== undefined ? Number(m[3]) : null;
    out.spoken_cues_dbfs = m[4] !== undefined ? Number(m[4]) : null;
  }

  return out;
}

function renderGenSummary(info) {
  if (!info || typeof info !== "object") {
    return `<div class="op-summary fail">
      <h4>Generation summary</h4>
      <div class="status-line status-fail">FAIL ❌ no valid generation result</div>
    </div>`;
  }

  const hasOut = !!optStr(info.out_path);
  const hasCore = Number.isFinite(info.sr) && Number.isFinite(info.dur_s);

  let statusClass = "fail";
  let statusText = "FAIL ❌ test WAV generation failed";
  let detailText = "The output file was not created successfully.";

  if (hasOut && hasCore) {
    statusClass = "ok";
    statusText = "OK ✅ test WAV was generated successfully";
    detailText = "The reference WAV was written and is ready for use.";
  } else if (hasOut) {
    statusClass = "warn";
    statusText = "WARNING ⚠️ file was written, but the summary could not be parsed fully";
    detailText = "The WAV may still be usable, but some output fields were not parsed from the terminal log.";
  }

  const peakAmp = Number.isFinite(info.peak_sample_amplitude)
    ? `${esc(info.peak_sample_amplitude.toFixed(6))}
        <div class="muted sum-help">normalized linear sample amplitude; 1.0 = full scale</div>`
    : "(not reported)";

  const peakDbfs = Number.isFinite(info.peak_dbfs)
    ? `${esc(info.peak_dbfs.toFixed(2))} dBFS`
    : "(not reported)";

  const toneText =
    Number.isFinite(info.tone_hz) && Number.isFinite(info.tone_s) && Number.isFinite(info.tone_dbfs)
      ? `${esc(String(info.tone_hz))} Hz for ${esc(String(info.tone_s))} s at ${esc(String(info.tone_dbfs))} dBFS`
      : "(not reported)";

  const sweepText =
    Number.isFinite(info.sweep_f1) && Number.isFinite(info.sweep_f2) &&
    Number.isFinite(info.sweep_s) && Number.isFinite(info.sweep_dbfs)
      ? `${esc(String(info.sweep_f1))}-${esc(String(info.sweep_f2))} Hz for ${esc(String(info.sweep_s))} s at ${esc(String(info.sweep_dbfs))} dBFS`
      : "(not reported)";

  const spokenMode = optStr(info.spoken_cues_mode) || "(not reported)";
  const spokenStatus = optStr(info.spoken_cues_status) || "(not reported)";
  const spokenPad = Number.isFinite(info.spoken_cues_pad_s)
    ? `${esc(String(info.spoken_cues_pad_s))} s`
    : "(not reported)";
  const spokenLevel = Number.isFinite(info.spoken_cues_dbfs)
    ? `${esc(String(info.spoken_cues_dbfs))} dBFS`
    : "(not reported)";

  return `
    <div class="op-summary ${statusClass}">
      <h4>Generation summary</h4>
      <div class="status-line status-${statusClass}">${esc(statusText)}</div>
      <div style="margin-bottom:10px;">${esc(detailText)}</div>

      <div class="sum-section">
        <h5>File</h5>
        <dl>
          <dt>Output file</dt>
          <dd><span class="mono">${esc(optStr(info.out_path) || "(unknown)")}</span></dd>

          <dt>Sample rate</dt>
          <dd>${Number.isFinite(info.sr) ? esc(String(info.sr)) + " Hz" : "(unknown)"}</dd>

          <dt>Duration</dt>
          <dd>${Number.isFinite(info.dur_s) ? esc(info.dur_s.toFixed(2)) + " s" : "(unknown)"}</dd>

          <dt>Peak sample amplitude</dt>
          <dd>${peakAmp}</dd>

          <dt>Audio peak level (dBFS)</dt>
          <dd>${peakDbfs}</dd>
        </dl>
      </div>

      <div class="sum-section">
        <h5>DTMF markers</h5>
        <dl>
          <dt>Start marker</dt>
          <dd><code>${esc(optStr(info.marker_start) || "(unknown)")}</code></dd>

          <dt>End marker</dt>
          <dd><code>${esc(optStr(info.marker_end) || "(unknown)")}</code></dd>
        </dl>
      </div>

      <div class="sum-section">
        <h5>Signal program</h5>
        <dl>
          <dt>Noise window</dt>
          <dd>${Number.isFinite(info.noisewin_s) ? esc(String(info.noisewin_s)) + " s" : "(not reported)"}</dd>

          <dt>Reference tone</dt>
          <dd>${toneText}</dd>

          <dt>Sweep</dt>
          <dd>${sweepText}</dd>
        </dl>
      </div>

      <div class="sum-section">
        <h5>Spoken cues</h5>
        <dl>
          <dt>Mode</dt>
          <dd>${esc(spokenMode)}</dd>

          <dt>Status</dt>
          <dd>${esc(spokenStatus)}</dd>

          <dt>Pad</dt>
          <dd>${spokenPad}</dd>

          <dt>Level</dt>
          <dd>${spokenLevel}</dd>
        </dl>
      </div>
    </div>
  `;
}

function renderDetectSummary(result) {
  if (!result || typeof result !== "object") {
    return `<div class="detect-summary fail">
      <h4>Detection summary</h4>
      <div class="status-line status-fail">FAIL -- no valid detection result</div>
    </div>`;
  }

  const hasStart = result.t_marker_start !== null && result.t_marker_start !== undefined;
  const hasEnd = result.t_marker_end !== null && result.t_marker_end !== undefined;

  let statusClass = "fail";
  let statusText = "FAIL ❌ marker detection failed";
  let detailText = "Neither the start marker nor the end marker was found.";

  if (hasStart && hasEnd) {
    statusClass = "ok";
    statusText = "OK ✅ both start and end markers were found";
    detailText = "This recording passed the basic marker check and should be usable for analysis.";
  } else if (hasStart || hasEnd) {
    statusClass = "warn";
    statusText = "WARNING ⚠️ only one marker was found, the result likely cannot be analyzed. Please re-record your audio with both start and end markers intact.";
    detailText = hasStart
      ? "The start marker was found, but the end marker is missing."
      : "The end marker was found, but the start marker is missing.";
  }

  const wav = result.wav || "(unknown)";
  const sr = result.sr ?? "(unknown)";
  const channel = result.channel || "(unknown)";
  const markerStart = result.marker_start || "(unknown)";
  const markerEnd = result.marker_end || "(unknown)";
  const events = result.events ?? "(unknown)";

  return `
    <div class="detect-summary ${statusClass}">
      <h4>Detection summary</h4>
      <div class="status-line status-${statusClass}">${esc(statusText)}</div>
      <div style="margin-bottom:10px;">${esc(detailText)}</div>

      <dl>
        <dt>File</dt>
        <dd><span class="mono">${esc(wav)}</span></dd>

        <dt>Sample rate</dt>
        <dd>${esc(String(sr))} Hz</dd>

        <dt>Detection channel</dt>
        <dd>${esc(String(channel))}</dd>

        <dt>Start marker pattern</dt>
        <dd><code>${esc(String(markerStart))}</code></dd>

        <dt>End marker pattern</dt>
        <dd><code>${esc(String(markerEnd))}</code></dd>

        <dt>Start marker</dt>
        <dd>${hasStart ? `✅ found at ${fmtTimeSec(result.t_marker_start)}` : `❌ not found`}</dd>

        <dt>End marker</dt>
        <dd>${hasEnd ? `✅ found at ${fmtTimeSec(result.t_marker_end)}` : `❌ not found`}</dd>

        <dt>Start marker center</dt>
        <dd>${fmtTimeSec(result.t_marker_start_center)}</dd>

        <dt>End marker center</dt>
        <dd>${fmtTimeSec(result.t_marker_end_center)}</dd>

        <dt>DTMF events detected</dt>
        <dd>${esc(String(events))}</dd>
      </dl>
    </div>
  `;
}

async function rerunSelectedRun() {
  try {
    const dir = (document.getElementById("runs_sel").value || "").trim();
    if (!dir) return;

    setLog("runs_log", "regenerating run from saved source WAVs...");

    const r = await api("/api/run_regen", {
      dir,
      lock_y_axis: document.getElementById("runs_lock_y_axis").checked,
      plot_y_min: parseFloat(document.getElementById("runs_plot_y_min").value),
      plot_y_max: parseFloat(document.getElementById("runs_plot_y_max").value)
    });

    const newRunDir = ((((r || {}).summary || {}).run || {}).outdir || "").trim();

    await refreshRuns({
      preselectDir: newRunDir,
      autoload: true,
      preserveSelection: false,
      placeholder: true
    });

    // prepend regen command log on top of the loaded summary JSON
    if (r.log) {
      const cur = document.getElementById("runs_log").textContent || "";
      setLog("runs_log", r.log + "\n\n" + cur);
    }

    await refreshCompareRuns(newRunDir);
  } catch (e) {
    setLog("runs_log", "ERROR: " + e.message);
  }
}

async function refreshCompareRuns(preselectDir) {
  try {
    const avail = document.getElementById("cmp_avail");

    // preserve old selections
    const prevSelected = new Set(
      Array.from(avail.selectedOptions || []).map(o => o.value)
    );

    setLog("cmp_log", "loading runs...");
    const r = await apiGetJson("/api/runs");

    let html = "";
    for (const it of (r.runs || [])) {
      const label = (it.label || it.dir || "");
      const val = it.dir || "";
      html += `<option value="${esc(val)}">${esc(label)}</option>`;
    }
    avail.innerHTML = html || "<option value=''>no runs found</option>";

    // restore old selection if possible
    for (const o of Array.from(avail.options || [])) {
      if (prevSelected.has(o.value)) o.selected = true;
    }

    // optional explicit preselect
    if (preselectDir) {
      for (const o of Array.from(avail.options || [])) {
        if (o.value === preselectDir) {
          o.selected = true;
          break;
        }
      }
    }

    setLog("cmp_log", JSON.stringify(r, null, 2));
  } catch (e) {
    setLog("cmp_log", "ERROR: " + e.message);
  }
}

function cmpAdd() {
  const avail = document.getElementById("cmp_avail");
  const list = document.getElementById("cmp_list");
  for (const o of Array.from(avail.selectedOptions || [])) {
    _moveOption(avail, list, o.value, o.textContent);
  }
}

function cmpRemove() {
  const list = document.getElementById("cmp_list");
  const gone = Array.from(list.selectedOptions || []);
  for (const o of gone) o.remove();
}

function cmpClear() {
  document.getElementById("cmp_list").innerHTML = "";
}

function cmpUp() {
  const list = document.getElementById("cmp_list");
  const sel = Array.from(list.selectedOptions || []);
  // move each selected option up one step (preserving order)
  for (const o of sel) {
    const prev = o.previousElementSibling;
    if (prev && !prev.selected) list.insertBefore(o, prev);
  }
}

function cmpDown() {
  const list = document.getElementById("cmp_list");
  const sel = Array.from(list.selectedOptions || []).reverse();
  for (const o of sel) {
    const next = o.nextElementSibling;
    if (next && !next.selected) list.insertBefore(next, o);
  }
}

function _cmpChannels() {
  const v = document.getElementById("cmp_channels").value;
  if (v === "LR") return ["L","R"];
  if (v === "auto") return ["auto"];
  return [v];
}

async function doCompare() {
  try {
    setLog("cmp_log", "rendering...");
    document.getElementById("cmp_img").innerHTML = "";

    const list = document.getElementById("cmp_list");
    const runs = Array.from(list.options || []).map(o => o.value).filter(Boolean);

    if (!runs.length) {
      setLog("cmp_log", "ERROR: compare list is empty");
      return;
    }

    const metric = document.getElementById("cmp_metric").value;
    const channels = _cmpChannels();

    const tile_w = parseInt(document.getElementById("cmp_tile_w").value || "1100", 10);
    const tile_h = parseInt(document.getElementById("cmp_tile_h").value || "650", 10);
    const dpi = parseInt(document.getElementById("cmp_dpi").value || "150", 10);

    const lock_y_axis = document.getElementById("cmp_lock_y_axis").checked;
    const plot_y_min = parseFloat(document.getElementById("cmp_plot_y_min").value);
    const plot_y_max = parseFloat(document.getElementById("cmp_plot_y_max").value);

    const r = await api("/api/compare", {
      runs,
      metric,
      channels,
      tile_w,
      tile_h,
      dpi,
      lock_y_axis,
      plot_y_min,
      plot_y_max
    });

    let head = `ok -- ${r.metric} -- channels=${(r.channels||[]).join(",")} -- dpi=${r.dpi} -- tile=${r.tile_w}x${r.tile_h}\n`;
    if (r.log) head += "\n" + r.log;
    setLog("cmp_log", head);

    if (r.out_png) {
      document.getElementById("cmp_img").innerHTML = imgTag(r.out_png);
    }
  } catch (e) {
    setLog("cmp_log", "ERROR: " + e.message);
  }
}

function fileUrl(path) {
  return prefixedUrl("/file?path=" + encodeURIComponent(path));
}

function downloadUrl(path) {
  return prefixedUrl("/download?path=" + encodeURIComponent(path));
}

function imgUrl(path) {
  return fileUrl(path);
}

function fileActionTag(path) {
  const openUrl = fileUrl(path);
  const dlUrl = downloadUrl(path);
  return `
    <div class="block">
      <div class="small"><span class="mono file-path">${esc(path)}</span></div>
      <div class="imglinks" style="margin-top:8px;">
        <a class="imglink" href="${openUrl}" target="_blank" rel="noopener">Open ↗</a>
        <a class="imglink" href="${dlUrl}">Download ⬇</a>
      </div>
    </div>
  `;
}

function ivSetFit(fit) {
  IV.fit = !!fit;
  const img = document.getElementById("iv_img");
  const btn = document.getElementById("iv_fitbtn");
  if (!img || !btn) return;

  if (IV.fit) {
    img.classList.remove("iv-actual");
    btn.textContent = "Actual";
  } else {
    img.classList.add("iv-actual");
    btn.textContent = "Fit";
  }
}

function ivToggleFit() {
  ivSetFit(!IV.fit);
}

async function openImgViewer(path) {
  try {
    path = String(path || "").trim();
    if (!path) return;

    IV.open = true;
    IV.path = path;

    const url = imgUrl(path);

    document.getElementById("iv_backdrop").style.display = "block";
    document.getElementById("iv").style.display = "block";
    document.getElementById("iv_title").textContent = path;

    const img = document.getElementById("iv_img");
    img.src = url;
    img.alt = path;

    // Click in viewer toggles Fit/Actual
    img.onclick = ivToggleFit;

    const open = document.getElementById("iv_open");
    open.href = url;

    document.getElementById("iv_meta").textContent = "loading…";

    // Fetch basic file stats (bytes + mtime) from server
    try {
      const st = await apiGetJson("/api/stat?path=" + encodeURIComponent(path));
      const size = (st && st.size != null) ? (st.size + " bytes") : "";
      const mtime = (st && st.mtime) ? (new Date(st.mtime * 1000).toLocaleString()) : "";

      let line = "";
      if (size) line += size;
      if (mtime) line += (line ? " -- " : "") + mtime;

      document.getElementById("iv_meta").textContent = line;
    } catch (e) {
      document.getElementById("iv_meta").textContent = "";
    }

    // Default to fit mode on open
    ivSetFit(true);
  } catch (e) {
    console.log("openImgViewer error:", e);
  }
}

function closeImgViewer() {
  IV.open = false;
  document.getElementById("iv_backdrop").style.display = "none";
  document.getElementById("iv").style.display = "none";

  const img = document.getElementById("iv_img");
  if (img) {
    img.src = "";
    img.alt = "";
    img.onclick = null;
  }
}

document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;

  if (OW.open) {
    owCancel();
    return;
  }

  if (IV.open) {
    closeImgViewer();
    return;
  }
});

function runsEditNotes() {
  if (!RUNS_CTX.dir || !RUNS_CTX.summary) return;

  const run = (RUNS_CTX.summary && RUNS_CTX.summary.run) ? RUNS_CTX.summary.run : {};
  const notes = optStr(run.notes);

  document.getElementById("runs_notes_text").value = notes;
  document.getElementById("runs_notes_status").textContent = "";

  document.getElementById("runs_notes_ctl").style.display = "block";
  document.getElementById("runs_notes_edit_btn").style.display = "none";
}

function runsCancelNotes() {
  document.getElementById("runs_notes_status").textContent = "";
  document.getElementById("runs_notes_ctl").style.display = "none";
  document.getElementById("runs_notes_edit_btn").style.display = RUNS_CTX.dir ? "inline-block" : "none";
}

async function runsSaveNotes() {
  try {
    if (!RUNS_CTX.dir || !RUNS_CTX.summary) return;

    const notes = document.getElementById("runs_notes_text").value || "";
    document.getElementById("runs_notes_status").textContent = "saving...";

    const r = await api("/api/run_notes", { dir: RUNS_CTX.dir, notes });
    const summary = (r && r.summary) ? r.summary : RUNS_CTX.summary;

    RUNS_CTX.summary = summary;

    // Re-render header + json + images
    document.getElementById("runs_header").innerHTML =
      renderRunHeader(summary, "Selected run", "runs", true) +
      `<div class="runhdr"><div class="meta">Dir: <span class="mono">${esc(RUNS_CTX.dir)}</span></div></div>`;

    setLog("runs_log", JSON.stringify(summary, null, 2));
    document.getElementById("runs_imgs").innerHTML =
      renderImagesWithAnchors(summary, "runs");

    document.getElementById("runs_notes_ctl").style.display = "none";
    document.getElementById("runs_notes_edit_btn").style.display = "inline-block";
    document.getElementById("runs_notes_status").textContent = "";
  } catch (e) {
    document.getElementById("runs_notes_status").textContent = "ERROR: " + e.message;
  }
}

function esc(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function jsq(s) {
  // returns a quoted JS string literal, safe to embed inside onclick=...
  return JSON.stringify(String(s || ""));
}

async function api(path, payload) {
  const r = await fetch(prefixedUrl(path), {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(payload || {})
  });

  const t = await r.text();
  let j = null;
  try { j = JSON.parse(t); } catch (e) {}

  if (!r.ok) {
    const msg = (j && j.error) ? String(j.error) : String(t || "request failed");
    const log = (j && j.log) ? String(j.log) : "";
    const extra = (j && j.path) ? ("\n\npath: " + String(j.path)) : "";

    const full = log ? (msg + extra + "\n\n--- log ---\n" + log) : (msg + extra);
    throw new Error(full);
  }

  return j;
}

async function apiGetJson(url) {
  const r = await fetch(prefixedUrl(url));
  const t = await r.text();
  let j = null;
  try { j = JSON.parse(t); } catch (e) {}

  if (!r.ok) {
    const msg = (j && j.error) ? String(j.error) : String(t || "request failed");
    const log = (j && j.log) ? String(j.log) : "";
    const extra = (j && j.path) ? ("\n\npath: " + String(j.path)) : "";

    const full = log ? (msg + extra + "\n\n--- log ---\n" + log) : (msg + extra);
    throw new Error(full);
  }

  return j;
}

function setLog(id, s) {
  document.getElementById(id).textContent = s || "";
}

function imgTag(path) {
  const url = imgUrl(path);
  return `
    <div class="imgwrap">
      <div class="imgmeta">
        <div class="small"><span class="mono">${esc(path)}</span></div>
        <div class="imglinks">
          <a class="imglink" href="${url}" target="_blank" rel="noopener">Open in new tab ↗</a>
          <button class="imgbtn" onclick='openImgViewer(${jsq(path)})'>Fullscreen ⤢</button>
        </div>
      </div>
      <img class="imgthumb" src="${url}" alt="${esc(path)}" loading="lazy"
           onclick='openImgViewer(${jsq(path)})' />
    </div>
  `;
}

function optStr(v) {
  if (v === null || v === undefined) return "";
  const s = String(v).trim();
  if (!s) return "";
  if (s.toLowerCase() === "null" || s.toLowerCase() === "none") return "";
  return s;
}

function escId(s) {
  // safe-ish HTML id: keep alnum, dash, underscore; everything else becomes "_"
  return String(s || "").replace(/[^a-zA-Z0-9_-]+/g, "_");
}

function renderRunHeader(summary, fallbackTitle, prefix, forceNotes) {
  prefix = prefix || "run";
  const run = (summary && summary.run) ? summary.run : {};
  const name = optStr(run.name);
  const notes = optStr(run.notes);
  const createdUtc = optStr(run.created_at_utc);
  const createdLocal = optStr(run.created_at_local);

  const title = name || optStr(fallbackTitle) || "Run";

  const topId = `${prefix}_top`;
  const notesId = `${prefix}_notes`;
  const jsonId = `${prefix}_json`;
  const imgsId = `${prefix}_imgs`;

  const showNotes = !!forceNotes || !!notes;

  let meta = "";
  if (createdUtc || createdLocal) {
    meta += `<div class="meta">Ran on:</div>`;
    if (createdUtc) meta += `<div class="meta"><span class="mono">${esc(createdUtc)}</span> (UTC)</div>`;
    if (createdLocal) meta += `<div class="meta"><span class="mono">${esc(createdLocal)}</span> (local)</div>`;
  }

  const toc = `
    <div class="toc">
      Jump:
      ${showNotes ? `<a href="#${notesId}">notes</a> | ` : ``}
      <a href="#${imgsId}">images</a> |
      <a href="#${jsonId}">json</a> |
      <a href="#${topId}">top</a>
    </div>
  `;

  const notesBlock = showNotes ? `
    <div class="block" id="${notesId}">
      <h4>Notes <span class="small"><a href="#${notesId}">#</a></span></h4>
      <div class="notes">${notes ? esc(notes) : "(no notes)"}</div>
    </div>
  ` : "";

  return `
    <div class="runhdr" id="${topId}">
      <div class="title">${esc(title)}</div>
      ${meta}
      ${toc}
      ${notesBlock}
    </div>
  `;
}

function syncRunsYAxisUi() {
  const locked = document.getElementById("runs_lock_y_axis").checked;
  document.getElementById("runs_plot_y_min").disabled = !locked;
  document.getElementById("runs_plot_y_max").disabled = !locked;
}

/* -------- modal browser -------- */

function openBrowser(targetId, opts) {
  MB.open = true;
  MB.targetId = targetId;
  MB.mode = (opts && opts.mode) ? opts.mode : "file";
  MB.exts = (opts && opts.exts) ? opts.exts : [];
  MB.title = (opts && opts.title) ? opts.title : "Browse";
  MB.allowNew = !!(opts && opts.allowNew);
  MB.allowUpload = !opts || !Object.prototype.hasOwnProperty.call(opts, "allowUpload")
    ? true
    : !!opts.allowUpload;
  MB.scope = (opts && opts.scope) ? String(opts.scope) : "";

  MB.cwd = "";

  document.getElementById("mb_title").textContent = MB.title;
  document.getElementById("mb_q").value = "";
  document.getElementById("mb_upload_status").textContent = "";
  mbClearStatus();

  document.getElementById("mb_hint").textContent =
    (MB.mode === "dir")
      ? "Pick a directory"
      : ("Pick a file" + (MB.exts.length ? (" (" + MB.exts.join(", ") + ")") : ""));

  document.getElementById("mb_backdrop").style.display = "block";
  document.getElementById("mb").style.display = "block";
  mbRefresh();
}

function closeBrowser() {
  MB.open = false;
  mbClearStatus();
  document.getElementById("mb_backdrop").style.display = "none";
  document.getElementById("mb").style.display = "none";
}

async function mbRefresh() {
  if (!MB.open) return;

  const q = document.getElementById("mb_q").value || "";
  const params = new URLSearchParams();
  if (MB.cwd) params.set("dir", MB.cwd);
  params.set("mode", MB.mode);
  for (const e of MB.exts) params.append("ext", e);
  if (q) params.set("q", q);
  if (MB.scope) params.set("scope", MB.scope);

  const st = await apiGetJson("/api/browse?" + params.toString());
  MB.cwd = st.cwd;
  document.getElementById("mb_path").textContent = st.cwd;

  const rootLabel = st.root_dir || ".";
  const atRoot = !!st.at_root;
  const upBtn = document.getElementById("mb_up_btn");

  if (upBtn) {
    upBtn.title = atRoot
      ? `Already at allowed root: ${rootLabel}`
      : "Go to parent directory";
  }

  document.getElementById("mb_hint").textContent =
    atRoot
      ? `At allowed root: ${rootLabel}`
      : (
          MB.mode === "dir"
            ? "Pick a directory"
            : ("Pick a file" + (MB.exts.length ? (" (" + MB.exts.join(", ") + ")") : ""))
        );  

  const canUploadWav =
    MB.mode === "file" &&
    (MB.exts.includes(".wav") || MB.exts.includes("wav"));

    document.getElementById("mb_upload_row").style.display =
    (canUploadWav && MB.allowUpload) ? "block" : "none";

  const list = document.getElementById("mb_list");
  let html = "";

  if (MB.allowNew && MB.mode === "dir") {
    html += "<div class='item'>"
         + "<div><span class='pill'>new</span> "
         + "<input id='mb_newdir' placeholder='e.g. cassette_results/run_001' style='margin-top:0; width:100%;' /></div>"
         + "<div class='muted'>in <span class='mono'>" + esc(st.cwd) + "</span></div>"
         + "<div><button onclick='mbCreateDir()'>Create</button></div>"
         + "</div>";
  }

  if (MB.allowNew && MB.mode === "file") {
    const current = document.getElementById(MB.targetId).value || "";
    html += "<div class='item'><div><span class='pill'>new</span> <span class='mono'>" + esc(current) + "</span></div>"
         + "<div class='muted'>use current</div>"
         + "<div><button onclick='mbPickValue()'>Use</button></div></div>";
  }

  for (const it of st.entries) {
    const pill = it.is_dir ? "<span class='pill'>dir</span>" : "<span class='pill'>file</span>";
    const meta = it.is_dir ? "" : ("<span class='muted'>" + (it.size || 0) + " bytes</span>");

    let btns = "";
    if (it.is_dir) {
      btns = `<button onclick='mbEnter(${jsq(it.path)})'>Open</button>`;
    } else {
      btns =
        `<button onclick='mbPick(${jsq(it.path)})'>Pick</button> ` +
        `<a class="imglink" href="${downloadUrl(it.path)}">Download</a>`;
    }

    html += "<div class='item'>"
         + "<div>" + pill + " <span class='mono'>" + esc(it.path) + "</span></div>"
         + "<div>" + meta + "</div>"
         + "<div>" + btns + "</div>"
         + "</div>";
  }

  if (!html) html = "<div class='muted'>No entries.</div>";
  list.innerHTML = html;
}

function mbEnter(path) {
  MB.cwd = path;
  mbClearStatus();
  mbRefresh();
}

async function mbUp() {
  try {
    const params = new URLSearchParams();
    if (MB.cwd) params.set("dir", MB.cwd);
    params.set("mode", MB.mode);
    for (const e of MB.exts) params.append("ext", e);
    if (MB.scope) params.set("scope", MB.scope);

    const st = await apiGetJson("/api/browse?" + params.toString());

    if (st.parent) {
      MB.cwd = st.parent;
      mbClearStatus();
      await mbRefresh();
      return;
    }

    MB.cwd = st.cwd || MB.cwd || ".";
    const rootLabel = st.root_dir || st.cwd || ".";
    mbSetStatus(`You can't go above the allowed root directory: ${rootLabel}`, "error");
  } catch (e) {
    mbSetStatus("ERROR: " + e.message, "error");
  }
}

function mbHome() {
  MB.cwd = "";
  mbClearStatus();
  mbRefresh();
}

function mbPick(path) {
  document.getElementById(MB.targetId).value = path;
  closeBrowser();
}

function mbPickValue() {
  // Keep whatever is typed in the target input
  closeBrowser();
}

async function mbCreateDir() {
  try {
    const el = document.getElementById("mb_newdir");
    const nameRaw = (el ? el.value : "").trim();
    if (!nameRaw) return;

    const base = String(MB.cwd || "").trim();

    let name = String(nameRaw);
    while (name.startsWith("/")) name = name.slice(1);
    while (name.endsWith("/")) name = name.slice(0, -1);
    name = name.trim();
    if (!name) return;

    if (name.includes("..")) {
      alert("ERROR: '..' is not allowed");
      return;
    }

    let path = name;

    if (base && base !== ".") {
      let baseClean = String(base);
      while (baseClean.endsWith("/")) baseClean = baseClean.slice(0, -1);
      if (baseClean === ".") baseClean = "";
      path = baseClean ? (baseClean + "/" + name) : name;
    }

    while (path.includes("//")) path = path.replaceAll("//", "/");

    const r = await api("/api/mkdir", { path });

    document.getElementById(MB.targetId).value = (r && r.path) ? r.path : path;
    closeBrowser();
  } catch (e) {
    alert("ERROR: " + e.message);
  }
}

async function mbUploadWav() {
  try {
    const input = document.getElementById("mb_upload_file");
    const status = document.getElementById("mb_upload_status");
    if (!input || !input.files || !input.files.length) {
      status.textContent = "Pick a WAV file first.";
      return;
    }

    const file = input.files[0];
    status.textContent = "uploading...";

    const dir = (MB.cwd && MB.cwd !== ".") ? MB.cwd : "data";
    const fd = new FormData();
    fd.append("file", file, file.name);

    const r = await fetch(prefixedUrl("/api/upload_wav?dir=" + encodeURIComponent(dir)), {
    method: "POST",
    body: fd
    });

    const t = await r.text();
    let j = null;
    try { j = JSON.parse(t); } catch (e) {}

    if (!r.ok) {
      const msg = (j && j.error) ? String(j.error) : String(t || "upload failed");
      throw new Error(msg);
    }

    status.textContent =
      `ok -- ${j.path} -- ${j.wav_info.sample_rate} Hz -- ${j.wav_info.channels} ch -- ${j.wav_info.duration_s.toFixed(2)} s`;

    document.getElementById(MB.targetId).value = j.path;
    await mbRefresh();
  } catch (e) {
    document.getElementById("mb_upload_status").textContent = "ERROR: " + e.message;
  }
}

/* -------- auto-generate timestamped filenames -------- */

function _pad2(n) {
  return String(n).padStart(2, "0");
}

function _fmtLocalTimestampForFilename(d) {
  return (
    String(d.getFullYear()) +
    _pad2(d.getMonth() + 1) +
    _pad2(d.getDate()) +
    "-" +
    _pad2(d.getHours()) +
    _pad2(d.getMinutes()) +
    _pad2(d.getSeconds())
  );
}

function _buildSuggestedTimestampedGenOut(seed) {
  const raw = String(seed || "data/test_audio.wav").trim() || "data/test_audio.wav";
  const norm = raw.replace(/\\/g, "/");

  const slash = norm.lastIndexOf("/");
  const dir = slash >= 0 ? norm.slice(0, slash) : "";
  const base = slash >= 0 ? norm.slice(slash + 1) : norm;

  let ext = ".wav";
  const dot = base.lastIndexOf(".");
  if (dot > 0 && dot < base.length - 1) {
    ext = base.slice(dot);
  }

  const ts = _fmtLocalTimestampForFilename(new Date());
  const name = `cc-test_audio-${ts}${ext}`;

  return (dir && dir !== ".") ? `${dir}/${name}` : name;
}

function initSuggestedGenOutDefault() {
  const cfg = WEBUI_TS_GEN_CFG || {};
  if (!cfg.enabled) return;

  const el = document.getElementById("gen_out");
  if (!el) return;

  const seed = String(cfg.seed || "").trim();
  const current = String(el.value || "").trim();

  // Only auto-fill when the field is still empty or still at the plain seed value.
  // If the user already changed it, leave it alone.
  if (current && seed && current !== seed) return;

  el.value = _buildSuggestedTimestampedGenOut(seed || current || "data/test_audio.wav");
}

/* -------- overwrite errors -------- */

function apiErrorText(status, j, t) {
  const msg = (j && j.error) ? String(j.error) : String(t || "request failed");
  const log = (j && j.log) ? String(j.log) : "";
  const extra = (j && j.path) ? ("\n\npath: " + String(j.path)) : "";
  return log ? (msg + extra + "\n\n--- log ---\n" + log) : (msg + extra);
}

async function postJsonRaw(path, payload) {
  const r = await fetch(prefixedUrl(path), {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(payload || {})
  });

  const t = await r.text();
  let j = null;
  try { j = JSON.parse(t); } catch (e) {}

  return {
    ok: r.ok,
    status: r.status,
    json: j,
    text: t
  };
}

function owOpen(info, onDecision) {
  OW.open = true;
  OW.info = info || null;
  OW.onDecision = onDecision || null;

  document.getElementById("ow_path").textContent =
    (info && info.path) ? String(info.path) : "";

  document.getElementById("ow_suggested_path").textContent =
    (info && info.suggested_path) ? String(info.suggested_path) : "";

  document.getElementById("ow_backdrop").style.display = "block";
  document.getElementById("ow_modal").style.display = "block";
}

function owClose() {
  OW.open = false;
  OW.info = null;
  OW.onDecision = null;

  document.getElementById("ow_backdrop").style.display = "none";
  document.getElementById("ow_modal").style.display = "none";
}

async function owCancel() {
  const cb = OW.onDecision;
  owClose();
  if (cb) await cb("cancel");
}

async function owChooseOverwrite() {
  const cb = OW.onDecision;
  owClose();
  if (cb) await cb("overwrite");
}

async function owChooseAutorename() {
  const cb = OW.onDecision;
  owClose();
  if (cb) await cb("autorename");
}

function handleGenError(e) {
  setLog("gen_log", "ERROR: " + e.message);
  document.getElementById("gen_file").innerHTML = "";
  document.getElementById("gen_summary").innerHTML = `
    <div class="op-summary fail">
      <h4>Generation summary</h4>
      <div class="status-line status-fail">FAIL ❌ generation request failed</div>
      <div>${esc(e.message || "Unknown error")}</div>
    </div>
  `;
}

function applyGenResult(r, requestedOut) {
  const logTxt = r.log || JSON.stringify(r, null, 2);
  setLog("gen_log", logTxt);

  const info = parseGenLog(logTxt);
  info.out_path = info.out_path || r.out || requestedOut;

  document.getElementById("gen_summary").innerHTML = renderGenSummary(info);

  if (r.out) {
    document.getElementById("gen_out").value = r.out;
    document.getElementById("gen_file").innerHTML = fileActionTag(r.out);
  }
}

async function runGenRequest(overwriteMode) {
  const out = (document.getElementById("gen_out").value || "").trim();

  const resp = await postJsonRaw("/api/gen", {
    out,
    overwrite_mode: overwriteMode || ""
  });

  if (resp.status === 409 && resp.json && resp.json.needs_confirmation) {
    setLog("gen_log", `Overwrite warning: ${resp.json.path} already exists.`);
    owOpen(resp.json, async (decision) => {
      if (decision === "cancel") {
        setLog("gen_log", "Generation cancelled.");
        return;
      }

      try {
        setLog("gen_log", "running...");
        await runGenRequest(decision);
      } catch (e) {
        handleGenError(e);
      }
    });
    return;
  }

  if (!resp.ok) {
    throw new Error(apiErrorText(resp.status, resp.json, resp.text));
  }

  applyGenResult(resp.json || {}, out);
}

/* -------- actions -------- */

async function doGen() {
  try {
    setLog("gen_log", "running...");
    document.getElementById("gen_file").innerHTML = "";
    document.getElementById("gen_summary").innerHTML = "";

    await runGenRequest("");
  } catch (e) {
    handleGenError(e);
  }
}

async function doDetect() {
  try {
    setLog("detect_out", "running...");
    document.getElementById("detect_summary").innerHTML = "";

    const wav = document.getElementById("detect_wav").value;
    const r = await api("/api/detect", { wav });

    // Keep the raw terminal-ish JSON view
    setLog("detect_out", JSON.stringify(r.result, null, 2));

    // Add the human-readable summary below it
    document.getElementById("detect_summary").innerHTML =
      renderDetectSummary(r.result);

  } catch (e) {
    setLog("detect_out", "ERROR: " + e.message);
    document.getElementById("detect_summary").innerHTML = `
      <div class="detect-summary fail">
        <h4>Detection summary</h4>
        <div class="status-line status-fail">FAIL -- detection request failed</div>
        <div>${esc(e.message || "Unknown error")}</div>
      </div>
    `;
  }
}

function renderImagesWithAnchors(summary, prefix) {
  prefix = prefix || "run";
  const imgsId = `${prefix}_imgs`;

  const per = (summary && summary.per_channel) ? summary.per_channel : {};
  const so = (summary && summary.stereo_outputs) ? summary.stereo_outputs : {};

  const channels = Object.keys(per).sort();
  let tocLinks = [];
  let body = "";

  // Per-channel sections
  for (const ch of channels) {
    const outs = (per[ch] && per[ch].outputs) ? per[ch].outputs : {};
    const secId = `${prefix}_ch_${escId(ch)}`;

    tocLinks.push(`<a href="#${secId}">${esc(ch)}</a>`);

    let sec = `<div class="imgsec" id="${secId}">
      <h4>Channel ${esc(ch)} <span class="small"><a href="#${secId}">#</a></span></h4>`;

    const items = [
      ["response_png", "Response"],
      ["difference_png", "Difference"],
      ["impulse_png", "Impulse"],
    ];

    let any = false;
    for (const [key, label] of items) {
      const p = outs[key];
      if (!p) continue;
      any = true;

      const blockId = `${secId}_${key}`;
      sec += `
        <div class="block" id="${blockId}">
          <div class="small"><b>${esc(label)}</b> <a href="#${blockId}">#</a> -- ${esc(p)}</div>
          ${imgTag(p)}
        </div>
      `;
    }

    if (!any) {
      sec += `<div class="small">No images for this channel.</div>`;
    }

    sec += `</div>`;
    body += sec;
  }

  // Stereo section
  const stereoItems = [
    ["lr_overlay_png", "L/R Overlay"],
    ["lr_diff_png", "L-R Difference"],
  ];

  let stereoAny = false;
  let stereoBody = "";
  for (const [key, label] of stereoItems) {
    const p = so[key];
    if (!p) continue;
    stereoAny = true;

    const blockId = `${prefix}_stereo_${key}`;
    stereoBody += `
      <div class="block" id="${blockId}">
        <div class="small"><b>${esc(label)}</b> <a href="#${blockId}">#</a> -- ${esc(p)}</div>
        ${imgTag(p)}
      </div>
    `;
  }

  if (stereoAny) {
    tocLinks.push(`<a href="#${prefix}_stereo">stereo</a>`);
    body += `
      <div class="imgsec" id="${prefix}_stereo">
        <h4>Stereo <span class="small"><a href="#${prefix}_stereo">#</a></span></h4>
        ${stereoBody}
      </div>
    `;
  }

  if (!body) {
    return `<div id="${imgsId}" class="small">No images listed in summary.</div>`;
  }

  const toc = tocLinks.length
    ? `<div class="toc" id="${imgsId}">Images: ${tocLinks.join(" | ")}</div>`
    : `<div class="toc" id="${imgsId}">Images</div>`;

  return `${toc}${body}`;
}

async function doAnalyze() {
  try {
    setLog("an_log", "running...");

    // clear previous content
    document.getElementById("an_header").innerHTML = "";
    document.getElementById("an_imgs").innerHTML = "";

    const ref = document.getElementById("an_ref").value;
    const rec = document.getElementById("an_rec").value;
    const loopback = document.getElementById("an_lb").value;
    const outdir = document.getElementById("an_outdir").value;
    const run_name = document.getElementById("an_name").value;

    const run_notes = (document.getElementById("an_notes").value || "").trim();

    const lock_y_axis = document.getElementById("an_lock_y_axis").checked;
    const plot_y_min = parseFloat(document.getElementById("an_plot_y_min").value);
    const plot_y_max = parseFloat(document.getElementById("an_plot_y_max").value);

    const dtmf = readDtmfCfg();
    const r = await api("/api/analyze", {
      ref,
      rec,
      loopback,
      outdir,
      run_name,
      run_notes,
      lock_y_axis,
      plot_y_min,
      plot_y_max,
      dtmf
    });

    // Header (includes notes block if summary.run.notes exists)
    document.getElementById("an_header").innerHTML =
      renderRunHeader(r.summary, "Most recent run", "an");

    // JSON anchor wrapper is id="an_json" (see HTML change above)
    setLog("an_log", (r.log || "") + "\n\n" + JSON.stringify(r.summary, null, 2));

    // Images with per-channel anchors + blocks
    document.getElementById("an_imgs").innerHTML =
      renderImagesWithAnchors(r.summary, "an");

    const newRunDir =
      (((r || {}).summary || {}).run || {}).outdir || "";

    await refreshRuns({
      preselectDir: newRunDir,
      autoload: true,
      preserveSelection: false
    });

    await refreshCompareRuns(newRunDir);

  } catch (e) {
    setLog("an_log", "ERROR: " + e.message);
  }
}

async function refreshRuns(opts) {
  try {
    const cfg = opts || {};
    const preselectDir = cfg.preselectDir || "";
    const autoload = !!cfg.autoload;
    const preserveSelection = (cfg.preserveSelection !== false);
    const placeholder = !!cfg.placeholder;

    const sel = document.getElementById("runs_sel");
    const prevValue = preserveSelection ? (sel.value || "") : "";

    // reset editor UI
    RUNS_CTX.dir = "";
    RUNS_CTX.summary = null;
    document.getElementById("runs_notes_ctl").style.display = "none";
    document.getElementById("runs_notes_status").textContent = "";
    document.getElementById("runs_notes_edit_btn").style.display = "none";

    // clear stale displayed run right away
    document.getElementById("runs_header").innerHTML = "";
    document.getElementById("runs_imgs").innerHTML = "";
    setLog("runs_log", "loading...");

    const r = await apiGetJson("/api/runs");

    let html = "";

    if (placeholder) {
    html += `<option value="">-- pick a run --</option>`;
    }

    for (const it of (r.runs || [])) {
    const label = (it.label || it.dir || "");
    const val = it.dir || "";
    html += `<option value="${esc(val)}">${esc(label)}</option>`;
    }

    sel.innerHTML = html || "<option value=''>no runs found</option>";

    // choose best selection
    const values = Array.from(sel.options || []).map(o => o.value);
    let chosen = "";

    if (preselectDir && values.includes(preselectDir)) {
    chosen = preselectDir;
    } else if (prevValue && values.includes(prevValue)) {
    chosen = prevValue;
    } else if (autoload && values.length && values[0]) {
    chosen = values[0];
    }

    if (chosen) {
      sel.value = chosen;
    }

    setLog("runs_log", JSON.stringify(r, null, 2));

    if (autoload && chosen) {
      await loadSelectedRun();
    }
  } catch (e) {
    setLog("runs_log", "ERROR: " + e.message);
  }
}

async function loadSelectedRun() {
  try {
    const sel = document.getElementById("runs_sel");
    const dir = (sel && sel.value) ? sel.value : "";
    if (!dir) return;

    // clear previous content early
    document.getElementById("runs_header").innerHTML = "";
    document.getElementById("runs_imgs").innerHTML = "";
    setLog("runs_log", "loading run: " + dir);

    // reset editor UI
    document.getElementById("runs_notes_ctl").style.display = "none";
    document.getElementById("runs_notes_status").textContent = "";
    document.getElementById("runs_notes_edit_btn").style.display = "none";

    // old method
    // const sumPath = dir.replace(/\/+$/,"") + "/summary.json";
    // const summary = await apiGetJson("/file?path=" + encodeURIComponent(sumPath));

    const r = await apiGetJson("/api/run_summary?dir=" + encodeURIComponent(dir));
    const summary = r.summary;

    const plotCfg = (summary && summary._webui_plot_cfg) ? summary._webui_plot_cfg : {};
    const runsLock = document.getElementById("runs_lock_y_axis");
    const runsMin = document.getElementById("runs_plot_y_min");
    const runsMax = document.getElementById("runs_plot_y_max");

    if (runsLock) runsLock.checked = !!plotCfg.lock_y_axis;
    if (runsMin && Number.isFinite(Number(plotCfg.plot_y_min))) runsMin.value = Number(plotCfg.plot_y_min);
    if (runsMax && Number.isFinite(Number(plotCfg.plot_y_max))) runsMax.value = Number(plotCfg.plot_y_max);
    syncRunsYAxisUi();
    
    RUNS_CTX.dir = dir;
    RUNS_CTX.summary = summary;

    // show edit button now that we have a run loaded
    document.getElementById("runs_notes_edit_btn").style.display = "inline-block";

    document.getElementById("runs_header").innerHTML =
      renderRunHeader(summary, "Selected run", "runs", true) +
      `<div class="runhdr"><div class="meta">Dir: <span class="mono">${esc(dir)}</span></div></div>`;

    setLog("runs_log", JSON.stringify(summary, null, 2));

    document.getElementById("runs_imgs").innerHTML =
      renderImagesWithAnchors(summary, "runs");

  } catch (e) {
    setLog("runs_log", "ERROR: " + e.message);
  }
}

// DTMF preset wiring
document.getElementById("dtmf_preset").addEventListener("change", (e) => {
  applyDtmfPreset(e.target.value);
});

// init page function
async function initPage() {
  try {
    await refreshCompareRuns();
    await refreshRuns({
      autoload: false,
      preserveSelection: false,
      placeholder: true
    });
  } catch (e) {
    console.log("initPage error:", e);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  initDtmfPresetSelect();
  initSuggestedGenOutDefault();
  applyDtmfPreset(document.getElementById("dtmf_preset").value);

  const anLock = document.getElementById("an_lock_y_axis");
  if (anLock) {
    anLock.addEventListener("change", syncAnalyzeYAxisUi);
    syncAnalyzeYAxisUi();
  }

  const runsLock = document.getElementById("runs_lock_y_axis");
  if (runsLock) {
    runsLock.addEventListener("change", syncRunsYAxisUi);
    syncRunsYAxisUi();
  }

  const cmpLock = document.getElementById("cmp_lock_y_axis");
  if (cmpLock) {
    cmpLock.addEventListener("change", syncCompareYAxisUi);
    syncCompareYAxisUi();
  }  

  initPage();
});

// Make inline onclick= handlers work no matter what scope rules apply
Object.assign(window, {
  doGen,
  doDetect,
  doAnalyze,
  refreshRuns,
  loadSelectedRun,
  rerunSelectedRun,
  openBrowser,
  closeBrowser,
  mbUp,
  mbHome,
  mbEnter,
  mbPick,
  mbPickValue,
  mbCreateDir,
  mbUploadWav,
  mbRefresh,
  runsEditNotes,
  runsSaveNotes,
  runsCancelNotes,
  openImgViewer,
  closeImgViewer,
  ivToggleFit,
  refreshCompareRuns,
  cmpAdd,
  cmpRemove,
  cmpClear,
  cmpUp,
  cmpDown,
  doCompare,
  owCancel,
  owChooseOverwrite,
  owChooseAutorename  
});

console.log("webui script loaded; handlers exported to window");

</script>
</body>
</html>
"""

class Handler(BaseHTTPRequestHandler):
    server_version = f"cassette-calibrator-webui/{APP_VERSION}"

    def _send(self, code: int, body: bytes, ctype: str, headers: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))

        for k, v in (headers or {}).items():
            self.send_header(str(k), str(v))

        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        b = json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")
        self._send(code, b, "application/json; charset=utf-8")

    def _text(self, code: int, s: str, ctype: str = "text/plain; charset=utf-8") -> None:
        self._send(code, (s or "").encode("utf-8"), ctype)

    def log_message(self, fmt, *args):  # quiet by default
        return

    # --- GET and POST with logging ---

    def do_GET(self) -> None:
        t0 = time.monotonic()
        peer = f"{self.client_address[0]}:{self.client_address[1]}"
        LOG.info("GET %s -- from %s", self.path, peer)
        try:
            return self._do_GET()
        except KeyboardInterrupt:
            raise
        except BaseException:
            LOG.exception("Unhandled exception in GET %s -- from %s", self.path, peer)
            try:
                self._json(500, {
                    "error": "internal server error (see server log)",
                    "path": self.path,
                })
            except Exception:
                pass
        finally:
            LOG.info("GET %s -- %.1f ms", self.path, (time.monotonic() - t0) * 1000.0)

    def do_POST(self) -> None:
        t0 = time.monotonic()
        peer = f"{self.client_address[0]}:{self.client_address[1]}"
        clen = self.headers.get("Content-Length", "-")
        LOG.info("POST %s -- from %s -- Content-Length=%s", self.path, peer, clen)
        try:
            return self._do_POST()
        except KeyboardInterrupt:
            raise
        except BaseException:
            LOG.exception("Unhandled exception in POST %s -- from %s", self.path, peer)
            try:
                self._json(500, {
                    "error": "internal server error (see server log)",
                    "path": self.path,
                })
            except Exception:
                pass
        finally:
            LOG.info("POST %s -- %.1f ms", self.path, (time.monotonic() - t0) * 1000.0)

    # --- original GET and POST ---

    def _do_GET(self) -> None:
        u = urlparse(self.path)
        prefix = getattr(self.server, "url_prefix", "")
        req_path = _strip_optional_prefix(u.path, prefix)

        # try to read config
        try:
            cfg = self.server.cfg  # type: ignore[attr-defined]
        except Exception:
            cfg = {}

        # If the configured prefix is hit without trailing slash, redirect to slash form.
        if prefix and u.path == prefix:
            self.send_response(302)
            self.send_header("Location", prefix + "/")
            self.end_headers()
            return

        if req_path == "/":
            logo_url = _prefix_url(prefix, "/assets/cassette_logo_garble_v2.svg")
            form_defaults = _build_form_defaults(getattr(self.server, "cfg", {}) or {})
            ts_gen_cfg = _webui_timestamped_gen_out_cfg(getattr(self.server, "cfg", {}) or {})

            html = INDEX_HTML
            html = html.replace("__DTMF_PRESETS_JSON__", json.dumps(DTMF_PRESETS, ensure_ascii=False))
            html = html.replace("__URL_PREFIX_JSON__", json.dumps(prefix, ensure_ascii=False))
            html = html.replace("__CASSETTE_LOGO_URL__", logo_url)
            html = html.replace("__APP_VERSION__", pyhtml.escape(APP_VERSION))

            html = html.replace("__GEN_OUT_DEFAULT__", pyhtml.escape(form_defaults["gen_out"], quote=True))
            html = html.replace("__AN_REF_DEFAULT__", pyhtml.escape(form_defaults["an_ref"], quote=True))
            html = html.replace("__AN_OUTDIR_DEFAULT__", pyhtml.escape(form_defaults["an_outdir"], quote=True))
            html = html.replace("__GEN_OUT_TS_CFG_JSON__", json.dumps(ts_gen_cfg, ensure_ascii=False))

            # y-axis locking check
            html = html.replace(
                "__AN_LOCK_Y_AXIS_CHECKED__",
                "checked" if form_defaults["an_lock_y_axis"] else "",
            )
            html = html.replace(
                "__AN_PLOT_Y_MIN_DEFAULT__",
                pyhtml.escape(str(form_defaults["an_plot_y_min"]), quote=True),
            )
            html = html.replace(
                "__AN_PLOT_Y_MAX_DEFAULT__",
                pyhtml.escape(str(form_defaults["an_plot_y_max"]), quote=True),
            )
            html = html.replace(
                "__CMP_PLOT_Y_MIN_DEFAULT__",
                pyhtml.escape(str(form_defaults["an_plot_y_min"]), quote=True),
            )
            html = html.replace(
                "__CMP_PLOT_Y_MAX_DEFAULT__",
                pyhtml.escape(str(form_defaults["an_plot_y_max"]), quote=True),
            )

            self._text(200, html, "text/html; charset=utf-8")
            return

        if req_path == "/api/state":
            st = {
                "wavs": _list_files((".wav",)),
                "results": _list_result_dirs(),
            }
            self._json(200, st)
            return

        if req_path == "/api/runs":
            st = {
                "runs": _list_runs(),
            }
            self._json(200, st)
            return

        if req_path == "/api/run_summary":
            qs = parse_qs(u.query)
            run_dir = (qs.get("dir", [""])[0] or "").strip()
            try:
                summary = _load_summary_for_run_dir(run_dir, cfg)
                self._json(200, {"ok": True, "summary": summary})
            except Exception as e:
                self._json(400, {"ok": False, "error": str(e)})
            return

        if req_path == "/api/browse":
            qs = parse_qs(u.query)
            rel_dir = (qs.get("dir", [""])[0] or "").strip()
            mode = (qs.get("mode", ["file"])[0] or "file").strip().lower()
            exts = [e for e in qs.get("ext", []) if e is not None]
            q = (qs.get("q", [""])[0] or "").strip()
            scope = (qs.get("scope", [""])[0] or "").strip()

            if mode not in ("file", "dir"):
                self._json(400, {"error": "invalid mode (use file|dir)"})
                return

            try:
                out = _browse_dir(rel_dir, mode=mode, exts=exts, q=q, scope=scope, cfg=cfg)
                self._json(200, out)
            except Exception as e:
                self._json(400, {"error": str(e)})
            return

        if req_path == "/api/stat":
            qs = parse_qs(u.query)
            p = (qs.get("path", [""])[0] or "").strip()
            try:
                rel = _rel_to_root_checked(p)
                full = (ROOT / rel)
                if not full.exists() or not full.is_file():
                    self._json(404, {"error": "file not found"})
                    return
                st = full.stat()
                self._json(200, {
                    "path": rel,
                    "size": int(st.st_size),
                    "mtime": int(st.st_mtime),
                })
            except Exception as e:
                self._json(400, {"error": str(e)})
            return

        if req_path == "/file":
            qs = parse_qs(u.query)
            p = (qs.get("path", [""])[0] or "").strip()
            try:
                rel = _rel_to_root_checked(p)
                full = (ROOT / rel)
                if not full.exists() or not full.is_file():
                    self._json(404, {"error": "file not found"})
                    return

                self._send(200, full.read_bytes(), _guess_ctype(full))
                return
            except Exception as e:
                self._json(400, {"error": str(e)})
                return

        if req_path == "/download":
            qs = parse_qs(u.query)
            p = (qs.get("path", [""])[0] or "").strip()
            try:
                rel = _rel_to_root_checked(p)
                full = (ROOT / rel)
                if not full.exists() or not full.is_file():
                    self._json(404, {"error": "file not found"})
                    return

                self._send(
                    200,
                    full.read_bytes(),
                    _guess_ctype(full),
                    headers={
                        "Content-Disposition": _content_disposition_attachment(full.name),
                    },
                )
                return
            except Exception as e:
                self._json(400, {"error": str(e)})
                return

        if req_path.startswith("/assets/"):
            try:
                rel = _rel_to_project_root_checked(req_path.lstrip("/"))
                full = (ROOT / rel)
                if not full.exists() or not full.is_file():
                    self._json(404, {"error": "file not found"})
                    return

                self._send(200, full.read_bytes(), _guess_ctype(full))
                return
            except Exception as e:
                self._json(400, {"error": str(e)})
                return

        self._json(404, {"error": "not found"})

    def _read_body_bytes(self) -> bytes:
        n = int(self.headers.get("Content-Length", "0") or "0")
        if n < 0:
            raise ValueError("invalid Content-Length")
        if n > MAX_UPLOAD_BYTES:
            raise ValueError(f"upload too large (max {MAX_UPLOAD_BYTES} bytes)")
        return self.rfile.read(n) if n > 0 else b""

    def _read_json_body(self) -> dict:
        raw = self._read_body_bytes() or b"{}"
        try:
            obj = json.loads(raw.decode("utf-8"))
        except Exception:
            obj = {}
        return obj if isinstance(obj, dict) else {}

    def _do_POST(self) -> None:
        u = urlparse(self.path)
        prefix = getattr(self.server, "url_prefix", "")
        req_path = _strip_optional_prefix(u.path, prefix)

        try:
            cfg = self.server.cfg  # type: ignore[attr-defined]
        except Exception:
            cfg = {}

        # multipart upload route must read raw bytes before JSON parsing
        if req_path == "/api/upload_wav":
            try:
                qs = parse_qs(u.query)

                # check if traversal is allowed
                default_upload_dir = WEBUI_ROOT_REL if _webui_is_restricted() else "data"
                dir_in = (qs.get("dir", [default_upload_dir])[0] or default_upload_dir).strip()

                rel_dir = _ensure_outdir_rel(dir_in)
                target_dir = ROOT / rel_dir

                content_type = self.headers.get("Content-Type", "")
                raw = self._read_body_bytes()
                form = _parse_multipart_form(content_type, raw)

                file_obj = form.get("files", {}).get("file")
                if not isinstance(file_obj, dict):
                    raise ValueError("missing upload field 'file'")

                orig_name = str(file_obj.get("filename", "") or "").strip()
                safe_name = _safe_upload_filename(orig_name, allowed_exts=UPLOAD_EXTS)

                tmp_path = target_dir / f".upload-{time.time_ns()}.tmp"
                tmp_path.write_bytes(file_obj.get("data", b""))

                try:
                    wav_info = _validate_uploaded_wav(tmp_path)
                    final_path = _unique_file_path(target_dir, safe_name)
                    tmp_path.replace(final_path)
                except Exception:
                    with contextlib.suppress(Exception):
                        tmp_path.unlink()
                    raise

                rel_final = str(final_path.relative_to(ROOT))
                self._json(200, {
                    "ok": True,
                    "path": rel_final,
                    "name": final_path.name,
                    "wav_info": wav_info,
                })
                return
            except Exception as e:
                self._json(400, {"ok": False, "error": str(e)})
                return

        payload = self._read_json_body()

        if req_path == "/api/run_regen":
            try:
                run_dir = str(payload.get("dir", "") or "").strip()
                if not run_dir:
                    raise ValueError("dir is required")

                summary = _load_summary_for_run_dir(run_dir, cfg)
                run_meta = summary.get("run", {}) if isinstance(summary.get("run"), dict) else {}
                plot_cfg = _summary_plot_cfg(summary, cfg)

                ref = _require_existing_rel_wav(summary.get("ref"), label="Reference WAV")
                rec = _require_existing_rel_wav(summary.get("rec"), label="Recorded WAV")

                loopback = None
                loopback_raw = _opt_str(summary.get("loopback"))
                if loopback_raw:
                    loopback = _require_existing_rel_wav(loopback_raw, label="Loopback WAV")

                lock_y_axis = bool(payload.get("lock_y_axis", plot_cfg["lock_y_axis"]))
                plot_y_min = _coerce_float(payload.get("plot_y_min"), default=plot_cfg["plot_y_min"])
                plot_y_max = _coerce_float(payload.get("plot_y_max"), default=plot_cfg["plot_y_max"])

                if plot_y_min is None or plot_y_max is None:
                    raise ValueError("plot y-axis values must be numeric")
                if plot_y_min >= plot_y_max:
                    raise ValueError("plot_y_min must be smaller than plot_y_max")

                base_outdir_raw = _opt_str(run_meta.get("base_outdir")) or _build_form_defaults(cfg)["an_outdir"]
                outdir = _ensure_outdir_rel(base_outdir_raw)

                old_name = _opt_str(run_meta.get("name"))
                regen_name = f"{old_name} -- regen" if old_name else f"{Path(run_dir).name} -- regen"

                old_notes = _opt_str(run_meta.get("notes"))
                regen_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                regen_note = f"[regen from {run_dir} at {regen_ts}]"
                regen_notes = f"{old_notes}\n\n{regen_note}".strip() if old_notes else regen_note

                saved_replay = summary.get("_webui_replay_payload")

                if isinstance(saved_replay, dict):
                    regen_payload = dict(saved_replay)

                    # hard overrides for this new regenerated run
                    regen_payload["ref"] = ref
                    regen_payload["rec"] = rec
                    regen_payload["outdir"] = outdir
                    regen_payload["run_name"] = regen_name
                    regen_payload["run_notes"] = regen_notes
                    regen_payload["lock_y_axis"] = lock_y_axis
                    regen_payload["plot_y_min"] = plot_y_min
                    regen_payload["plot_y_max"] = plot_y_max

                    if loopback:
                        regen_payload["loopback"] = loopback
                    else:
                        regen_payload.pop("loopback", None)

                else:
                    # legacy fallback for older runs without _webui_replay_payload
                    regen_payload = {
                        "ref": ref,
                        "rec": rec,
                        "outdir": outdir,
                    }

                    _copy_present(regen_payload, summary, REGEN_COPY_KEYS)

                    regen_payload.update({
                        "run_name": regen_name,
                        "run_notes": regen_notes,
                        "channels": _channels_mode_from_summary(summary),
                        "marker_channel": _norm_ch_name(summary.get("marker_channel") or "mono"),
                        "marker_start": str(summary.get("marker_start") or ""),
                        "marker_end": str(summary.get("marker_end") or ""),
                        "lock_y_axis": lock_y_axis,
                        "plot_y_min": plot_y_min,
                        "plot_y_max": plot_y_max,
                        "lr_overlay": bool((summary.get("stereo_outputs") or {}).get("lr_overlay_png")),
                        "save_ir": _summary_has_impulse(summary),
                    })

                    if loopback:
                        regen_payload["loopback"] = loopback

                    ticks = summary.get("ticks", {})
                    if isinstance(ticks, dict) and bool(ticks.get("enabled", False)):
                        regen_payload["ticks"] = True
                        if ticks.get("sym") is not None:
                            regen_payload["tick_sym"] = ticks.get("sym")
                        if ticks.get("interval_s") is not None:
                            regen_payload["tick_interval_s"] = ticks.get("interval_s")
                        if ticks.get("tone_s") is not None:
                            regen_payload["tick_tone_s"] = ticks.get("tone_s")
                        if ticks.get("dbfs") is not None:
                            regen_payload["tick_dbfs"] = ticks.get("dbfs")
                        if ticks.get("offset_s") is not None:
                            regen_payload["tick_offset_s"] = ticks.get("offset_s")
                        if ticks.get("match_tol_s") is not None:
                            regen_payload["tick_match_tol_s"] = ticks.get("match_tol_s")
                        if ticks.get("min_matches") is not None:
                            regen_payload["tick_min_matches"] = ticks.get("min_matches")

                result = _api_analyze_from_payload(regen_payload, cfg)

                if isinstance(result.get("summary"), dict):
                    result["summary"]["regenerated_from_run"] = run_dir

                    new_run_dir = _opt_str(((result["summary"].get("run") or {}).get("outdir")))
                    if new_run_dir:
                        new_summary_path = ROOT / _rel_to_root_checked(new_run_dir) / "summary.json"
                        if new_summary_path.exists() and new_summary_path.is_file():
                            _write_json_atomic_preserve_mtime(new_summary_path, result["summary"])

                self._json(200, result)

            except Exception as e:
                msg = str(e)

                try:
                    obj = json.loads(msg)
                    if isinstance(obj, dict) and obj.get("ok") is False:
                        self._json(400, obj)
                        return
                except Exception:
                    pass

                self._json(400, {"ok": False, "error": msg})
            return
        
        LOG.info("POST %s -- parsed payload keys=%s", req_path, sorted(payload.keys()))

        if req_path == "/api/mkdir":
            try:
                path = str(payload.get("path", "") or "").strip()
                if not path:
                    raise ValueError("path is required")

                rel = _rel_to_root_checked(path)
                if rel in ("", "."):
                    raise ValueError("refusing to create project root")

                full = ROOT / rel
                full.mkdir(parents=True, exist_ok=True)

                self._json(200, {"ok": True, "path": rel})
            except Exception as e:
                self._json(400, {"error": str(e)})
            return

        if req_path == "/api/run_notes":
            try:
                run_dir = str(payload.get("dir", "") or "").strip()
                if not run_dir:
                    raise ValueError("dir is required")

                rel_dir = _rel_to_root_checked(run_dir)
                if rel_dir in ("", "."):
                    raise ValueError("invalid run dir")

                summary_path = (ROOT / rel_dir / "summary.json")
                if not summary_path.exists() or not summary_path.is_file():
                    raise ValueError("summary.json not found for run")

                obj = json.loads(summary_path.read_text(encoding="utf-8"))
                if not isinstance(obj, dict):
                    raise ValueError("summary.json is not an object")

                run = obj.get("run")
                if not isinstance(run, dict):
                    run = {}
                    obj["run"] = run

                notes_raw = str(payload.get("notes", "") or "")
                notes = notes_raw.replace("\r\n", "\n").replace("\r", "\n")

                if notes.strip() == "":
                    run.pop("notes", None)
                else:
                    run["notes"] = notes

                _write_json_atomic_preserve_mtime(summary_path, obj)

                self._json(200, {"ok": True, "summary": obj})
            except Exception as e:
                self._json(400, {"error": str(e)})
            return

        if req_path == "/api/compare":
            try:
                runs_in = payload.get("runs", None)
                if not isinstance(runs_in, list) or not runs_in:
                    raise ValueError("runs must be a non-empty list of run dirs")

                lock_y_axis = bool(payload.get("lock_y_axis", True))
                plot_y_min = _coerce_float(payload.get("plot_y_min"), default=None)
                plot_y_max = _coerce_float(payload.get("plot_y_max"), default=None)

                metric = str(payload.get("metric", "response") or "response").strip().lower()
                outdir = str(payload.get("outdir", "data/compare") or "data/compare").strip()

                tile_w = int(payload.get("tile_w", 1100) or 1100)
                tile_h = int(payload.get("tile_h", 650) or 650)
                dpi = int(payload.get("dpi", 150) or 150)

                ch_in = payload.get("channels", None)
                channels: list[str] = []
                if isinstance(ch_in, list) and ch_in:
                    channels = [_norm_ch_name(str(x)) for x in ch_in if str(x).strip()]
                else:
                    channels = ["L", "R"]

                runs: list[dict] = []
                for rd in runs_in:
                    rd_s = str(rd or "").strip()
                    if not rd_s:
                        continue
                    rd_rel = _rel_to_root_checked(rd_s)
                    summ = _load_summary_for_run_dir(rd_rel, cfg)
                    runs.append({
                        "dir": rd_rel,
                        "summary": summ,
                        "label": _run_label(rd_rel, summ),
                    })

                if not runs:
                    raise ValueError("no valid runs were provided")

                if len(channels) == 1 and channels[0].lower() == "auto":
                    seen = []
                    for r in runs:
                        per = r["summary"].get("per_channel", {})
                        if isinstance(per, dict):
                            for k in per.keys():
                                nk = _norm_ch_name(str(k))
                                if nk and nk not in seen:
                                    seen.append(nk)
                    channels = seen or ["L", "R"]

                out_png_rel, log_txt = _compare_render_grid(
                    runs=runs,
                    channels=channels,
                    metric=metric,
                    outdir=outdir,
                    tile_w=tile_w,
                    tile_h=tile_h,
                    dpi=dpi,
                    lock_y_axis=lock_y_axis,
                    plot_y_min=plot_y_min,
                    plot_y_max=plot_y_max,
                )

                self._json(200, {
                    "ok": True,
                    "out_png": out_png_rel,
                    "metric": metric,
                    "channels": channels,
                    "runs": [{"dir": r["dir"], "label": r["label"]} for r in runs],
                    "tile_w": tile_w,
                    "tile_h": tile_h,
                    "dpi": dpi,
                    "log": log_txt,
                    "lock_y_axis": lock_y_axis,
                    "plot_y_min": plot_y_min,
                    "plot_y_max": plot_y_max,                    
                })
            except Exception as e:
                self._json(400, {"ok": False, "error": str(e)})
            return

        if req_path == "/api/gen":
            try:
                form_defaults = _build_form_defaults(cfg)
                out_raw = _payload_str_or_default(payload, "out", form_defaults["gen_out"])
                out = _ensure_gen_out_rel(out_raw, cfg)

                overwrite_mode = str(payload.get("overwrite_mode", "") or "").strip().lower()
                if overwrite_mode not in ("", "overwrite", "autorename"):
                    raise ValueError("invalid overwrite_mode")

                out_full = ROOT / out

                if out_full.exists():
                    if _webui_warn_on_overwrite(cfg) and overwrite_mode == "":
                        self._json(409, {
                            "ok": False,
                            "needs_confirmation": True,
                            "error": "output file already exists",
                            "path": out,
                            "suggested_path": _timestamped_nonconflicting_rel(out, cfg=cfg),
                        })
                        return

                    if overwrite_mode == "autorename":
                        out = _timestamped_nonconflicting_rel(out, cfg=cfg)
                        out_full = ROOT / out

                out_full.parent.mkdir(parents=True, exist_ok=True)

                args = _make_args("gen", cfg, {"out": out})

                ok, log_txt, err = _run_cc_cmd(cc.cmd_gen, args)
                if not ok:
                    self._json(400, {"ok": False, "error": err, "log": log_txt})
                    return

                self._json(200, {"ok": True, "out": out, "log": log_txt})
            except Exception as e:
                self._json(400, {"ok": False, "error": str(e)})
            return

        if req_path == "/api/detect":
            try:
                wav = _rel_to_root_checked(str(payload.get("wav", "")))
                args = _make_args("detect", cfg, {"wav": wav, "json": True})

                ok, log_txt, err = _run_cc_cmd(cc.cmd_detect, args)
                if not ok:
                    self._json(400, {"ok": False, "error": err, "log": log_txt})
                    return

                s = (log_txt or "").strip()
                try:
                    result = json.loads(s or "{}")
                except Exception:
                    self._json(500, {
                        "ok": False,
                        "error": "detect did not output valid JSON",
                        "log": log_txt,
                    })
                    return

                self._json(200, {"ok": True, "result": result, "log": log_txt})
            except Exception as e:
                self._json(400, {"ok": False, "error": str(e)})
            return

        if req_path == "/api/analyze":
            try:
                result = _api_analyze_from_payload(payload, cfg)
                self._json(200, result)
            except Exception as e:
                msg = str(e)

                try:
                    obj = json.loads(msg)
                    if isinstance(obj, dict) and "ok" in obj and obj.get("ok") is False:
                        self._json(400, obj)
                        return
                except Exception:
                    pass

                self._json(400, {"ok": False, "error": msg})
            return

        self._json(404, {"error": "not found"})

def _format_bind_error(host: str, port: int, exc: OSError) -> str:
    url = f"http://{host}:{port}/"

    if exc.errno == errno.EADDRINUSE:
        return "\n".join([
            f"ERROR: could not start cassette-calibrator WebUI on {url}",
            "",
            f"Address already in use: {host}:{port}",
            "",
            "Common causes:",
            "  - another cassette-calibrator WebUI instance is already running",
            "  - another program is already using that port",
            "",
            "What to do:",
            "  - stop the other process using that port",
            f"  - start this WebUI on another port, e.g. ./webui.py --port {port + 1}",
            f"  - or change [webui].port in cassette_calibrator.toml",
        ])

    if exc.errno == errno.EACCES:
        return "\n".join([
            f"ERROR: could not start cassette-calibrator WebUI on {url}",
            "",
            f"Permission denied while binding to {host}:{port}",
            "",
            "What to do:",
            "  - use a port number above 1024",
            "  - or run with a host/port that your user can bind to",
        ])

    return "\n".join([
        f"ERROR: could not start cassette-calibrator WebUI on {url}",
        "",
        f"Bind failed: {exc}",
    ])

def main() -> int:
    ap = argparse.ArgumentParser(description="Local WebUI for cassette-calibrator (stdlib http.server; no web framework).")
    ap.add_argument("--config", default=None, help="TOML config path (default: auto-search)")
    ap.add_argument("--host", default=None, help="override bind host (default: from TOML or 127.0.0.1)")
    ap.add_argument("--port", type=int, default=None, help="override bind port (default: from TOML or 8765)")
    ap.add_argument("--no-browser", action="store_true", help="do not auto-open browser tab")
    args = ap.parse_args()

    os.chdir(ROOT)

    cfg = _load_cfg(args.config)

    global WEBUI_ROOT, WEBUI_ROOT_REL, WEBUI_RESTRICT_TO_ROOT_DIR, WEBUI_ALLOW_PROJECT_ROOT_ACCESS
    global DTMF_PRESETS

    w = _webui_cfg(cfg)
    WEBUI_RESTRICT_TO_ROOT_DIR = bool(w.get("restrict_to_root_dir", False))
    WEBUI_ALLOW_PROJECT_ROOT_ACCESS = bool(w.get("allow_project_root_access", True))
    WEBUI_ROOT, WEBUI_ROOT_REL = _resolve_webui_root_from_cfg(cfg)

    # Hard startup validation for forced generated-audio root
    gen_out_root = None
    gen_out_root_rel = None
    if bool(w.get("restrict_gen_out_to_dir", False)):
        gen_out_root, gen_out_root_rel = _ensure_gen_out_root_ready(cfg)

    try:
        DTMF_PRESETS = _build_dtmf_presets_from_cfg(cfg)
    except Exception as e:
        LOG.warning("DTMF preset build failed: %s -- using fallback presets", e)
        DTMF_PRESETS = dict(DTMF_PRESETS_FALLBACK)

    env_file_cfg = _load_simple_env(ROOT / ".env")

    host = str(
        args.host
        or os.environ.get("WEBUI_HOST")
        or env_file_cfg.get("WEBUI_HOST")
        or w.get("host", "127.0.0.1")
    )

    port = int(
        args.port
        or os.environ.get("WEBUI_PORT")
        or env_file_cfg.get("WEBUI_PORT")
        or w.get("port", 8765)
    )

    open_browser_default = bool(w.get("open_browser", True))

    open_browser = _parse_boolish(
        os.environ.get("WEBUI_OPEN_BROWSER")
        or env_file_cfg.get("WEBUI_OPEN_BROWSER"),
        open_browser_default,
    ) and (not args.no_browser)

    url_prefix = _normalize_url_prefix(
        os.environ.get("WEBUI_URL_PREFIX")
        or env_file_cfg.get("WEBUI_URL_PREFIX")
        or w.get("url_prefix", "")
    )

    if host not in ("127.0.0.1", "localhost"):
        print(f"[warn] Binding to '{host}' -- this WebUI is intended for local use.", file=sys.stderr)

    try:
        httpd = ThreadingHTTPServer((host, port), Handler)
    except OSError as e:
        msg = _format_bind_error(host, port, e)
        LOG.error(msg.replace("\n", " | "))
        print(msg, file=sys.stderr)
        return 2

    httpd.cfg = cfg  # type: ignore[attr-defined]
    httpd.url_prefix = url_prefix  # type: ignore[attr-defined]

    url = f"http://{host}:{port}/"
    open_url = f"http://{host}:{port}{url_prefix}/" if url_prefix else url

    term_width = shutil.get_terminal_size(fallback=(80, 24)).columns
    hr = "-" * max(20, term_width)

    print()
    print(hr)
    print(f"::: cassette-calibrator (WebUI) | v{APP_VERSION}")
    print(f"::: Project homepage: github.com/FlyingFathead/cassette-calibrator")

    if url_prefix:
        print(f"::: Server listening on: {url}")
        print(f"::: Using custom URL prefix: {url_prefix}/")
        print(f"::: Open the WebUI at: {open_url}")
    else:
        print(f"::: Server listening on: {url}")

    if _webui_is_restricted():
        print(f"::: WebUI file root restricted to: {WEBUI_ROOT_REL}/")
    else:
        print("::: WebUI file root: project root")

    if bool(w.get("restrict_gen_out_to_dir", False)):
        print(f"::: Generated test WAV root restricted to: {gen_out_root_rel}/")

    print(hr)
    print()

    if open_browser and host in ("127.0.0.1", "localhost"):
        try:
            import webbrowser
            webbrowser.open(open_url, new=1)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
