#!/usr/bin/env python3
"""
webui.py

Local-only WebUI for cassette-calibrator.
- Binds to 127.0.0.1 by default (configurable via [webui] in cassette_calibrator.toml)
- Zero extra dependencies (stdlib only)
- Calls the "core program" by importing cassette_calibrator.py and invoking cmd_* funcs

Security posture:
- Rejects absolute paths and any ".." path traversal
- Only serves/browses files under the project directory
"""

from __future__ import annotations

import argparse
import contextlib
import html as pyhtml
from functools import lru_cache
import io
import json
import os
import sys
import time
import csv
import math
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse, parse_qs

# --- TOML-backed defaults + preset merging (WebUI "Default" == TOML base) ---
from typing import Any, Dict, Optional

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

# Force a headless matplotlib backend BEFORE importing cassette_calibrator
os.environ.setdefault("MPLBACKEND", "Agg")

# set up logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("cassette_calibrator.webui")

import cassette_calibrator as cc  # noqa: E402

ROOT = Path(__file__).resolve().parent

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
        return float(v)
    except Exception:
        return default

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
    cands.append(("combo_-10dbfs_-0.10thresh", c))

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

def _rel_to_root_checked(p: str) -> str:
    p = (p or "").strip()
    if not _is_rel_safe(p):
        raise ValueError("path must be relative (no absolute paths, no '..')")
    full = (ROOT / p).resolve()
    # Ensure it doesn't escape ROOT (resolve can still escape via symlinks)
    try:
        full.relative_to(ROOT)
    except Exception as e:
        raise ValueError("path escapes project root") from e
    return str(full.relative_to(ROOT))

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
    if cmd == "analyze" and "channels" in args:
        args["channels"] = cc.parse_channels(str(args["channels"]))

    if cmd == "detect" and "channel" in args:
        args["channel"] = cc._normalize_marker_channel(str(args["channel"]))

    if cmd == "analyze" and "marker_channel" in args:
        args["marker_channel"] = cc._normalize_marker_channel(str(args["marker_channel"]))

    return SimpleNamespace(**args)

def _walk_pruned(base: Path):
    # Prune IGNORE_DIRS during traversal
    for root, dirs, files in os.walk(base):
        root_p = Path(root)
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        yield root_p, dirs, files


def _list_files(exts: tuple[str, ...], max_items: int = 2000) -> list[str]:
    out: list[str] = []
    # Prefer data/ first
    for base in (ROOT / "data", ROOT):
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
    return sorted(set(out))


def _list_result_dirs(max_items: int = 2000) -> list[str]:
    out: list[str] = []
    for p in ROOT.rglob("summary.json"):
        try:
            rel = str(p.parent.relative_to(ROOT))
        except Exception:
            continue
        out.append(rel)
        if len(out) >= max_items:
            break
    return sorted(set(out))


def _list_runs(max_items: int = 500) -> list[dict]:
    """
    Return runs with lightweight metadata:
      {dir, mtime, name, created_at, label}
    Prefers scanning under data/ if it exists.
    """
    runs: list[dict] = []

    bases = []
    if (ROOT / "data").exists():
        bases.append(ROOT / "data")
    bases.append(ROOT)

    seen = set()

    for base in bases:
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

            # label: "name -- dir" or just dir
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


def _browse_dir(rel_dir: str, *, mode: str, exts: list[str] | None, q: str | None) -> dict:
    """
    mode: "file" or "dir"
    exts: list like [".wav", ".png"] to filter files; ignored for mode="dir"
    q: optional substring filter (case-insensitive) on name/path
    """
    rel_dir = (rel_dir or "").strip()
    if rel_dir == "":
        # Default to data/ when present
        rel_dir = "data" if (ROOT / "data").exists() else "."

    # Special-case "." for root browsing
    if rel_dir == ".":
        rel_checked = "."
        full = ROOT
    else:
        rel_checked = _rel_to_root_checked(rel_dir)
        full = (ROOT / rel_checked)

    if not full.exists() or not full.is_dir():
        raise ValueError(f"not a directory: {rel_dir}")

    # Normalize exts
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

    # Always include dirs; include files only if mode=="file"
    for p in sorted(full.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        name = p.name
        if name in IGNORE_DIRS:
            continue

        try:
            relp = str(p.relative_to(ROOT))
        except Exception:
            continue

        # Query filter
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
            if exts_norm:
                if p.suffix.lower() not in exts_norm:
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
    if full != ROOT:
        parent = str(full.parent.relative_to(ROOT))

    return {
        "cwd": rel_checked if rel_checked != "." else ".",
        "parent": parent,
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

def _load_summary_for_run_dir(rel_run_dir: str) -> dict:
    rel_run_dir = _rel_to_root_checked(rel_run_dir)
    p = (ROOT / rel_run_dir / "summary.json")
    if not p.exists() or not p.is_file():
        raise ValueError(f"summary.json not found for run: {rel_run_dir}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"summary.json is not an object for run: {rel_run_dir}")
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
) -> tuple[str, str]:
    """
    Returns (out_png_rel, log_text).
    If CSVs exist for the metric, re-plot with shared axes.
    If not, tile PNGs (axes matching not guaranteed) and log a warning.
    """
    metric = (metric or "").strip().lower()
    if metric not in _COMPARE_METRICS:
        raise ValueError(f"unknown metric: {metric}")

    metric_prefix, xlabel, ylabel, logx_default = _COMPARE_METRICS[metric]

    # Collect series and/or image tiles
    log_lines: list[str] = []
    series = {}  # (ch, run_idx) -> (xs, ys)
    tiles  = {}  # (ch, run_idx) -> png_rel
    have_any_series = False

    for j, r in enumerate(runs):
        rel_dir = r["dir"]
        summ = r["summary"]

        for ch in channels:
            csv_rel, png_rel = _find_metric_paths(summ, metric_prefix, ch)

            # prefer CSV (true axis matching)
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

            # fallback PNG tiling
            if png_rel:
                tiles[(ch, j)] = png_rel
            else:
                log_lines.append(f"[warn] missing outputs for {rel_dir} ch={ch} metric={metric_prefix}")

    # Output path
    outdir_rel = _ensure_outdir_rel(outdir)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    ch_tag = _slug("_".join(channels))
    out_name = f"compare-{metric}-{ch_tag}-{ts}.png"
    out_png_rel = str(Path(outdir_rel) / out_name)

    # Figure geometry
    nrows = max(1, len(channels))
    ncols = max(1, len(runs))
    fig_w_in = (tile_w * ncols) / max(1, dpi)
    fig_h_in = (tile_h * nrows) / max(1, dpi)

    import matplotlib.pyplot as plt  # uses Agg (already set)
    from matplotlib.image import imread as _imread

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w_in, fig_h_in),
        dpi=dpi,
        sharex=True if have_any_series else False,
        sharey=True if have_any_series else False,
    )

    # Normalize axes to 2D array-ish
    if nrows == 1 and ncols == 1:
        axes2 = [[axes]]
    elif nrows == 1:
        axes2 = [list(axes)]
    elif ncols == 1:
        axes2 = [[ax] for ax in axes]
    else:
        axes2 = [list(row) for row in axes]

    # Determine global axis l

INDEX_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>cassette-calibrator -- local WebUI</title>
  <style>

    body { font-family: system-ui, sans-serif; margin: 20px; }
    .row { display: flex; gap: 18px; flex-wrap: wrap; align-items: flex-start; }
    .card { border: 1px solid #ccc; border-radius: 10px; padding: 14px; min-width: 340px; max-width: 560px; flex: 1; }
    label { display:block; margin-top: 8px; font-size: 13px; color: #333; }
    input { width: 100%; padding: 6px; margin-top: 4px; }
    select { width: 100%; padding: 6px; margin-top: 4px; }
    details { margin-top: 8px; }
    summary { cursor: pointer; }
    button { margin-top: 10px; padding: 8px 12px; cursor: pointer; }
    pre {
      background: #111;
      color: #ddd;
      padding: 10px;
      border-radius: 8px;
      overflow: auto;
      max-height: 260px;

      /* make long errors readable */
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;

      /* let the user expand the log box */
      resize: vertical;
      min-height: 80px;
    }

    img { max-width: 100%; border-radius: 8px; border: 1px solid #222; }
    .small { font-size: 12px; color: #666; }
    .grid2 { display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: end; }

    /* Modal file browser */
    .modal-backdrop { position: fixed; inset: 0; background: rgba(0,0,0,0.55); display: none; z-index: 9999; }
    .modal { position: fixed; inset: 40px; background: #fff; border-radius: 12px; padding: 12px; display: none; z-index: 10000;
             box-shadow: 0 10px 40px rgba(0,0,0,0.35); }
    .modal-header { display: flex; gap: 10px; align-items: center; }
    .modal-header .path { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; color: #444; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .modal-body { margin-top: 10px; display: grid; grid-template-columns: 1fr; gap: 8px; height: calc(100% - 52px); }
    .modal-controls { display: flex; gap: 8px; align-items: center; }
    .modal-controls input { margin-top: 0; }
    .modal-list { border: 1px solid #ddd; border-radius: 10px; overflow: auto; padding: 8px; height: 100%; }
    .item { display: grid; grid-template-columns: 1fr auto auto; gap: 10px; padding: 6px 8px; border-radius: 8px; }
    .item:hover { background: #f3f3f3; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
    .pill { font-size: 11px; padding: 2px 8px; border-radius: 999px; border: 1px solid #ccc; color: #333; }
    .muted { color: #777; font-size: 12px; }
    .runhdr { margin-top: 10px; margin-bottom: 6px; }
    .runhdr .title { font-size: 22px; font-weight: 700; margin: 0 0 4px 0; }
    .runhdr .meta { font-size: 12px; color: #666; line-height: 1.35; }
    .runhdr .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }    
    
    textarea { width: 100%; padding: 6px; margin-top: 4px; resize: vertical; }  

    .toc { margin-top: 6px; font-size: 12px; color: #666; }
    .toc a { color: inherit; text-decoration: none; border-bottom: 1px dotted #999; }
    .toc a:hover { border-bottom-style: solid; }

    .block { border: 1px solid #ddd; border-radius: 10px; padding: 10px; margin-top: 10px; }
    .block h4 { margin: 0 0 8px 0; }

    .notes {
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      background: #f6f6f6;
      padding: 10px;
      border-radius: 8px;
      border: 1px solid #ddd;
    }

    .imgsec { margin-top: 14px; padding-top: 10px; border-top: 1px dashed #ccc; }
    .imgsec h4 { margin: 0 0 8px 0; }

    /* ---- Image viewer / clickable images ---- */

    .imgwrap { margin-top: 10px; }
    .imgmeta {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
    }
    .imglinks { display: flex; gap: 10px; align-items: center; }
    .imglink {
      font-size: 12px;
      color: #444;
      text-decoration: none;
      border-bottom: 1px dotted #999;
    }
    .imglink:hover { border-bottom-style: solid; }
    .imgbtn { padding: 4px 8px; font-size: 12px; margin-top: 0; }

    .imgthumb { cursor: zoom-in; }

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
      background: #fff;
      border-radius: 12px;
      padding: 10px;
      display: none;
      z-index: 20001;
      box-shadow: 0 10px 40px rgba(0,0,0,0.5);
    }
    .iv-head { display: flex; gap: 10px; align-items: center; }
    .iv-title {
      flex: 1;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 600;
      font-size: 13px;
    }
    .iv-actions { display: flex; gap: 8px; align-items: center; }
    .iv-meta { margin-top: 6px; font-size: 12px; color: #666; }

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

  </style>
</head>
<body>
  <h2>cassette-calibrator -- local WebUI</h2>
  <div class="small">
    Runs locally only. Uses your existing <code>cassette_calibrator.toml</code> defaults.
  </div>

  <div class="row">
    <div class="card">
      <h3>gen</h3>
      <label>Output WAV (relative path)</label>
      <div class="grid2">
        <input id="gen_out" value="data/sweepcass.wav" />
        <button onclick="openBrowser('gen_out', {mode:'file', exts:['.wav'], title:'Choose output WAV', allowNew:true})">Browse</button>
      </div>
      <button onclick="doGen()">Generate</button>
      <pre id="gen_log"></pre>
    </div>

    <div class="card">
      <h3>detect</h3>
      <label>WAV (relative path)</label>
      <div class="grid2">
        <input id="detect_wav" placeholder="data/recorded.wav" />
        <button onclick="openBrowser('detect_wav', {mode:'file', exts:['.wav'], title:'Choose WAV'})">Browse</button>
      </div>
      <button onclick="doDetect()">Detect markers</button>
      <pre id="detect_out"></pre>
    </div>

    <div class="card">
      <h3>analyze</h3>

      <label>Run name (optional)</label>
      <input id="an_name" placeholder="e.g.: this cassette => that deck" />

      <label>Run notes (optional)</label>
      <textarea id="an_notes" rows="7" placeholder="Long notes: deck, tape, settings, azimuth tweaks, Dolby/NR, weirdness..."></textarea>
      
      <label>Ref WAV</label>
      <div class="grid2">
        <input id="an_ref" placeholder="data/sweepcass.wav" />
        <button onclick="openBrowser('an_ref', {mode:'file', exts:['.wav'], title:'Choose ref WAV'})">Browse</button>
      </div>

      <label>Recorded WAV</label>
      <div class="grid2">
        <input id="an_rec" placeholder="data/recorded.wav" />
        <button onclick="openBrowser('an_rec', {mode:'file', exts:['.wav'], title:'Choose recorded WAV'})">Browse</button>
      </div>

      <div class="block">
        <h4>Markers (DTMF)</h4>

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
      
      <label>Loopback WAV (optional)</label>
      <div class="grid2">
        <input id="an_lb" placeholder="(none)" />
        <button onclick="openBrowser('an_lb', {mode:'file', exts:['.wav'], title:'Choose loopback WAV'})">Browse</button>
      </div>

      <label>Outdir (relative path)</label>
      <div class="grid2">
        <input id="an_outdir" value="data/cassette_results" />
        <button onclick="openBrowser('an_outdir', {mode:'dir', title:'Choose or create output directory', allowNew:true})">Browse</button>
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

      <h3>compare</h3>
      <div class="small">Build a compare list of runs, then render a single high-res grid PNG (shared axes when CSV exists).</div>

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

      <button onclick="doCompare()">Render compare grid</button>

      <pre id="cmp_log"></pre>
      <div id="cmp_img"></div>
    </div>

    <h3>runs</h3>
    <div class="small">Browse previous analysis runs (reads run name from summary.json).</div>

    <button onclick="refreshRuns()">Refresh list</button>
    <label>Pick a run</label>
    <select id="runs_sel" style="width:100%; padding:6px; margin-top:4px;"></select>

    <button onclick="loadSelectedRun()">Load run</button>
    <div id="runs_header"></div>

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
        <button onclick="mbUp()">Up</button>
        <button onclick="mbHome()">Home</button>
        <input id="mb_q" placeholder="filter..." oninput="mbRefresh()" />
        <span id="mb_hint" class="muted"></span>
      </div>
      <div id="mb_list" class="modal-list"></div>
    </div>
  </div>

<script>
const DTMF_PRESETS = __DTMF_PRESETS_JSON__;

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

let MB = {
  open: false,
  targetId: null,
  mode: "file",
  exts: [],
  cwd: "",
  title: "Browse",
  allowNew: false
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

async function refreshCompareRuns() {
  try {
    setLog("cmp_log", "loading runs...");
    const r = await apiGetJson("/api/runs");
    const avail = document.getElementById("cmp_avail");
    let html = "";
    for (const it of (r.runs || [])) {
      const label = (it.label || it.dir || "");
      const val = it.dir || "";
      html += `<option value="${esc(val)}">${esc(label)}</option>`;
    }
    avail.innerHTML = html || "<option value=''>no runs found</option>";
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

    const r = await api("/api/compare", { runs, metric, channels, tile_w, tile_h, dpi });

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

function imgUrl(path) {
  return "/file?path=" + encodeURIComponent(path);
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
  if (e.key === "Escape") closeImgViewer();
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
  return (s || "").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;");
}

function jsq(s) {
  // returns a quoted JS string literal, safe to embed inside onclick=...
  return JSON.stringify(String(s || ""));
}

async function api(path, payload) {
  const r = await fetch(path, {
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

    // If backend returned log text, append it under the error headline.
    const full = log ? (msg + extra + "\n\n--- log ---\n" + log) : (msg + extra);
    throw new Error(full);
  }

  return j;
}

async function apiGetJson(url) {
  const r = await fetch(url);
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

/* -------- modal browser -------- */

function openBrowser(targetId, opts) {
  MB.open = true;
  MB.targetId = targetId;
  MB.mode = (opts && opts.mode) ? opts.mode : "file";
  MB.exts = (opts && opts.exts) ? opts.exts : [];
  MB.title = (opts && opts.title) ? opts.title : "Browse";
  MB.allowNew = !!(opts && opts.allowNew);

  // Start in data/ if it exists server-side; server defaults to data anyway when dir=""
  MB.cwd = "";

  document.getElementById("mb_title").textContent = MB.title;
  document.getElementById("mb_q").value = "";
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

  const st = await apiGetJson("/api/browse?" + params.toString());
  MB.cwd = st.cwd;
  document.getElementById("mb_path").textContent = st.cwd;

  const list = document.getElementById("mb_list");
  let html = "";

  // helper: join paths without making absolute
  function joinPath(base, leaf) {
    base = String(base || "");
    leaf = String(leaf || "");
    while (base.endsWith("/")) base = base.slice(0, -1);
    while (leaf.startsWith("/")) leaf = leaf.slice(1);
    if (!base || base === ".") return leaf;
    if (!leaf) return base;
    return base + "/" + leaf;
  }

  // Create-directory row (only when browsing dirs and allowNew=true)
  if (MB.allowNew && MB.mode === "dir") {
    html += "<div class='item'>"
         + "<div><span class='pill'>new</span> "
         + "<input id='mb_newdir' placeholder='e.g. cassette_results/run_001' style='margin-top:0; width:100%;' /></div>"
         + "<div class='muted'>in <span class='mono'>" + esc(st.cwd) + "</span></div>"
         + "<div><button onclick='mbCreateDir()'>Create</button></div>"
         + "</div>";
  }

  // Optional "use typed value" for allowNew outputs (gen out file)
  if (MB.allowNew && MB.mode === "file") {
    const current = document.getElementById(MB.targetId).value || "";
    html += "<div class='item'><div><span class='pill'>new</span> <span class='mono'>" + esc(current) + "</span></div>"
         + "<div class='muted'>use current</div>"
         + "<div><button onclick='mbPickValue()'>Use</button></div></div>";
  }

  for (const it of st.entries) {
    const pill = it.is_dir ? "<span class='pill'>dir</span>" : "<span class='pill'>file</span>";
    const meta = it.is_dir ? "" : ("<span class='muted'>" + (it.size || 0) + " bytes</span>");
    const btn = it.is_dir
      ? `<button onclick='mbEnter(${jsq(it.path)})'>Open</button>`
      : `<button onclick='mbPick(${jsq(it.path)})'>Pick</button>`;

    html += "<div class='item'>"
         + "<div>" + pill + " <span class='mono'>" + esc(it.path) + "</span></div>"
         + "<div>" + meta + "</div>"
         + "<div>" + btn + "</div>"
         + "</div>";
  }

  if (!html) html = "<div class='muted'>No entries.</div>";
  list.innerHTML = html;
}

function mbEnter(path) {
  MB.cwd = path;
  mbRefresh();
}

async function mbUp() {
  // Ask server for current, then use parent
  const params = new URLSearchParams();
  if (MB.cwd) params.set("dir", MB.cwd);
  params.set("mode", MB.mode);
  for (const e of MB.exts) params.append("ext", e);

  const st = await apiGetJson("/api/browse?" + params.toString());
  if (st.parent) {
    MB.cwd = st.parent;
  } else {
    MB.cwd = ".";
  }
  mbRefresh();
}

function mbHome() {
  MB.cwd = "";
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

    // Create relative to current browse directory
    const base = String(MB.cwd || "").trim();

    // normalize: strip leading/trailing forward slashes
    let name = String(nameRaw);
    while (name.startsWith("/")) name = name.slice(1);
    while (name.endsWith("/")) name = name.slice(0, -1);
    name = name.trim();
    if (!name) return;

    // basic safety: reject traversal or absolute-ish junk
    if (name.includes("..")) {
      alert("ERROR: '..' is not allowed");
      return;
    }

    let path = name;

    if (base && base !== ".") {
      let baseClean = String(base);
      while (baseClean.endsWith("/")) baseClean = baseClean.slice(0, -1);

      // If base came back as ".", treat as root
      if (baseClean === ".") baseClean = "";

      path = baseClean ? (baseClean + "/" + name) : name;
    }

    // one more sanity pass: collapse accidental double slashes
    while (path.includes("//")) path = path.replaceAll("//", "/");

    const r = await api("/api/mkdir", { path });

    // Pick it into the target input and close
    document.getElementById(MB.targetId).value = (r && r.path) ? r.path : path;
    closeBrowser();
  } catch (e) {
    alert("ERROR: " + e.message);
  }
}


/* -------- actions -------- */

async function doGen() {
  try {
    setLog("gen_log", "running...");
    const out = document.getElementById("gen_out").value;
    const r = await api("/api/gen", { out });
    setLog("gen_log", r.log || JSON.stringify(r, null, 2));
  } catch (e) {
    setLog("gen_log", "ERROR: " + e.message);
  }
}

async function doDetect() {
  try {
    setLog("detect_out", "running...");
    const wav = document.getElementById("detect_wav").value;
    const r = await api("/api/detect", { wav });
    setLog("detect_out", JSON.stringify(r.result, null, 2));
  } catch (e) {
    setLog("detect_out", "ERROR: " + e.message);
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

    const dtmf = readDtmfCfg();
    const r = await api("/api/analyze", { ref, rec, loopback, outdir, run_name, run_notes, dtmf });

    // Header (includes notes block if summary.run.notes exists)
    document.getElementById("an_header").innerHTML =
      renderRunHeader(r.summary, "Most recent run", "an");

    // JSON anchor wrapper is id="an_json" (see HTML change above)
    setLog("an_log", (r.log || "") + "\n\n" + JSON.stringify(r.summary, null, 2));

    // Images with per-channel anchors + blocks
    document.getElementById("an_imgs").innerHTML =
      renderImagesWithAnchors(r.summary, "an");

  } catch (e) {
    setLog("an_log", "ERROR: " + e.message);
  }
}

async function refreshRuns() {
  try {
    // reset context + editor UI
    RUNS_CTX.dir = "";
    RUNS_CTX.summary = null;
    document.getElementById("runs_notes_ctl").style.display = "none";
    document.getElementById("runs_notes_status").textContent = "";
    document.getElementById("runs_notes_edit_btn").style.display = "none";

    // clear stale displayed run (header + images) right away
    document.getElementById("runs_header").innerHTML = "";
    document.getElementById("runs_imgs").innerHTML = "";
    setLog("runs_log", "loading...");

    const r = await apiGetJson("/api/runs");
    const sel = document.getElementById("runs_sel");

    let html = "";
    for (const it of (r.runs || [])) {
      const label = (it.label || it.dir || "");
      const val = it.dir || "";
      html += `<option value="${esc(val)}">${esc(label)}</option>`;
    }
    sel.innerHTML = html || "<option value=''>no runs found</option>";

    setLog("runs_log", JSON.stringify(r, null, 2));
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

    const sumPath = dir.replace(/\/+$/,"") + "/summary.json";
    const summary = await apiGetJson("/file?path=" + encodeURIComponent(sumPath));

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

// Init defaults on load
applyDtmfPreset(document.getElementById("dtmf_preset").value);

// Make inline onclick= handlers work no matter what scope rules apply
Object.assign(window, {
  doGen,
  doDetect,
  doAnalyze,
  refreshRuns,
  loadSelectedRun,
  openBrowser,
  closeBrowser,
  mbUp,
  mbHome,
  mbEnter,
  mbPick,
  mbPickValue,
  mbCreateDir,
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
  doCompare
});

console.log("webui script loaded; handlers exported to window");

</script>
</body>
</html>
"""

class Handler(BaseHTTPRequestHandler):
    server_version = "cassette-calibrator-webui/0.2"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
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
        if u.path == "/":
            html = INDEX_HTML.replace("__DTMF_PRESETS_JSON__", json.dumps(DTMF_PRESETS, ensure_ascii=False))
            self._text(200, html, "text/html; charset=utf-8")
            return

        if u.path == "/api/state":
            st = {
                "wavs": _list_files((".wav",)),
                "results": _list_result_dirs(),
            }
            self._json(200, st)
            return

        if u.path == "/api/runs":
            st = {
                "runs": _list_runs(),
            }
            self._json(200, st)
            return

        if u.path == "/api/browse":
            qs = parse_qs(u.query)
            rel_dir = (qs.get("dir", [""])[0] or "").strip()
            mode = (qs.get("mode", ["file"])[0] or "file").strip().lower()
            exts = [e for e in qs.get("ext", []) if e is not None]
            q = (qs.get("q", [""])[0] or "").strip()

            if mode not in ("file", "dir"):
                self._json(400, {"error": "invalid mode (use file|dir)"})
                return

            try:
                out = _browse_dir(rel_dir, mode=mode, exts=exts, q=q)
                self._json(200, out)
            except Exception as e:
                self._json(400, {"error": str(e)})
            return

        if u.path == "/api/stat":
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

        if u.path == "/file":
            qs = parse_qs(u.query)
            p = (qs.get("path", [""])[0] or "").strip()
            try:
                rel = _rel_to_root_checked(p)
                full = (ROOT / rel)
                if not full.exists() or not full.is_file():
                    self._json(404, {"error": "file not found"})
                    return

                suf = full.suffix.lower()
                if suf == ".png":
                    self._send(200, full.read_bytes(), "image/png")
                elif suf == ".csv":
                    self._send(200, full.read_bytes(), "text/csv; charset=utf-8")
                elif suf == ".json":
                    self._send(200, full.read_bytes(), "application/json; charset=utf-8")
                elif suf == ".wav":
                    self._send(200, full.read_bytes(), "audio/wav")
                else:
                    self._send(200, full.read_bytes(), "application/octet-stream")
                return
            except Exception as e:
                self._json(400, {"error": str(e)})
                return

        self._json(404, {"error": "not found"})

    def _read_json_body(self) -> dict:
        n = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(n) if n > 0 else b"{}"
        try:
            obj = json.loads(raw.decode("utf-8"))
        except Exception:
            obj = {}
        return obj if isinstance(obj, dict) else {}

    def _do_POST(self) -> None:
        u = urlparse(self.path)
        payload = self._read_json_body()

        LOG.info("POST %s -- parsed payload keys=%s", u.path, sorted(payload.keys()))

        try:
            cfg = self.server.cfg  # type: ignore[attr-defined]
        except Exception:
            cfg = {}

        if u.path == "/api/mkdir":
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

        if u.path == "/api/run_notes":
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
                # normalize line endings
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

        if u.path == "/api/compare":
            try:
                runs_in = payload.get("runs", None)
                if not isinstance(runs_in, list) or not runs_in:
                    raise ValueError("runs must be a non-empty list of run dirs")

                metric = str(payload.get("metric", "response") or "response").strip().lower()
                outdir = str(payload.get("outdir", "data/compare") or "data/compare").strip()

                # UI controls for resolution
                tile_w = int(payload.get("tile_w", 1100) or 1100)
                tile_h = int(payload.get("tile_h", 650) or 650)
                dpi = int(payload.get("dpi", 150) or 150)

                # channels selection
                ch_in = payload.get("channels", None)
                channels: list[str] = []
                if isinstance(ch_in, list) and ch_in:
                    channels = [_norm_ch_name(str(x)) for x in ch_in if str(x).strip()]
                else:
                    # default: union of channels found in first run summary
                    channels = ["L", "R"]

                # load summaries
                runs: list[dict] = []
                for rd in runs_in:
                    rd_s = str(rd or "").strip()
                    if not rd_s:
                        continue
                    rd_rel = _rel_to_root_checked(rd_s)
                    summ = _load_summary_for_run_dir(rd_rel)
                    runs.append({
                        "dir": rd_rel,
                        "summary": summ,
                        "label": _run_label(rd_rel, summ),
                    })

                if not runs:
                    raise ValueError("no valid runs were provided")

                # if channels == ["auto"], use union across runs
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
                })
            except Exception as e:
                self._json(400, {"ok": False, "error": str(e)})
            return

        if u.path == "/api/gen":
            try:
                out = _rel_to_root_checked(str(payload.get("out", "data/sweepcass.wav")))
                (ROOT / out).parent.mkdir(parents=True, exist_ok=True)

                args = _make_args("gen", cfg, {"out": out})

                ok, log_txt, err = _run_cc_cmd(cc.cmd_gen, args)
                if not ok:
                    self._json(400, {"ok": False, "error": err, "log": log_txt})
                    return

                self._json(200, {"ok": True, "out": out, "log": log_txt})
            except Exception as e:
                self._json(400, {"ok": False, "error": str(e)})
            return

        if u.path == "/api/detect":
            try:
                wav = _rel_to_root_checked(str(payload.get("wav", "")))
                args = _make_args("detect", cfg, {"wav": wav, "json": True})

                ok, log_txt, err = _run_cc_cmd(cc.cmd_detect, args)
                if not ok:
                    # cc.cmd_detect may have printed something; return it to UI
                    self._json(400, {"ok": False, "error": err, "log": log_txt})
                    return

                # On success, stdout should be JSON (because json=True)
                s = (log_txt or "").strip()
                try:
                    result = json.loads(s or "{}")
                except Exception:
                    # Defensive: if detect printed non-JSON despite json=True
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

        if u.path == "/api/analyze":
            # Always have a buffer-backed log, even when we fail early
            try:
                ref = _rel_to_root_checked(str(payload.get("ref", "")))
                rec = _rel_to_root_checked(str(payload.get("rec", "")))

                loopback_in = str(payload.get("loopback", "") or "").strip()
                loopback = _rel_to_root_checked(loopback_in) if loopback_in else None

                outdir = _ensure_outdir_rel(str(payload.get("outdir", "data/cassette_results")))

                overrides = {"ref": ref, "rec": rec, "outdir": outdir}

                # Fetch defaults ONCE
                dfl = _cmd_argparse_defaults("analyze")

                # DTMF / marker tuning from UI (best effort)
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

                # run name
                run_name = _opt_str(payload.get("run_name"))
                if run_name:
                    if "run_name" in dfl:
                        overrides["run_name"] = run_name
                    elif "name" in dfl:
                        overrides["name"] = run_name
                    else:
                        overrides["run_name"] = run_name
                        overrides["name"] = run_name

                # run notes
                run_notes = payload.get("run_notes", None)
                if isinstance(run_notes, str):
                    # preserve content but normalize line endings; ignore if only whitespace
                    rn = run_notes.replace("\r\n", "\n").replace("\r", "\n")
                    if rn.strip():
                        if "run_notes" in dfl:
                            overrides["run_notes"] = rn
                        elif "notes" in dfl:
                            overrides["notes"] = rn
                        else:
                            overrides["run_notes"] = rn
                            overrides["notes"] = rn

                if loopback:
                    overrides["loopback"] = loopback

                # OPTIONAL: let UI pass through marker-finding knobs if you add fields later
                # (safe: only apply if the argparse dest exists)
                def _maybe_float(api_key: str, *dest_names: str):
                    v = payload.get(api_key, None)
                    if v is None or v == "":
                        return
                    try:
                        fv = float(v)
                    except Exception:
                        return
                    for dn in dest_names:
                        if dn in dfl:
                            overrides[dn] = fv
                            return

                def _maybe_int(api_key: str, *dest_names: str):
                    v = payload.get(api_key, None)
                    if v is None or v == "":
                        return
                    try:
                        iv = int(v)
                    except Exception:
                        return
                    for dn in dest_names:
                        if dn in dfl:
                            overrides[dn] = iv
                            return

                # common knobs you referenced in the SystemExit hint:
                _maybe_float("min_dbfs", "min_dbfs")
                _maybe_float("thresh", "thresh")
                # (and if your core uses different names, add aliases here)

                LOG.info("analyze: ref=%s rec=%s loopback=%s outdir=%s", ref, rec, loopback, outdir)

                attempts = []
                used_label = None
                used_cfg = None

                if dtmf_autotune:
                    cand_list = _dtmf_candidate_configs(base_dtmf_cfg, dtmf_preset)
                else:
                    # single attempt: preset + payload overrides
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
                    # per-attempt overrides copy
                    ov = dict(overrides)

                    # apply only args that actually exist in argparse defaults
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
                        break

                    # If not autotuning, stop immediately
                    if not dtmf_autotune:
                        break

                    # If it failed for reasons NOT related to markers, don't spin the ladder
                    if not _is_marker_failure(err, log_txt):
                        break

                if not ok:
                    self._json(400, {
                        "ok": False,
                        "error": err,
                        "log": log_txt,
                        "dtmf": {
                            "preset": dtmf_preset,
                            "autotune": dtmf_autotune,
                            "attempts": attempts,
                        },
                    })
                    return

                base = ROOT / outdir
                candidates = list(base.rglob("summary.json")) if base.exists() else []
                if not candidates:
                    self._json(500, {"ok": False, "error": "analyze succeeded but summary.json was not created", "log": log_txt})
                    return

                candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
                summary_path = candidates[0]
                summary = json.loads(summary_path.read_text(encoding="utf-8"))

                self._json(200, {
                    "ok": True,
                    "summary": summary,
                    "log": log_txt,
                    "dtmf": {
                        "preset": dtmf_preset,
                        "used": used_label,
                        "cfg": used_cfg,
                        "attempts": attempts,
                    },
                })

            except Exception as e:
                self._json(400, {"ok": False, "error": str(e)})
            return


        self._json(404, {"error": "not found"})


def main() -> int:
    ap = argparse.ArgumentParser(description="Local WebUI for cassette-calibrator (stdlib only).")
    ap.add_argument("--config", default=None, help="TOML config path (default: auto-search)")
    ap.add_argument("--host", default=None, help="override bind host (default: from TOML or 127.0.0.1)")
    ap.add_argument("--port", type=int, default=None, help="override bind port (default: from TOML or 8765)")
    ap.add_argument("--no-browser", action="store_true", help="do not auto-open browser tab")
    args = ap.parse_args()

    os.chdir(ROOT)

    cfg = _load_cfg(args.config)

    global DTMF_PRESETS
    try:
        DTMF_PRESETS = _build_dtmf_presets_from_cfg(cfg)
    except Exception as e:
        LOG.warning("DTMF preset build failed: %s -- using fallback presets", e)
        DTMF_PRESETS = dict(DTMF_PRESETS_FALLBACK)

    w = _webui_cfg(cfg)

    host = str(args.host or w.get("host", "127.0.0.1"))
    port = int(args.port or w.get("port", 8765))
    open_browser = bool(w.get("open_browser", True)) and (not args.no_browser)

    if host not in ("127.0.0.1", "localhost"):
        print(f"[warn] Binding to '{host}' -- this WebUI is intended for local use.", file=sys.stderr)

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.cfg = cfg  # type: ignore[attr-defined]

    url = f"http://{host}:{port}/"
    print(f"cassette-calibrator WebUI listening on: {url}")
    if open_browser and host in ("127.0.0.1", "localhost"):
        try:
            import webbrowser
            webbrowser.open(url, new=1)
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
