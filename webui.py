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
from functools import lru_cache
import io
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse, parse_qs

# Force a headless matplotlib backend BEFORE importing cassette_calibrator
os.environ.setdefault("MPLBACKEND", "Agg")

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
    return cc.load_toml_config(cfg_path)

def _webui_cfg(cfg: dict) -> dict:
    w = cfg.get("webui", {})
    return w if isinstance(w, dict) else {}

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
    button { margin-top: 10px; padding: 8px 12px; cursor: pointer; }
    pre { background: #111; color: #ddd; padding: 10px; border-radius: 8px; overflow:auto; max-height: 260px; }
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
    const msg = (j && j.error) ? j.error : t;
    throw new Error(msg);
  }
  return j;
}

async function apiGetJson(url) {
  const r = await fetch(url);
  const t = await r.text();
  let j = null;
  try { j = JSON.parse(t); } catch (e) {}
  if (!r.ok) {
    const msg = (j && j.error) ? j.error : t;
    throw new Error(msg);
  }
  return j;
}

function setLog(id, s) {
  document.getElementById(id).textContent = s || "";
}

function imgTag(path) {
  const url = "/file?path=" + encodeURIComponent(path);
  return `<div style="margin-top:10px"><div class="small">${esc(path)}</div><img src="${url}" /></div>`;
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

    const r = await api("/api/analyze", { ref, rec, loopback, outdir, run_name, run_notes });

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
  runsCancelNotes
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

    def do_GET(self) -> None:
        u = urlparse(self.path)
        if u.path == "/":
            self._text(200, INDEX_HTML, "text/html; charset=utf-8")
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

    def do_POST(self) -> None:
        u = urlparse(self.path)
        payload = self._read_json_body()

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

        if u.path == "/api/gen":
            try:
                out = _rel_to_root_checked(str(payload.get("out", "data/sweepcass.wav")))
                # Ensure parent directory exists for the output wav
                (ROOT / out).parent.mkdir(parents=True, exist_ok=True)

                args = _make_args("gen", cfg, {"out": out})

                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    cc.cmd_gen(args)

                self._json(200, {"ok": True, "out": out, "log": buf.getvalue()})
            except Exception as e:
                self._json(400, {"error": str(e)})
            return

        if u.path == "/api/detect":
            try:
                wav = _rel_to_root_checked(str(payload.get("wav", "")))
                args = _make_args("detect", cfg, {"wav": wav, "json": True})

                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    cc.cmd_detect(args)

                result = json.loads(buf.getvalue().strip() or "{}")
                self._json(200, {"ok": True, "result": result})
            except Exception as e:
                self._json(400, {"error": str(e)})
            return

        if u.path == "/api/analyze":
            try:
                ref = _rel_to_root_checked(str(payload.get("ref", "")))
                rec = _rel_to_root_checked(str(payload.get("rec", "")))
                loopback_in = str(payload.get("loopback", "") or "").strip()
                loopback = _rel_to_root_checked(loopback_in) if loopback_in else None
                outdir = _ensure_outdir_rel(str(payload.get("outdir", "data/cassette_results")))

                overrides = {"ref": ref, "rec": rec, "outdir": outdir}

                run_name = _opt_str(payload.get("run_name"))
                if run_name:
                    # match whatever cassette_calibrator's argparse uses
                    dfl = _cmd_argparse_defaults("analyze")
                    if "run_name" in dfl:
                        overrides["run_name"] = run_name
                    elif "name" in dfl:
                        overrides["name"] = run_name
                    else:
                        # last-resort: set both
                        overrides["run_name"] = run_name
                        overrides["name"] = run_name

                # Run notes (optional) -- pass to core so it can persist into summary.json
                run_notes = _opt_str(payload.get("run_notes"))
                if run_notes:
                    dfl = _cmd_argparse_defaults("analyze")
                    if "run_notes" in dfl:
                        overrides["run_notes"] = run_notes
                    elif "notes" in dfl:
                        overrides["notes"] = run_notes
                    else:
                        # last-resort: set both
                        overrides["run_notes"] = run_notes
                        overrides["notes"] = run_notes

                if loopback:
                    overrides["loopback"] = loopback

                args = _make_args("analyze", cfg, overrides)

                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    cc.cmd_analyze(args)

                base = ROOT / outdir
                # Find newest summary.json under base outdir (supports run-subdir mode)
                candidates = list(base.rglob("summary.json")) if base.exists() else []
                if not candidates:
                    raise RuntimeError("analyze finished but summary.json was not created")

                candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
                summary_path = candidates[0]

                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                self._json(200, {"ok": True, "summary": summary, "log": buf.getvalue()})

            except Exception as e:
                self._json(400, {"error": str(e)})
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
