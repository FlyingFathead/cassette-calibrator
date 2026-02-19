# cassette-calibrator

A CLI tool with a separate easy-to-use webUI for measuring and calibrating a compact cassette recording/playback chain using an audio test program and automated alignment. Also comes with an optional local WebUI for ease of use.

It generates a cassette-friendly measurement WAV with user-configurable DTMF markers, then analyzes a recorded playback capture to estimate the chain's magnitude response (and optional loopback-subtracted response), plus an SNR estimate.

## What this does

- Generates a print-to-tape WAV containing:
  - DTMF start marker (audio timecode anchor)
  - Dedicated silence window for noise-floor measurement
  - DTMF countdown (default on; mostly for humans, alignment uses start/end markers)
  - 1 kHz reference tone for setting record level
  - ESS log sweep (default 20 Hz -> 20 kHz)
  - Optional periodic DTMF "ticks" embedded during the sweep (for non-linear drift correction)
  - DTMF end marker

- Analyzes a recorded capture by:
  - Detecting start/end markers automatically (no manual alignment)
  - Extracting the sweep region based on known layout
  - Applying a linear time-warp based on marker-to-marker drift (transport speed mismatch)
  - Optional tick-based piecewise time-warp (handles non-linear wow/flutter better than a single linear warp)
  - Optional fine alignment via correlation
  - ESS deconvolution -> impulse response -> magnitude response
  - Exporting plots + CSV + summary JSON
  - Estimating SNR from the dedicated silence window vs the mid-tone RMS

## Install

Python 3.9+ recommended. Tested on Python 3.12.x.

```bash
python3 -m pip install -r requirements.txt
```

Notes:

* `soundfile` is currently listed in `requirements.txt` (optional / future-proofing for better audio I/O).
* If you want minimal deps, remove `soundfile` from `requirements.txt` unless you actually use it in code.

## Local WebUI (optional)

A local-only, stdlib-only WebUI is included as `webui.py`. It calls into the same `cassette_calibrator.py` command handlers and uses your `cassette_calibrator.toml` defaults.

Launch:

```bash
python3 webui.py
# options:
# python3 webui.py --host 127.0.0.1 --port 8765
# python3 webui.py --no-browser
```

Config (optional) via `cassette_calibrator.toml`:

```toml
[webui]
host = "127.0.0.1"
port = 8765
open_browser = true
```

### WebUI features

* Run `gen`, `detect`, `analyze` with the same defaults you use from CLI
* Browse files safely (relative paths only) and create output directories
* Browse prior runs by scanning for `summary.json`
* View plots/images referenced in `summary.json`
* **Edit run notes for an existing run** (Save or Cancel) without re-running analysis

### Editing run notes (WebUI)

When you load a run under the "runs" card, you can click **Edit notes** to change the run notes stored in that run's `summary.json`.

Behavior:

* Saves to `<run-dir>/summary.json` under `run.notes`
* Clearing notes (blank) removes `run.notes`
* Uses an atomic write (temp file + replace)
* Preserves `mtime` so the run list ordering does not change just because notes were edited

### Path rules (important)

**Important:** the WebUI only accepts **relative paths under the project directory** (no absolute paths, no "..").
If you paste `/home/you/recorded.wav`, you'll get:

`ERROR: path must be relative (no absolute paths, no '..')`

Fix: put the file under the repo (recommended `data/`) and use `data/recorded.wav` (or use the Browse button which returns relative paths).

Security posture:

* Binds to `127.0.0.1` by default.
* Rejects absolute paths and any `..` traversal.
* Only serves/browses files under the project directory.
* Can write only within the project directory (for output dirs and run note edits).

## Workflow

### 1) Generate the test WAV

```bash
python3 cassette_calibrator.py gen --out sweepcass.wav
# disable countdown if you want:
# python3 cassette_calibrator.py gen --out sweepcass.wav --no-countdown
```

Recommended cassette-friendly settings (more buffer for sloppy transport + better noise measurement):

```bash
python3 cassette_calibrator.py gen --out sweepcass.wav --pre-s 3 --noisewin-s 2
```

### 2) Print to tape and capture playback

**Before you start (quick sanity):**

* Clean heads + capstan + pinch roller.
* Set the deck’s tape type correctly (Type I/II/IV).
* Disable anything "helpful" in the chain: Dolby/NR, EQ, AGC, enhancers, expander, etc. (deck + interface mixer + OS).

**A) Record pass (interface -> deck -> tape)**

* Connect **interface line out** (L/R) -> deck **line in / aux in** (do **not** use mic inputs).
* Play `sweepcass.wav` from a player that doesn’t resample or add DSP.

  * Avoid OS "audio enhancements" at all costs! Disable **all** spatial sound, EQ, loudness normalization and other audio enhancements from your audio host device.
  * Don’t let system sounds mix into the output.
* Start deck recording, then start playback of `sweepcass.wav` from the beginning.
* When the file reaches the **1 kHz reference tone**, set the deck’s record level:

  * Aim for a safe, repeatable level (avoid clipping/redlining).
  * The goal is consistency, not "as hot as possible".
* Rewind and do a clean, uninterrupted record pass of the full file (including pre/post silence).

**B) Playback capture (deck -> interface -> WAV)**

* Connect deck **line out / rec out** -> **interface line in**.
* Capture as `recorded.wav` at the **same sample rate** as generated (default 44.1 kHz).
* Record at **24-bit** if you can; leave headroom (avoid clipping).
* Start recording first, then start cassette playback; stop after the end marker + post silence.

Avoid any AGC/NR/enhancers in the interface path.

### 3) (Optional) sanity-check marker detection

```bash
python3 cassette_calibrator.py detect --wav recorded.wav
# or machine-readable:
python3 cassette_calibrator.py detect --wav recorded.wav --json
```

### 4) Analyze and export results

For stereo:

```bash
python3 cassette_calibrator.py analyze --ref sweepcass.wav --rec recorded.wav --outdir results --fine-align --channels stereo
# or print summary JSON:
python3 cassette_calibrator.py analyze --ref sweepcass.wav --rec recorded.wav --outdir results --fine-align --channels stereo --json
```

For mono:

```bash
python3 cassette_calibrator.py analyze --ref sweepcass.wav --rec recorded.wav --outdir results --fine-align --channels mono
```

Run name/notes:

* You can set a run name and run notes via CLI flags (see `python3 cassette_calibrator.py analyze --help`) and/or via the WebUI.
* WebUI can also edit notes later for an existing run by updating that run's `summary.json`.

### 4b) Optional: ticks mode (non-linear drift correction)

Cassette transports can drift non-linearly (wow/flutter). The default method uses a single linear warp based on the start/end markers, which corrects overall speed mismatch but can't fully fix curvature within the sweep.

Ticks mode embeds short periodic DTMF symbols inside the sweep, then uses them to build a piecewise time warp. Use it if you see "wavy" response curves or inconsistent alignment between runs.

Generate with ticks:

```bash
python3 cassette_calibrator.py gen --out sweepcass.wav --ticks
```

Analyze with ticks:

```bash
python3 cassette_calibrator.py analyze --ref sweepcass.wav --rec recorded.wav --outdir results --ticks
```

If tick matching fails (falls back to linear warp), try:

* raise tick level at generation: `--tick-dbfs -16` (or louder)
* loosen detection: `--thresh 5.0` or `--min-dbfs -60`
* widen matching tolerance: `--tick-match-tol-s 0.5`

### 5) Optional: subtract interface coloration with a loopback capture

Make a loopback capture by patching interface line out -> line in (no cassette), while playing `sweepcass.wav`.

```bash
python3 cassette_calibrator.py analyze \
  --ref sweepcass.wav \
  --rec recorded.wav \
  --loopback loopback.wav \
  --outdir results \
  --fine-align
```

This produces `difference.png` ("cassette chain minus loopback").

## Outputs

In `--outdir`:

* If analyzing **mono** (`--channels mono`):

  * `response.png` -- smoothed magnitude response (log frequency)
  * `response.csv` -- raw + smoothed response data
  * `summary.json` -- marker times, drift ratio, SNR estimate, settings used (plus optional run name/notes metadata)
  * `difference.png` -- only if `--loopback` is provided
  * `impulse.png` -- only if `--save-ir` is used

* If analyzing **stereo** (default, `--channels stereo`):

  * `response_l.png`, `response_r.png`
  * `response_l.csv`, `response_r.csv`
  * `summary.json` (includes optional run name/notes metadata)
  * `response_lr_overlay.png` -- L/R overlay plot (smoothed)
  * `lr_diff.png` -- L minus R mismatch (smoothed)
  * `difference_l.png`, `difference_r.png` -- only if `--loopback` is provided
  * `impulse_l.png`, `impulse_r.png` -- only if `--save-ir` is used

## Notes and gotchas

* HF loss is often mechanical (azimuth, head wear, dirty path, wrong tape-type EQ) before it's "EQ fixing".

* Cassette transports drift. This tool compensates drift with a linear warp between start/end markers.

* For non-linear drift (wow/flutter), enable ticks mode (`gen --ticks` + `analyze --ticks`) for piecewise correction.

* If marker detection fails:

  * Generate hotter markers: `--marker-dbfs -10`
  * Loosen detection: lower `--thresh` (e.g. 4.5) or set `--min-dbfs -60`

* Before measuring: clean heads/capstan/pinch roller; check azimuth if HF looks nuked.

* Set record level using the 1 kHz reference tone (avoid clipping / redlining).

* Keep any NR / enhancers off unless you are specifically measuring them.

* WebUI paths are **relative to the project root**. Put files under `data/` and browse/pick them.

## TODO / WIP

* A/B spectrogram view: one horizontal spectrogram for the reference "input" WAV and one for the synced recorded capture -- with optional L/R split if the files are stereo.

* Comparative spectrum analyzer / visualizer: timeline + playhead, with color-coded overlay (baseline vs cassette-looped) so you can see how they interact/differ over time.

* Phase correlation check (correlation meter / phase relationship), ideally also available per-channel.

## Changelog / History

* 0.1.8 - WebUI enhancements
  * Enhanced logging, errors/status
  * Matplotlib output plots are now named with a title set as the run name
  * Plenty of bugfixes

* 0.1.7 - WebUI: clickable images + fullscreen viewer

  * Thumbnails are now clickable: opens a fullscreen overlay viewer for easier inspection/zooming.
  * Added "Open in new tab" link per image so you can view the raw PNG directly (browser zoom works properly).
  * Viewer shows basic file details (path + dimensions + byte size) for quick sanity checks.
  * Added fit/actual-size toggle controls (plus keyboard ESC to close).
  * Added lightweight file stat endpoint used by the viewer to display image metadata.

* 0.1.6 - WebUI: edit notes for existing runs

  * Added "Edit notes" UI for loaded runs with Save/Cancel.
  * Added `/api/run_notes` endpoint to update `summary.json` -> `run.notes` for a run directory.
  * Atomic JSON write with mtime preservation so run list ordering does not change due to note edits.
  * README updated to document the notes editor and write behavior.

* 0.1.5 - Stereo outputs + tick-warp improvements; README/output docs sync
  * Added stereo analysis outputs: per-channel plots/CSVs (`response_l/r.*`) plus `lr_diff.png`.
  * Added optional L/R overlay plot output: `response_lr_overlay.png` (enabled by default when analyzing both L and R).
  * Added tick-based non-linear drift correction ("ticks mode") with matching/tolerance controls and a quality gate to avoid bad warps.
  * Improved DTMF detection stability: safer/auto dedupe behavior and optional timing stats logging (`--dtmf-stats`).
  * Config/TOML support expanded for new options (channels, marker_channel, overlay toggles/colors, tick settings, SNR section mapping).
  * WebUI now surfaces the new stereo outputs/overlay images from `summary.json`.
  * Documentation updated to reflect stereo-default outputs and filenames.

* 0.1.4 - Local WebUI (stdlib) introduced; fixes

  * Added `webui.py` local-only WebUI (binds to 127.0.0.1 by default).
  * Modal file browser with directory creation and safe relative-path handling.
  * Fixed onclick/quoting issues in generated browse list buttons.
  * Documents relative-path requirement (no absolute paths / no "..").

* 0.1.3 - x-axis plot style; drift alignment fixes and more

  * Improved plot x-axis formatting/ticks for audio frequencies (20 Hz–20 kHz readability).
  * Drift alignment fixes for more reliable marker-to-marker speed compensation.
  * Better exports/plot naming consistency (response plots + CSV + summary JSON).

* 0.1.2 - bugfixes; config changes; dedupe parsing
* 0.1.1 - Patches to `--help` etc
* 0.1.0 - Initial release

## About

By [FlyingFathead](https://github.com/FlyingFathead) with bits and bytes of help from ChaosWhisperer.
