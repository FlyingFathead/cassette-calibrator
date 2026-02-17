# cassette-calibrator

A CLI tool for measuring and calibrating a compact cassette recording/playback chain using an audio test program and automated alignment.

It generates a cassette-friendly measurement WAV with robust DTMF markers, then analyzes a recorded playback capture to estimate the chain's magnitude response (and optional loopback-subtracted response), plus an SNR estimate.

## What this does

- Generates a print-to-tape WAV containing:
  - DTMF start marker (robust "audio timecode" anchor)
  - Dedicated silence window for noise-floor measurement
  - DTMF countdown (default on; mostly for humans, alignment uses start/end markers)
  - 1 kHz reference tone for setting record level
  - ESS log sweep (default 20 Hz -> 20 kHz)
  - DTMF end marker

- Analyzes a recorded capture by:
  - Detecting start/end markers automatically (no manual alignment)
  - Extracting the sweep region based on known layout
  - Applying a linear time-warp based on marker-to-marker drift (transport speed mismatch)
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
- Clean heads + capstan + pinch roller.
- Set the deck’s tape type correctly (Type I/II/IV).
- Disable anything “helpful” in the chain: Dolby/NR, EQ, AGC, enhancers, expander, etc. (deck + interface mixer + OS).

**A) Record pass (interface -> deck -> tape)**
- Connect **interface line out** (L/R) -> deck **line in / aux in** (do **not** use mic inputs).
- Play `sweepcass.wav` from a player that doesn’t resample or add DSP.
  - Avoid OS “audio enhancements”, spatial sound, EQ, loudness normalization, etc.
  - Don’t let system sounds mix into the output.
- Start deck recording, then start playback of `sweepcass.wav` from the beginning.
- When the file reaches the **1 kHz reference tone**, set the deck’s record level:
  - Aim for a safe, repeatable level (avoid clipping/redlining).
  - The goal is consistency, not “as hot as possible”.
- Rewind and do a clean, uninterrupted record pass of the full file (including pre/post silence).

**B) Playback capture (deck -> interface -> WAV)**
- Connect deck **line out / rec out** -> **interface line in**.
- Capture as `recorded.wav` at the **same sample rate** as generated (default 44.1 kHz).
- Record at **24-bit** if you can; leave headroom (avoid clipping).
- Start recording first, then start cassette playback; stop after the end marker + post silence.

Avoid any AGC/NR/enhancers in the interface path.

### 3) (Optional) sanity-check marker detection

```bash
python3 cassette_calibrator.py detect --wav recorded.wav
# or machine-readable:
python3 cassette_calibrator.py detect --wav recorded.wav --json
```

### 4) Analyze and export results

```bash
python3 cassette_calibrator.py analyze --ref sweepcass.wav --rec recorded.wav --outdir results --fine-align
# or print summary JSON:
python3 cassette_calibrator.py analyze --ref sweepcass.wav --rec recorded.wav --outdir results --fine-align --json
```

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

* `response.png` -- smoothed magnitude response (log frequency)
* `response.csv` -- raw + smoothed response data
* `summary.json` -- marker times, drift ratio, SNR estimate, settings used
* `difference.png` -- only if `--loopback` is provided
* `impulse.png` -- only if `--save-ir` is used

## Notes and gotchas

* HF loss is often mechanical (azimuth, head wear, dirty path, wrong tape-type EQ) before it's "EQ fixing".
* Cassette transports drift. This tool compensates drift with a linear warp between start/end markers.
* If marker detection fails:

  * Generate hotter markers: `--marker-dbfs -10`
  * Loosen detection: lower `--thresh` (e.g. 4.5) or set `--min-dbfs -60`

* Before measuring: clean heads/capstan/pinch roller; check azimuth if HF looks nuked.
* Set record level using the 1 kHz reference tone (avoid clipping / redlining).
* Keep any NR / enhancers off unless you are specifically measuring them.

## About

By [FlyingFathead](https://github.com/FlyingFathead) with a bit of help from ChaosWhisperer.