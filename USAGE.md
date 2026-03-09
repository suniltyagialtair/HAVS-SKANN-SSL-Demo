# SKANN-SSL V5 Demo — Usage Guide

**Script:** `skann_demo.py`  
**Model:** `models/SKANN_SSL_V5_Production_Bundle.joblib`  
**Deployment:** NUWR Goa | Seabed hydrophone | 30m depth | Arabian Sea | 512 Hz

---

## Prerequisites

Python 3.9+ required. Install dependencies:

```cmd
pip install -r requirements.txt
```

Ensure the model bundle is present at `models\SKANN_SSL_V5_Production_Bundle.joblib`.  
If missing, download from Google Drive: `G:\My Drive\SKANN_SSL\ssl_output_v5\SKANN_SSL_V5_Production_Bundle.joblib`

---

## Usage

```cmd
cd D:\HAVS-SKANN-SSL-Demo
python skann_demo.py --wav <path_to_wav_file>
```

The input WAV must be:
- Sample rate: 512 Hz (hard requirement — architecture is locked to this rate)
- Format: mono, float32 or int16
- Duration: minimum 30 seconds (one analysis window)

---

## Test Clips

Reference clips are provided in `test_clips\` to verify correct operation:

| Clip | Expected result |
|---|---|
| `ambient_raw_30s.wav` | Gate silent — No Vessel |
| `vessel_event001_offset90s.wav` | Gate fires — Vessel Detected |
| `event_0137_20260205_081052_CONFIRMED_VESSEL_train.wav` | Gate fires — 25-minute confirmed transit |
| `session_145_2026-02-05_08-00-52_to_2026-02-05_09-00-50.wav` | Full 1-hour session timeline |

### Run all tests in sequence

```cmd
:: Test 1 — Ambient baseline (gate must be silent)
python skann_demo.py --wav test_clips\ambient_raw_30s.wav

:: Test 2 — Vessel event (gate must fire)
python skann_demo.py --wav test_clips\vessel_event001_offset90s.wav

:: Test 3 — Confirmed 25-minute vessel transit
python skann_demo.py --wav test_clips\event_0137_20260205_081052_CONFIRMED_VESSEL_train.wav

:: Test 4 — Full 1-hour session timeline
python skann_demo.py --wav test_clips\session_145_2026-02-05_08-00-52_to_2026-02-05_09-00-50.wav
```

---

## Output Format

For each 30-second analysis window the demo prints:

```
Gate: VESSEL DETECTED
C0: 0.237  C1: 0.194  C2: 0.200  C3: 0.232  C4: 0.138
```

Or for ambient:

```
Gate: NO VESSEL  (RMS 0.66x threshold)
```

**Gate decision:** Based on global P15 RMS baseline (0.00010593) with x2.0 threshold multiplier. Computed across all 292 NUWR sessions (28 Jan - 11 Feb 2026).

**C0-C4 probabilities:** Softmax scores from Euclidean distance to 5 KMeans centroids in 512-dimensional h-space. Scores are normalised to sum to 1.0.

---

## Status

| Function | Status | Notes |
|---|---|---|
| Detection (gate) | Deployable | 326x dynamic range, validated on full dataset |
| Classification (C0-C4) | Indicative only | Cosine sim gap 0.036 — below reliable threshold |

**Classification limitation:** The V5 encoder learned spectral slope as the dominant feature (PC1 explains 75% of embedding variance). The five clusters separate vessel events along an acoustic character axis but reliable per-vessel-type classification requires additional ground-truth labels. AIS data cross-referenced against event timestamps would resolve this.

See `README.md` for full technical background.

---

## Detection Threshold

The RMS gate threshold is stored in `models\deployment_config.json`:

```json
{
  "global_baseline_rms": { "value": 0.00010593 },
  "threshold_multiplier": 2.0
}
```

To adjust sensitivity, modify `threshold_multiplier`. Increasing it reduces false alarms; decreasing it increases sensitivity at the cost of more ambient triggers.

---

## Diagnostic Scripts

Two additional scripts are provided for model diagnostics:

**`diagnose_v5.py`** — Embeds training tensors through the model and checks cluster recovery. Use to verify model integrity after any environment change.

```cmd
python diagnose_v5.py --bundle models\SKANN_SSL_V5_Production_Bundle.joblib ^
                      --tensor_dir "G:\My Drive\SKANN_SSL\v5_data\tensors\vessel" ^
                      --pool "G:\My Drive\SKANN_SSL\ssl_data_50w\window_pool_50.csv"
```

**`recluster_cosine.py`** — Reclusters stored embeddings using cosine metric and compares recovery against Euclidean baseline. Used to diagnose classification performance.

```cmd
python recluster_cosine.py --bundle models\SKANN_SSL_V5_Production_Bundle.joblib ^
                           --tensor_dir "G:\My Drive\SKANN_SSL\v5_data\tensors\vessel" ^
                           --pool "G:\My Drive\SKANN_SSL\ssl_data_50w\window_pool_50.csv"
```

---

*Oravont Systems LLP | Capt (Dr) Sunil Tyagi | March 2026*
