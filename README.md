# SKANN-SSL — Self-Supervised Acoustic Neural Network for Vessel Detection & Classification

**Client:** Altair Infrasec Pvt Ltd  
**Delivery:** Oravont Systems LLP  
**Deployment:** NUWR Goa — Seabed hydrophone, 30m depth, Arabian Sea, 512 Hz  
**Dataset:** 28 January – 11 February 2026 | 292 one-hour sessions | ~292 hrs of acoustic data

---

## System Overview

SKANN-SSL is a passive acoustic vessel detection and classification system built on self-supervised learning (SSL). It processes raw hydrophone recordings to detect vessel transits and learn acoustic vessel signatures without requiring labelled training data.

The system comprises two functional layers:

- **Detection layer (HAVS):** RMS energy-based gate using a global P15 baseline across all 292 sessions. Confirmed 326× dynamic range, clean day/night separation, 3.3% detection rate.
- **Classification layer (SKANN-SSL):** Self-supervised encoder (HybridSKEncoderV5) trained using Barlow Twins contrastive learning on 141 confirmed/probable vessel events.

---

## Pipeline — Chronological Workflow

### Step 1 — Session Data Management
**Notebook:** `NUWR_Session_Data_Management.ipynb`

Staged and validated all 292 hydrophone sessions from Google Drive. Resolved naming conflicts between 28 January and 30 January sessions 1–4. Identified sessions 145–176 as bad recordings requiring replacement. Built the session manifest used by all downstream notebooks.

---

### Step 2 — HAVS Detection Pipeline (All Sessions)
**Notebook:** `RMS_lofar_demon_pipeline_v5_allsessions.ipynb`

Ran the full HAVS (Hydrophone Acoustic Vessel Signature) pipeline across all authorised sessions:

- **RMS energy detector:** Global P15 baseline (0.00010593, computed across 292 sessions). Detection threshold ×2.0. Produced 326× dynamic range, clean ambient/vessel separation.
- **LOFAR stage:** Tonal detection across 0–256 Hz. Functional but over-sensitive at 73% window flagging rate — threshold requires refinement.
- **DEMON stage:** Assessed as inappropriate for detection at 512 Hz (cavitation signatures require higher sample rates). Retained as a characteriser for confirmed transits only.

Key finding: measured spectral levels (~27.7 dB re 1 µPa at 100 Hz) are below Wenz curve predictions, indicating electronic self-noise dominance — flagged to Altair as a hydrophone sensitivity/gain issue.

---

### Step 2.5 — HAVS Top-Up (28 January Sessions)
**Notebook:** `RMS_lofar_demon_pipeline_v5_28jan_topup.ipynb`

Applied HAVS pipeline to the 28 January sessions which were not included in the main run due to naming conflicts resolved in Step 1. Merged results into the master detection output.

---

### Step 3 — SKANN-SSL Data Pipeline
**Notebook:** `SKANN_SSL_Data_Pipeline_v5.ipynb`

Built the SSL training dataset from HAVS-detected vessel events:

- Extracted 30-second tensor windows (15,360 samples @ 512 Hz) for each vessel event
- Applied z-score normalisation per window: `(x − mean) / max(std, 1e-6)`
- Produced 141 vessel events across `v5_data/tensors/vessel/` (7,050 tensors total at 50 windows/event)
- Generated `window_pool_50.csv` — the master manifest of all training tensors

---

### Step 4 — Event Normalisation
**Notebook:** `event_normalisation.ipynb`

Normalised all 141 vessel events to exactly 50 windows each using spectral peakiness as the selection criterion. Greedy seed window (highest peakiness) selected first; remaining 49 windows sampled to maximise temporal diversity across the transit. Produced `window_class_metadata.csv` mapping every tensor to its event ID and HAVS assessment.

---

### Step 5 — Pairing Manifest Generation
**Notebook:** `pairing_manifest_generator.ipynb`

Generated 56,400 contrastive training pairs for Barlow Twins SSL:

- K=8 pairs per anchor (5 temporal + 3 semantic for GT-labelled events)
- Temporal pairs: windows from the same transit, minimum 90s separation
- Semantic pairs: cross-event pairing anchored to 3 ground-truth labelled events (fishing trawler, tanker, small craft)
- Hard negative selection via cosine distance in 6D spectral feature space
- Train/val split: 46,400 / 10,000 (82/18%), event-isolated (no anchor leakage)

---

### Step 6 — V5 SSL Training
**Notebook:** `SKANN_SSL_V5_Training_512Hz_v5.ipynb`

Trained HybridSKEncoderV5 using Barlow Twins self-supervised learning:

**Architecture (locked):**
- 1D SK Filterbank: 8 parallel Conv1d branches, kernels (7–1023), 64 filters, GELU + GroupNorm
- 2D Backbone: 5× Conv2d BN+ReLU, strides (1,1)→(1,2)→(1,2)→(2,1)→(2,2), AdaptiveAvgPool2d → h (512-dim)
- Projector: 512→2048→2048→256 (training only, discarded at deployment)
- Total parameters: 9.83M (4.05M backbone, 5.78M projector)

**Training configuration:**
- 60 epochs, batch size 32, AdamW lr=3e-5, Barlow Twins λ=5e-3, AMP enabled
- GPU: NVIDIA RTX PRO 6000 Blackwell (102 GB VRAM)
- Best same-event cosine similarity: 0.9781

**Post-training clustering (KMeans, k=5):**
- Silhouette score: 0.406 (cosine metric)
- Cluster sizes: C0=50, C1=35, C2=23, C3=18, C4=15
- t-SNE (cosine metric) shows clear cluster separation

---

### Step 7 — V5 Evaluation & Diagnostic
**Scripts:** `diagnose_v5.py`, `recluster_cosine.py`

Comprehensive post-training evaluation of the V5 production bundle:

**Detection:** Fully functional. Gate fires correctly on vessel transits, silent on ambient. Global P15 baseline performing as designed.

**Classification — finding:**  
Fresh-embed recovery of training tensors: 65%. Cosine similarity gap (same vs different cluster): **0.0356** — below the 0.05 threshold required for reliable classification.

Root cause analysis via PCA of h-space:
- PC1 explains **75% of embedding variance**
- PC1 correlates with sub-20 Hz spectral energy at Spearman **|r| = 0.794**
- The encoder learned spectral slope as the dominant feature, consuming 75% of embedding capacity on a single acoustic axis
- KMeans clusters are strung along this axis with small inter-centroid gaps — classification is near-flat for any given event

**V5 status:** Detection deployable. Classification not reliable with current training data volume.

---

### Step 8 — V6 Investigation (In Progress)
**Notebook:** `SKANN_SSL_V6_Training_512Hz.ipynb`

Two augmentation strategies investigated to force the encoder off the spectral slope axis:

**Attempt 1 — Random gain augmentation (±12 dB):**  
Hypothesis: encoder learned amplitude. Disproved — the diagnostic `tonal_score` is a spectral ratio, invariant to uniform gain scaling. Tonal |r| remained flat at 0.71 across 10 epochs. Abandoned.

**Attempt 2 — Random spectral tilt (α ~ U[−1, +1]):**  
Applies f^α tilt independently to each view, making spectral slope inconsistent within a pair. Tonal |r| declined from 0.697 → 0.606 over 20 epochs (vs flat in random_gain), confirming the augmentation is engaging. However silhouette remained negative (−0.28), indicating insufficient vessel-discriminative structure in the residual embedding space once slope variation is introduced.

**Root constraint identified:**  
Spectral slope is a genuine vessel acoustic feature (larger vessels radiate proportionally more low-frequency energy). Suppressing it removes the primary discriminant. Classification requires additional ground-truth vessel-type labels — AIS data cross-referenced against event timestamps would resolve this without manual annotation.

**V6 status:** Under investigation. Architecture unchanged from V5.

---

## Key Results Summary

| Metric | Value |
|---|---|
| Sessions processed | 292 |
| Vessel events detected | 141 |
| Training tensors | 7,050 (50 windows × 141 events) |
| V5 best same-event similarity | 0.9781 |
| V5 clustering silhouette | 0.406 |
| V5 detection | ✓ Deployable |
| V5 classification | ✗ Cosine sim gap 0.036 — below threshold |
| V6 status | Under investigation |

---

## Repository Structure

```
HAVS-SKANN-SSL-Demo/
├── models/
│   └── SKANN_SSL_V5_Production_Bundle.joblib   # Weights + centroids + embeddings
├── test_clips/                                  # Reference WAV clips for demo testing
├── skann_demo.py                                # V5 demo — detection only
├── diagnose_v5.py                               # Post-training diagnostic
├── recluster_cosine.py                          # Cosine recluster diagnostic
├── NUWR_Session_Data_Management.ipynb           # Step 1
├── RMS_lofar_demon_pipeline_v5_allsessions.ipynb # Step 2
├── RMS_lofar_demon_pipeline_v5_28jan_topup.ipynb # Step 2.5
├── SKANN_SSL_Data_Pipeline_v5.ipynb             # Step 3
├── event_normalisation.ipynb                    # Step 4
├── pairing_manifest_generator.ipynb             # Step 5
├── SKANN_SSL_V5_Training_512Hz_v5.ipynb         # Step 6
└── SKANN_SSL_V6_Training_512Hz.ipynb            # Step 8
```

---

## Deployment Notes

- **Sample rate:** 512 Hz (fixed — architecture locked to this rate)
- **Window length:** 30s (15,360 samples)
- **Detection threshold:** Global P15 RMS baseline × 2.0
- **Embedding dimension:** 512 (h-vector, deployment)
- **Demo:** `python skann_demo.py --wav <path>` — returns gate decision + cluster probabilities (V5: detection reliable, classification indicative only)

---

## Next Steps

1. Obtain AIS vessel type data for Arabian Sea, 28 Jan – 11 Feb 2026 — cross-reference against 141 detected events to assign vessel-type ground truth labels
2. Re-evaluate V5 cluster structure against labelled ground truth
3. Retrain V6 with label-informed pairing strategy once sufficient GT labels are available

---

*Oravont Systems LLP | Capt (Dr) Sunil Tyagi | March 2026*
