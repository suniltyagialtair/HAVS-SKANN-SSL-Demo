"""
SKANN-SSL V5 — Vessel Acoustic Demo
=====================================
Two operating modes, auto-selected by WAV duration:

  TIMELINE mode  (WAV > 35 s)
      Input  : Any-length 512 Hz WAV (typically a 60-min session)
      Output : timeline_<session>.png  — RMS trace + coloured event bands + event table
               radar_<clip_id>.png     — one radar plot per detected event
               summary_<session>.csv   — full event table with probabilities

  RADAR mode  (WAV <= 35 s, or --mode radar)
      Input  : Exactly 30-second 512 Hz WAV (one vessel event clip)
      Output : radar_<clip_id>.png     — radar confidence plot

Radar axes: No Vessel | C0 | C1 | C2 | C3 | C4
  - RMS detector fires  → embed → classify → C0-C4 probs, No Vessel = 0
  - RMS detector silent → No Vessel = 1.0, C0-C4 = 0, no embedding computed

Usage
-----
  python skann_demo.py --wav session.wav   --bundle bundle.joblib
  python skann_demo.py --wav event_30s.wav --bundle bundle.joblib
  python skann_demo.py --wav session.wav   --bundle bundle.joblib --mode timeline
  python skann_demo.py --wav event_30s.wav --bundle bundle.joblib --mode radar
  python skann_demo.py --wav session.wav   --bundle bundle.joblib --rms-thresh 1.5

Requirements
------------
  pip install numpy scipy soundfile torch joblib matplotlib scikit-learn

Architecture (V5 — LOCKED)
--------------------------
  Window         : 15,360 samples = 30 s @ 512 Hz
  SK Filterbank  : SKConv1D, 8 kernels (7,15,31,63,127,255,511,1023), 64 ch
  2D Backbone    : l1(1x1) l2(1x2) l3(1x2) l4(2x1) l5(2x2) -> GAP -> 512-dim h
  Projector      : 512->2048->2048->256  (SSL only, discarded at deployment)
  Classification : Euclidean distance to K-Means centroids C0-C4 (1/(d+0.8), normalised)
"""

import argparse
import csv
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch


# ── Constants (MUST match training notebook exactly) ─────────────────────────
FS           = 512
WINDOW_SEC   = 30.0
WINDOW_SAMP  = int(WINDOW_SEC * FS)    # 15,360
LATENT_DIM   = 256

RADAR_LABELS  = ["No\nVessel", "C0", "C1", "C2", "C3", "C4"]
N_AXES        = len(RADAR_LABELS)
CLUSTER_NAMES = ["C0", "C1", "C2", "C3", "C4"]
N_CLUSTERS    = len(CLUSTER_NAMES)

AXIS_COLORS = {
    "No\nVessel": "#8C8C8C",
    "C0":         "#2166AC",
    "C1":         "#74ADD1",
    "C2":         "#E08010",
    "C3":         "#1A9850",
    "C4":         "#D73027",
}

# RMS detector parameters
RMS_FRAME_SEC  = 2.0
RMS_HOP_SEC    = 1.0
SUSTAIN_SEC    = 8.0
GAP_BRIDGE_SEC = 5.0
MIN_EVENT_SEC  = 3.0

RADAR_MODE_MAX_SEC = 35.0


# ══════════════════════════════════════════════════════════════════════════════
# MODEL — exact match to HybridSKEncoderV5 in training notebook
# ══════════════════════════════════════════════════════════════════════════════

def _norm_1d(channels, groups=8):
    return nn.GroupNorm(min(groups, channels), channels)


class SKConv1D(nn.Module):
    def __init__(self, in_ch, out_ch,
                 kernel_sizes=(7, 15, 31, 63, 127, 255, 511, 1023),
                 stride=1, reduction=16, residual=False):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(in_ch, out_ch, k, stride=stride,
                      padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.n_branches = len(kernel_sizes)
        self.out_ch = out_ch
        hidden = max(out_ch // reduction, 8)
        self.fc1  = nn.Linear(out_ch, hidden)
        self.fc2  = nn.Linear(hidden, out_ch * self.n_branches)
        self.norm = _norm_1d(out_ch)
        self.act  = nn.GELU()
        self.residual = residual
        self.match = None
        if residual and in_ch != out_ch:
            self.match = nn.Conv1d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        U = torch.stack(feats, dim=1).sum(dim=1)
        s = F.adaptive_avg_pool1d(U, 1).squeeze(-1)
        z = self.fc2(F.relu(self.fc1(s), inplace=False))
        a = F.softmax(
            z.view(z.size(0), self.n_branches, self.out_ch), dim=1
        ).unsqueeze(-1)
        V   = (a * torch.stack(feats, dim=1)).sum(dim=1)
        out = self.act(self.norm(V))
        if self.residual:
            out = out + (x if self.match is None else self.match(x))
        return out


class SKFilterbank(nn.Module):
    KERNELS = (7, 15, 31, 63, 127, 255, 511, 1023)

    def __init__(self, out_ch=64):
        super().__init__()
        self.stem      = SKConv1D(1, out_ch, self.KERNELS, residual=False)
        self.post_norm = _norm_1d(out_ch)

    def forward(self, x):
        return self.post_norm(self.stem(x))


def _conv2d_bn_relu(in_ch, out_ch, kernel=3, stride=(1, 1)):
    pad = (kernel // 2, kernel // 2)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=False),
    )


class HybridSKEncoderV5(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.sk_frontend = SKFilterbank(out_ch=64)
        self.l1 = _conv2d_bn_relu(  1,  64, stride=(1, 1))
        self.l2 = _conv2d_bn_relu( 64, 128, stride=(1, 2))
        self.l3 = _conv2d_bn_relu(128, 256, stride=(1, 2))
        self.l4 = _conv2d_bn_relu(256, 512, stride=(2, 1))
        self.l5 = _conv2d_bn_relu(512, 512, stride=(2, 2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Sequential(
            nn.Linear(512,  2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=False),
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=False),
            nn.Linear(2048, latent_dim),
        )

    def forward(self, x, return_features=False):
        if x.dim() == 1: x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2: x = x.unsqueeze(1)
        x = self.sk_frontend(x)        # (B, 64, 15360)
        x = x.unsqueeze(1)             # (B, 1, 64, 15360)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        h = self.pool(x).flatten(1)    # (B, 512)
        z = self.projector(h)          # (B, latent_dim)
        return (h, z) if return_features else z


# ══════════════════════════════════════════════════════════════════════════════
# RMS EVENT DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

def detect_events(audio, fs, thresh_mult=2.0, global_baseline=None):
    frame_samp = int(fs * RMS_FRAME_SEC)
    hop_samp   = int(fs * RMS_HOP_SEC)
    n_frames   = max(1, (len(audio) - frame_samp) // hop_samp + 1)

    rms = np.array([
        np.sqrt(np.mean(audio[i * hop_samp: i * hop_samp + frame_samp] ** 2))
        for i in range(n_frames)
    ])
    times_sec = np.arange(n_frames) * RMS_HOP_SEC + RMS_FRAME_SEC / 2.0

    # Global baseline from deployment_config.json — fixed across all sessions.
    # Falls back to local P15 only if config is unavailable (should not happen
    # in normal deployment).
    if global_baseline is not None:
        baseline = global_baseline
    else:
        baseline = np.percentile(rms, 15)
        print("[WARN]  No global baseline — falling back to local P15. "
              "Ensure deployment_config.json is present in models/")

    threshold = baseline * thresh_mult
    active    = rms > threshold

    # Gap bridging
    bridge = int(GAP_BRIDGE_SEC / RMS_HOP_SEC)
    for i in range(len(active) - bridge):
        if active[i] and active[i + bridge]:
            active[i:i + bridge] = True

    # Sustain gate
    sustain_f = int(SUSTAIN_SEC / RMS_HOP_SEC)
    sustained = np.zeros_like(active)
    i = 0
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            if (j - i) >= sustain_f:
                sustained[i:j] = True
            i = j
        else:
            i += 1

    events = []
    in_ev, start_f = False, 0
    for i, s in enumerate(sustained):
        if s and not in_ev:
            in_ev, start_f = True, i
        elif not s and in_ev:
            in_ev = False
            dur = (i - start_f) * RMS_HOP_SEC
            if dur >= MIN_EVENT_SEC:
                events.append((
                    start_f * hop_samp,
                    min(i * hop_samp + frame_samp, len(audio))
                ))
    if in_ev:
        dur = (len(sustained) - start_f) * RMS_HOP_SEC
        if dur >= MIN_EVENT_SEC:
            events.append((start_f * hop_samp, len(audio)))

    return events, rms, times_sec, threshold, baseline


# ══════════════════════════════════════════════════════════════════════════════
# TENSOR EXTRACTION & EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════

def normalise_window(w, silence_guard=1e-6):
    """Zero mean, unit std — matches training normalisation exactly (Cell 22)."""
    w    = w.astype(np.float32)
    mean = w.mean()
    std  = w.std()
    return (w - mean) / max(std, silence_guard)


def spectral_peakiness(w, fs=FS):
    """Peak/median power ratio in 1-180 Hz.
    Matches Step 3 (SKANN_SSL_Data_Pipeline_v5) Cell 22 spectral_peakiness() exactly.
    Used to select the most acoustically distinctive window from an unseen clip.
    """
    n     = len(w)
    X     = np.abs(np.fft.rfft(w)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    mask  = (freqs >= 1.0) & (freqs <= 180.0)
    band  = X[mask]
    if len(band) == 0:
        return 0.0
    median = np.median(band)
    return float(band.max() / median) if median > 1e-20 else 0.0


def extract_best_window(audio):
    """Select the most acoustically distinctive 30-s window from audio.

    Maximises spectral peakiness across all 15-s-hop windows — matches the
    greedy_diverse_subset seed criterion in Step 4 (event_normalisation):
    the seed is the window furthest from the feature centroid in IQR-scaled
    6D feature space, dominated by peakiness.

    Returns raw float32 window (NOT yet z-scored).
    The caller applies normalise_window() after this.
    Short clips zero-padded to WINDOW_SAMP.
    """
    n = len(audio)
    if n < WINDOW_SAMP:
        w = np.zeros(WINDOW_SAMP, dtype=np.float32)
        w[:n] = audio.astype(np.float32)
        return w

    best_window = audio[:WINDOW_SAMP].astype(np.float32)
    best_score  = -1.0

    for s in range(0, n - WINDOW_SAMP + 1, int(FS * 15)):   # 15-s hop = HOP_SAMP
        w     = audio[s: s + WINDOW_SAMP].astype(np.float32)
        score = spectral_peakiness(w)
        if score > best_score:
            best_score  = score
            best_window = w.copy()

    return best_window


@torch.no_grad()
def embed_one(model, window_raw, device):
    """Embed a single raw 30-s audio window. Returns h (512-dim numpy vector).

    Preprocessing chain (matches training exactly):
      1. extract_best_window() selects the window (caller's responsibility)
      2. normalise_window()  — z-score once, matches Step 3 Cell 22
      3. model forward pass  — no further normalisation
    window_raw: numpy float32 (WINDOW_SAMP,) raw audio, NOT yet z-scored.
    """
    model.eval()
    w_norm = normalise_window(window_raw)                  # z-score once
    t      = torch.from_numpy(w_norm).float().unsqueeze(0).to(device)
    h, _   = model(t, return_features=True)
    return h.cpu().numpy().squeeze()   # (512,)


def classify(h_mean, centroids):
    """Classify by Euclidean distance to K-Means centroids.
    Matches training: KMeans was fit in raw h-space (Euclidean), NOT cosine.
    Scores = 1 / (dist + 0.8), normalised to sum to 1.
    """
    dists  = np.array([np.linalg.norm(h_mean - c) for c in centroids])
    scores = 1.0 / (dists + 0.8)
    return scores / scores.sum()


# ══════════════════════════════════════════════════════════════════════════════
# RADAR PLOT
# ══════════════════════════════════════════════════════════════════════════════

def make_radar(cluster_probs, no_vessel, clip_id, out_path, meta=""):
    """
    Radar plot — V3 aesthetic.
    White background, single steel-blue fill, predicted axis highlighted crimson.
    No Vessel node at 12 o'clock (index 0).
    """
    if no_vessel:
        full_probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        full_probs = np.concatenate([[0.0], cluster_probs])

    display_labels = ["No Vessel", "C0", "C1", "C2", "C3", "C4"]
    num_vars = len(display_labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    stats  = full_probs.tolist() + [full_probs[0]]
    angles_closed = angles + angles[:1]

    pred_idx   = int(np.argmax(full_probs))
    pred_label = display_labels[pred_idx]
    confidence = full_probs[pred_idx] * 100

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_ylim(0, 1.0)

    # ── Radar fill and outline ────────────────────────────────────────────────
    if no_vessel:
        fill_color = "#888888"
    else:
        fill_color = "tab:blue"

    ax.fill(angles_closed, stats, alpha=0.15, color=fill_color)
    ax.plot(angles_closed, stats, linewidth=2, color=fill_color, alpha=0.8)

    # ── Grid ──────────────────────────────────────────────────────────────────
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],
                       fontsize=9, color="#888888")
    ax.yaxis.set_tick_params(pad=22)
    ax.grid(color="#CCCCCC", linestyle="--", linewidth=0.7, alpha=0.9)
    ax.spines["polar"].set_color("#CCCCCC")

    # ── Axis labels ───────────────────────────────────────────────────────────
    ax.set_xticks(angles)
    ax.set_xticklabels(display_labels, fontsize=12)
    ax.tick_params(axis="x", pad=18)

    # Highlight predicted axis label
    for i, label_obj in enumerate(ax.get_xticklabels()):
        if i == pred_idx:
            label_obj.set_color("crimson")
            label_obj.set_fontweight("bold")
            label_obj.set_fontsize(14)

    # ── Correct/incorrect annotation (top-right) ──────────────────────────────
    if no_vessel:
        status_text = "No vessel\ndetected"
        status_color = "#555555"
    else:
        status_text = f"{pred_label}\n({confidence:.1f}%)"
        status_color = "crimson"

    ax.text(
        0.86, 0.98, status_text,
        transform=ax.transAxes,
        ha="left", va="top",
        multialignment="left",
        fontsize=12, fontweight="bold",
        color=status_color,
    )

    # ── Header ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.985, f"CLIP: {clip_id}",
             ha="center", va="top", fontsize=10, fontweight="bold")

    if meta:
        fig.text(0.5, 0.958, meta,
                 ha="center", va="top", fontsize=9, color="#555555", style="italic")

    if no_vessel:
        id_text  = "NO VESSEL DETECTED"
        id_color = "#555555"
        id_bg    = "#F0F0F0"
        id_edge  = "#AAAAAA"
    else:
        id_text  = f"IDENTIFIED: {pred_label}  ({confidence:.1f}%)"
        id_color = "black"
        id_bg    = "crimson"
        id_alpha = 0.18
        id_edge  = "crimson"

    bbox_kw = dict(facecolor=id_bg, alpha=0.18 if not no_vessel else 1.0,
                   edgecolor=id_edge, boxstyle="round,pad=0.35")
    fig.text(0.5, 0.928, id_text,
             ha="center", va="top",
             fontsize=11, fontweight="bold",
             bbox=bbox_kw, color=id_color)

    fig.subplots_adjust(top=0.82)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)



def make_timeline(rms_frames, times_sec, threshold, baseline,
                  results, wav_name, duration_sec, out_path):
    n_ev  = len(results)
    fig_h = max(7.5, 5 + n_ev * 0.45)
    fig, axes = plt.subplots(
        2, 1, figsize=(17, fig_h),
        gridspec_kw={"height_ratios": [3, max(1.4, n_ev * 0.60)],
                     "hspace": 0.40}
    )
    fig.patch.set_facecolor("white")

    ax   = axes[0]
    t_mn = times_sec / 60.0

    ax.fill_between(t_mn, rms_frames, alpha=0.08, color="#333333", zorder=1)
    ax.plot(t_mn, rms_frames, color="#333333", linewidth=0.9,
            label="RMS energy", zorder=3)
    ax.axhline(threshold, color="#CC2222", linewidth=1.2, linestyle="--",
               label="Detection threshold", zorder=4)
    ax.axhline(baseline, color="#2255AA", linewidth=0.8, linestyle=":",
               label="Baseline (15th pct)", zorder=4)

    for r in results:
        s_mn = r["start_sec"] / 60.0
        e_mn = r["end_sec"]   / 60.0
        col  = AXIS_COLORS[r["predicted"]]
        ax.axvspan(s_mn, e_mn, alpha=0.15, color=col, zorder=2)
        ax.axvspan(s_mn, e_mn, alpha=0.85, color=col,
                   ymin=0.92, ymax=1.00, zorder=5)
        mid = (s_mn + e_mn) / 2.0
        ax.text(mid, 1.0, f" {r['predicted']} ",
                ha="center", va="top",
                fontsize=7.5, fontweight="bold", color="white",
                transform=ax.get_xaxis_transform(),
                clip_on=False, zorder=6)

    ax.set_xlim(0, duration_sec / 60.0)
    ax.set_xlabel("Time (minutes)", fontsize=11)
    ax.set_ylabel("RMS Energy", fontsize=11)
    ax.set_title(
        f"SKANN-SSL V5  —  Session Timeline\n"
        f"{wav_name}   |   Duration: {duration_sec/60:.1f} min   |   "
        f"Events detected: {n_ev}",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        plt.Line2D([0], [0], color="#333333", linewidth=1.2, label="RMS energy"),
        plt.Line2D([0], [0], color="#CC2222", linewidth=1.2,
                   linestyle="--", label="Detection threshold"),
        plt.Line2D([0], [0], color="#2255AA", linewidth=0.8,
                   linestyle=":", label="Baseline (15th pct)"),
    ] + [mpatches.Patch(color=AXIS_COLORS[c], alpha=0.75, label=c)
         for c in CLUSTER_NAMES]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=8, ncol=2, framealpha=0.92, edgecolor="#CCCCCC")

    ax2 = axes[1]
    ax2.set_facecolor("white")
    ax2.axis("off")

    if n_ev == 0:
        ax2.text(0.5, 0.5,
                 "No vessel events detected in this session.",
                 ha="center", va="center",
                 fontsize=12, color="#999999", style="italic",
                 transform=ax2.transAxes)
    else:
        col_labels = ["#", "Start\n(min)", "End\n(min)", "Dur\n(s)",
                      "Win", "Cluster", "Conf",
                      "C0", "C1", "C2", "C3", "C4"]
        col_widths = [0.030, 0.075, 0.075, 0.065, 0.045, 0.065, 0.060,
                      0.055, 0.055, 0.055, 0.055, 0.055]
        xs  = []
        acc = 0.015
        for w in col_widths:
            xs.append(acc + w / 2)
            acc += w

        for xi, lbl in zip(xs, col_labels):
            ax2.text(xi, 0.97, lbl, ha="center", va="top",
                     fontsize=8, fontweight="bold", color="#1F3864",
                     transform=ax2.transAxes, linespacing=1.3)
        ax2.plot([0.01, 0.96], [0.875, 0.875], color="#1F3864", linewidth=1.0,
                 transform=ax2.transAxes, clip_on=False)

        row_h = 0.80 / max(n_ev, 1)
        for ri, r in enumerate(results):
            y   = 0.845 - ri * row_h
            col = AXIS_COLORS[r["predicted"]]
            bg  = "#F4F7FF" if ri % 2 == 0 else "white"
            ax2.add_patch(FancyBboxPatch(
                (0.01, y - row_h * 0.52), 0.95, row_h * 0.90,
                boxstyle="square,pad=0", linewidth=0,
                facecolor=bg, transform=ax2.transAxes, zorder=0
            ))
            row_vals = [
                str(ri + 1),
                f"{r['start_sec']/60:.2f}",
                f"{r['end_sec']/60:.2f}",
                f"{r['duration_sec']:.0f}",
                str(r["n_windows"]),
                r["predicted"],
                f"{r['confidence_pct']:.1f}%",
                f"{r['prob_C0']:.2f}",
                f"{r['prob_C1']:.2f}",
                f"{r['prob_C2']:.2f}",
                f"{r['prob_C3']:.2f}",
                f"{r['prob_C4']:.2f}",
            ]
            for xi, val, cn in zip(xs, row_vals, col_labels):
                is_cluster = (cn == "Cluster")
                is_prob    = cn.strip() in CLUSTER_NAMES
                fc = col if is_cluster else (
                    AXIS_COLORS.get(cn.strip(), "#444444") if is_prob else "#222222"
                )
                fw = "bold" if is_cluster else "normal"
                ax2.text(xi, y, val, ha="center", va="center",
                         fontsize=8, color=fc, fontweight=fw,
                         transform=ax2.transAxes)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SKANN] Timeline saved : {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# BUNDLE LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_bundle(bundle_path):
    print(f"[SKANN] Loading bundle : {bundle_path}")
    bundle     = joblib.load(bundle_path)
    latent_dim = bundle["metadata"].get("latent_dim", LATENT_DIM)

    model = HybridSKEncoderV5(latent_dim=latent_dim)
    state = {k.replace("module.", ""): v
             for k, v in bundle["model_state"].items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    h_all     = bundle["embeddings_h"]
    c_labels  = bundle["cluster_labels"]
    centroids = np.stack([
        h_all[c_labels == k].mean(axis=0) for k in range(N_CLUSTERS)
    ], axis=0)

    print(f"[SKANN] Architecture   : {bundle['metadata']['architecture']}")
    print(f"[SKANN] Training events: {bundle['metrics']['n_events']}")
    print(f"[SKANN] Window         : {WINDOW_SEC}s = {WINDOW_SAMP} samples @ {FS} Hz")

    # Load deployment config — global baseline lives here, not in the bundle
    import json
    config_path     = os.path.join(os.path.dirname(os.path.abspath(bundle_path)), "deployment_config.json")
    global_baseline = None
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        global_baseline = cfg.get("rms_baseline") or cfg.get("global_baseline_rms", {}).get("value")
        n_sess = cfg.get("n_sessions", "?")
        print(f"[SKANN] Config         : {config_path}")
        print(f"[SKANN] Global baseline: {global_baseline:.8f}  (P15 across {n_sess} sessions)")
    else:
        print(f"[WARN]  deployment_config.json not found alongside bundle.")
        print(f"[WARN]  Falling back to local P15 — radar mode unreliable on pre-clipped clips.")

    return model, centroids, global_baseline


# ══════════════════════════════════════════════════════════════════════════════
# TIMELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_timeline(audio, fs, wav_name, model, centroids, device,
                 out_dir, rms_thresh, global_baseline=None):
    duration_sec = len(audio) / fs
    print(f"[SKANN] Duration       : {duration_sec/60:.1f} min")
    print(f"[SKANN] Detecting events (threshold x{rms_thresh})...")

    events, rms_frames, times_sec, threshold, baseline = detect_events(
        audio, fs, thresh_mult=rms_thresh, global_baseline=global_baseline
    )
    print(f"[SKANN] Baseline RMS   : {baseline:.6f}")
    print(f"[SKANN] Threshold      : {threshold:.6f}")
    print(f"[SKANN] Events found   : {len(events)}")

    results = []
    for ev_idx, (s_samp, e_samp) in enumerate(events):
        s_sec   = s_samp / fs
        e_sec   = e_samp / fs
        dur_sec = e_sec - s_sec
        clip_id = f"{wav_name}_{ev_idx+1:03d}"

        print(f"\n  [{ev_idx+1:3d}] {s_sec/60:.2f}-{e_sec/60:.2f} min "
              f"({dur_sec:.0f}s)  {clip_id}")

        window  = extract_best_window(audio[s_samp:e_samp])
        h       = embed_one(model, window, device)
        probs   = classify(h, centroids)
        pred    = CLUSTER_NAMES[np.argmax(probs)]
        conf    = probs.max() * 100

        print(f"       Embedded  : centre 30s window")
        print(f"       Predicted : {pred} ({conf:.1f}%)")
        print("       Probs     : "
              + "  ".join(f"{CLUSTER_NAMES[i]}={probs[i]:.3f}"
                          for i in range(N_CLUSTERS)))

        radar_path = os.path.join(out_dir, f"radar_{clip_id}.png")
        make_radar(probs, no_vessel=False, clip_id=clip_id,
                   out_path=radar_path,
                   meta=f"{s_sec/60:.2f}-{e_sec/60:.2f} min  ({dur_sec:.0f} s)")
        print(f"       Radar     : {radar_path}")

        results.append({
            "clip_id":        clip_id,
            "event_idx":      ev_idx + 1,
            "start_sec":      round(s_sec, 2),
            "end_sec":        round(e_sec, 2),
            "duration_sec":   round(dur_sec, 1),
            "n_windows":      1,
            "predicted":      pred,
            "confidence_pct": round(conf, 1),
            **{f"prob_{c}": round(float(probs[i]), 4)
               for i, c in enumerate(CLUSTER_NAMES)}
        })

    timeline_path = os.path.join(out_dir, f"timeline_{wav_name}.png")
    make_timeline(rms_frames, times_sec, threshold, baseline,
                  results, wav_name, duration_sec, timeline_path)

    csv_path = os.path.join(out_dir, f"summary_{wav_name}.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"[SKANN] CSV saved      : {csv_path}")

    print(f"\n{'='*68}")
    print(f"  SKANN-SSL V5  |  {wav_name}  |  "
          f"{duration_sec/60:.1f} min  |  {len(results)} events")
    print(f"{'='*68}")
    if results:
        print(f"  {'#':<4} {'Start':>8}  {'End':>8}  {'Dur(s)':>7}  "
              f"{'Cluster':>8}  {'Conf':>6}")
        print(f"  {'-'*55}")
        for r in results:
            print(f"  {r['event_idx']:<4} "
                  f"{r['start_sec']/60:>7.2f}m  "
                  f"{r['end_sec']/60:>7.2f}m  "
                  f"{r['duration_sec']:>7.0f}  "
                  f"{r['predicted']:>8}  "
                  f"{r['confidence_pct']:>5.1f}%")
    else:
        print("  No vessel events detected.")
        print("  Try lowering --rms-thresh (e.g. --rms-thresh 1.5)")
    print(f"{'='*68}\n")


# ══════════════════════════════════════════════════════════════════════════════
# RADAR RUNNER  (single 30-s clip)
# ══════════════════════════════════════════════════════════════════════════════

def run_radar_single(audio, fs, wav_name, model, centroids, device,
                     out_dir, rms_thresh, global_baseline=None):
    n = len(audio)
    if n < WINDOW_SAMP:
        print(f"[WARN]  Clip is {n/fs:.1f} s — zero-padding to {WINDOW_SEC} s.")
        pad     = np.zeros(WINDOW_SAMP, dtype=np.float32)
        pad[:n] = audio
        audio   = pad
    elif n > WINDOW_SAMP:
        print(f"[WARN]  Clip is {n/fs:.1f} s — truncating to {WINDOW_SEC} s.")
        audio = audio[:WINDOW_SAMP]

    events, _, _, threshold, baseline = detect_events(
        audio, fs, thresh_mult=rms_thresh, global_baseline=global_baseline
    )
    radar_path = os.path.join(out_dir, f"radar_{wav_name}.png")

    if len(events) == 0:
        print(f"[SKANN] RMS gate       : NO EVENT DETECTED")
        print(f"[SKANN] Baseline: {baseline:.6f}  Threshold: {threshold:.6f}")
        make_radar(
            cluster_probs=np.zeros(N_CLUSTERS),
            no_vessel=True,
            clip_id=wav_name,
            out_path=radar_path,
            meta=f"Duration: {WINDOW_SEC} s  |  RMS below detection threshold"
        )
    else:
        print(f"[SKANN] RMS gate       : EVENT DETECTED")
        window = extract_best_window(audio)
        h      = embed_one(model, window, device)
        probs  = classify(h, centroids)
        pred    = CLUSTER_NAMES[np.argmax(probs)]
        conf    = probs.max() * 100
        print(f"[SKANN] Predicted      : {pred} ({conf:.1f}%)")
        print("[SKANN] Probabilities  : "
              + "  ".join(f"{CLUSTER_NAMES[i]}={probs[i]:.3f}"
                          for i in range(N_CLUSTERS)))
        make_radar(
            cluster_probs=probs,
            no_vessel=False,
            clip_id=wav_name,
            out_path=radar_path,
            meta=f"Duration: {WINDOW_SEC} s"
        )

    print(f"[SKANN] Radar saved    : {radar_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SKANN-SSL V5 Vessel Acoustic Demo"
    )
    parser.add_argument("--wav",        required=True,
                        help="Path to 512 Hz WAV file")
    parser.add_argument("--bundle",     required=True,
                        help="Path to SKANN_SSL_V5_Production_Bundle.joblib")
    parser.add_argument("--out",        default="./skann_results",
                        help="Output directory (default: ./skann_results)")
    parser.add_argument("--mode",       choices=["auto", "timeline", "radar"],
                        default="auto",
                        help="auto: timeline if >35 s, radar if <=35 s")
    parser.add_argument("--rms-thresh", type=float, default=2.0,
                        help="RMS threshold multiplier (default 2.0)")
    args = parser.parse_args()

    for p in [args.wav, args.bundle]:
        if not os.path.exists(p):
            print(f"[ERROR] File not found: {p}")
            sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    print(f"\n[SKANN] Loading WAV    : {args.wav}")
    audio, file_fs = sf.read(args.wav, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_fs != FS:
        print(f"[WARN]  Sample rate {file_fs} Hz — expected {FS} Hz.")
    duration_sec = len(audio) / file_fs
    wav_name     = os.path.splitext(os.path.basename(args.wav))[0]
    print(f"[SKANN] Samples        : {len(audio):,}  |  "
          f"Duration: {duration_sec/60:.2f} min  |  fs: {file_fs} Hz")

    model, centroids, global_baseline = load_bundle(args.bundle)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    print(f"[SKANN] Device         : {device}")

    mode = args.mode
    if mode == "auto":
        mode = "radar" if duration_sec <= RADAR_MODE_MAX_SEC else "timeline"
    print(f"[SKANN] Mode           : {mode.upper()}"
          + (" (auto-selected)" if args.mode == "auto" else ""))

    if mode == "timeline":
        run_timeline(audio, file_fs, wav_name, model, centroids,
                     device, args.out, args.rms_thresh, global_baseline)
    else:
        run_radar_single(audio, file_fs, wav_name, model, centroids,
                         device, args.out, args.rms_thresh, global_baseline)


if __name__ == "__main__":
    main()
