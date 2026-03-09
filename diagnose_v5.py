"""
diagnose_v5.py  —  SKANN-SSL V5 Training Tensor Diagnostic
============================================================
Passes actual training .npy tensors directly through the model
and checks whether they recover their correct training cluster.

Run from D:\\HAVS-SKANN-SSL-Demo\\:
    python diagnose_v5.py --bundle models\\SKANN_SSL_V5_Production_Bundle.joblib
                          --tensor_dir <path_to_v5_data/tensors/vessel>

tensor_dir: the folder on your machine or mounted Drive that contains
            the .npy files listed in window_pool_50.csv.
            e.g. G:\\MyDrive\\SKANN_SSL\\v5_data\\tensors\\vessel

Optional:
    --pool  path to ssl_data_50w/window_pool_50.csv  (for event_id lookup)
"""

import argparse, sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

# ── Model definition (must match training exactly) ────────────────────────────

def _norm_1d(channels, groups=8):
    return nn.GroupNorm(min(groups, channels), channels)

class SKConv1D(nn.Module):
    def __init__(self, in_ch, out_ch,
                 kernel_sizes=(7,15,31,63,127,255,511,1023),
                 stride=1, reduction=16, residual=False):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(in_ch, out_ch, k, stride=stride,
                      padding=k//2, bias=False) for k in kernel_sizes
        ])
        self.n_branches = len(kernel_sizes)
        self.out_ch = out_ch
        hidden = max(out_ch // reduction, 8)
        self.fc1 = nn.Linear(out_ch, hidden)
        self.fc2 = nn.Linear(hidden, out_ch * self.n_branches)
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
        a = F.softmax(z.view(z.size(0), self.n_branches, self.out_ch), dim=1).unsqueeze(-1)
        V   = (a * torch.stack(feats, dim=1)).sum(dim=1)
        out = self.act(self.norm(V))
        if self.residual:
            out = out + (x if self.match is None else self.match(x))
        return out

class SKFilterbank(nn.Module):
    KERNELS = (7,15,31,63,127,255,511,1023)
    def __init__(self, out_ch=64):
        super().__init__()
        self.stem      = SKConv1D(1, out_ch, self.KERNELS, residual=False)
        self.post_norm = _norm_1d(out_ch)
    def forward(self, x):
        return self.post_norm(self.stem(x))

def _conv2d_bn_relu(in_ch, out_ch, kernel=3, stride=(1,1)):
    pad = (kernel//2, kernel//2)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=False),
    )

class HybridSKEncoderV5(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.sk_frontend = SKFilterbank(out_ch=64)
        self.l1 = _conv2d_bn_relu(  1,  64, stride=(1,1))
        self.l2 = _conv2d_bn_relu( 64, 128, stride=(1,2))
        self.l3 = _conv2d_bn_relu(128, 256, stride=(1,2))
        self.l4 = _conv2d_bn_relu(256, 512, stride=(2,1))
        self.l5 = _conv2d_bn_relu(512, 512, stride=(2,2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Sequential(
            nn.Linear(512, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=False),
            nn.Linear(2048,2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=False),
            nn.Linear(2048, latent_dim),
        )
    def forward(self, x, return_features=False):
        if x.dim() == 1:   x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2: x = x.unsqueeze(1)
        x = self.sk_frontend(x)
        x = x.unsqueeze(1)
        x = self.l1(x); x = self.l2(x); x = self.l3(x)
        x = self.l4(x); x = self.l5(x)
        h = self.pool(x).flatten(1)
        z = self.projector(h)
        return (h, z) if return_features else z


# ── Helpers ───────────────────────────────────────────────────────────────────

def classify(h, centroids):
    dists  = np.array([np.linalg.norm(h - c) for c in centroids])
    scores = 1.0 / (dists + 0.8)
    return scores / scores.sum()


def embed_npy(model, npy_path, device):
    """Load a pre-saved training tensor and embed it.
    Matches _load_tensor() in training Cell 12:
        np.load(path).astype(float32).flatten()[:15360]
    NO re-normalisation — tensor is already z-scored.
    """
    x = np.load(npy_path).astype(np.float32)
    x = x.flatten()[:15360]
    if len(x) < 15360:
        x = np.pad(x, (0, 15360 - len(x)))
    t    = torch.from_numpy(x).float().unsqueeze(0).to(device)  # (1, 15360)
    h, _ = model(t, return_features=True)
    return h.detach().cpu().numpy().squeeze(), x


def embed_npy_with_renorm(model, npy_path, device):
    """Same but applies z-score again (double normalisation — wrong path)."""
    x = np.load(npy_path).astype(np.float32).flatten()[:15360]
    mean, std = x.mean(), x.std()
    x = (x - mean) / max(std, 1e-6)
    t    = torch.from_numpy(x).float().unsqueeze(0).to(device)
    h, _ = model(t, return_features=True)
    return h.detach().cpu().numpy().squeeze()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle",     required=True)
    parser.add_argument("--tensor_dir", required=True,
                        help="Folder containing .npy training tensors")
    parser.add_argument("--pool",       default=None,
                        help="Path to window_pool_50.csv (optional, for event_id lookup)")
    parser.add_argument("--n",          type=int, default=10,
                        help="Number of tensors to test (default 10)")
    args = parser.parse_args()

    # ── Load bundle ──
    print(f"\nLoading bundle: {args.bundle}")
    bundle    = joblib.load(args.bundle)
    h_all     = bundle["embeddings_h"]        # (141, 512)
    c_labels  = bundle["cluster_labels"]      # (141,)
    event_ids = bundle["event_ids"]           # (141,)
    n_cl      = int(c_labels.max()) + 1

    centroids = np.array([h_all[c_labels == k].mean(axis=0) for k in range(n_cl)])
    print(f"Clusters: {n_cl}  |  Training events: {len(h_all)}")

    # ── Inter-centroid distances (baseline expectation) ──
    print("\n── Inter-centroid Euclidean distances ──")
    for i in range(n_cl):
        for j in range(i+1, n_cl):
            d = np.linalg.norm(centroids[i] - centroids[j])
            print(f"  C{i}↔C{j}: {d:.4f}")

    # ── Training embedding self-check ──
    # Pass the stored h_all back through classify and check recovery
    print("\n── Training embedding → centroid recovery (stored h vectors) ──")
    correct = 0
    for idx in range(len(h_all)):
        probs   = classify(h_all[idx], centroids)
        pred    = int(np.argmax(probs))
        correct += (pred == c_labels[idx])
    print(f"  Stored h → correct cluster: {correct}/{len(h_all)}  "
          f"({100*correct/len(h_all):.1f}%)")
    print(f"  (Should be ~100% — centroids derived from same h vectors)")

    # ── Load model ──
    print(f"\nLoading model weights...")
    latent_dim = bundle["metadata"].get("latent_dim", 256)
    model = HybridSKEncoderV5(latent_dim=latent_dim)
    state = {k.replace("module.", ""): v for k, v in bundle["model_state"].items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    device = torch.device("cpu")
    print(f"  Model loaded OK  |  device: {device}")

    # ── Load window pool if provided ──
    pool_lookup = {}   # tensor_filename -> event_id
    if args.pool and os.path.exists(args.pool):
        import csv
        with open(args.pool) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pool_lookup[row["tensor_file"]] = int(row["event_id"])
        print(f"  Pool loaded: {len(pool_lookup)} entries from {args.pool}")

    # ── Find .npy files ──
    npy_files = sorted([
        f for f in os.listdir(args.tensor_dir)
        if f.endswith(".npy") and not f.startswith("aug")  # skip augmented
    ])
    if len(npy_files) == 0:
        print(f"\nERROR: No .npy files found in {args.tensor_dir}")
        sys.exit(1)

    # Sample evenly across available files
    import random
    random.seed(42)
    sample = npy_files[:args.n] if len(npy_files) <= args.n \
             else random.sample(npy_files, args.n)
    sample.sort()

    print(f"\n── Fresh embed: {len(sample)} training tensors → model → classify ──")
    print(f"  (Tensors loaded as-is — no re-normalisation, matches training _load_tensor)")
    print()
    print(f"  {'Tensor':<52} {'Shape':>10}  {'RMS':>8}  {'Pred':>5}  "
          f"{'C0':>6} {'C1':>6} {'C2':>6} {'C3':>6} {'C4':>6}  "
          f"{'GT cl':>6}  {'Match':>5}")
    print("  " + "-"*130)

    correct_fresh = 0
    for fname in sample:
        fpath  = os.path.join(args.tensor_dir, fname)
        h_new, x_raw = embed_npy(model, fpath, device)
        probs  = classify(h_new, centroids)
        pred   = int(np.argmax(probs))
        rms    = float(np.sqrt(np.mean(x_raw**2)))

        # Ground truth cluster from stored embeddings (match by event_id)
        event_id = pool_lookup.get(fname, None)
        gt_mask  = event_ids == event_id if event_id is not None else None
        gt_cl    = int(c_labels[gt_mask][0]) if (gt_mask is not None and gt_mask.any()) else -1

        match = "✓" if pred == gt_cl else ("?" if gt_cl == -1 else "✗")
        if pred == gt_cl and gt_cl != -1:
            correct_fresh += 1

        p_str = "  ".join(f"{p:.3f}" for p in probs)
        print(f"  {fname:<52} {str(x_raw.shape):>10}  {rms:>8.4f}  C{pred:>1}     "
              f"{p_str}  {f'C{gt_cl}' if gt_cl>=0 else '?':>6}  {match:>5}")

    # ── Double normalisation test ──
    print(f"\n── Double normalisation test (WRONG path — for comparison) ──")
    print(f"  (Applies z-score again to already z-scored tensor)")
    print()
    test_file = sample[0]
    fpath = os.path.join(args.tensor_dir, test_file)

    h_correct, _ = embed_npy(model, fpath, device)
    h_renorm     = embed_npy_with_renorm(model, fpath, device)

    print(f"  File: {test_file}")
    print(f"  h (no renorm):   norm={np.linalg.norm(h_correct):.4f}  "
          f"→ C{int(np.argmax(classify(h_correct, centroids)))}  "
          f"probs={[f'{p:.3f}' for p in classify(h_correct, centroids)]}")
    print(f"  h (renormed):    norm={np.linalg.norm(h_renorm):.4f}  "
          f"→ C{int(np.argmax(classify(h_renorm, centroids)))}  "
          f"probs={[f'{p:.3f}' for p in classify(h_renorm, centroids)]}")
    print(f"  h difference:    L2={np.linalg.norm(h_correct - h_renorm):.4f}")

    # ── Summary ──
    n_with_gt = sum(1 for f in sample
                    if pool_lookup.get(f) in event_ids)
    print(f"\n── Summary ──")
    print(f"  Tensors tested              : {len(sample)}")
    print(f"  With ground-truth cluster   : {n_with_gt}")
    if n_with_gt > 0:
        print(f"  Correct cluster predictions : {correct_fresh}/{n_with_gt}  "
              f"({100*correct_fresh/n_with_gt:.1f}%)")
    print(f"\n  If correct predictions << 100%: model state or architecture mismatch.")
    print(f"  If correct ~100%: demo inference chain has a bug feeding the model.")
    print(f"  If double-norm produces flat probs: that confirms double-norm was the bug.")


if __name__ == "__main__":
    main()
