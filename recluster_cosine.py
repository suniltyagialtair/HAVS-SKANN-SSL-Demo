"""
recluster_cosine.py  —  SKANN-SSL V5 Cosine Recluster Diagnostic
=================================================================
Reclusters the stored h_all embeddings using cosine metric (spherical KMeans)
and tests whether fresh tensor embeddings recover training cluster assignments.

Also tests cosine-similarity classification vs Euclidean-distance classification.

Run from D:\\HAVS-SKANN-SSL-Demo\\:
    python recluster_cosine.py ^
        --bundle models\\SKANN_SSL_V5_Production_Bundle.joblib ^
        --tensor_dir "G:\\My Drive\\SKANN_SSL\\v5_data\\tensors\\vessel" ^
        --pool "G:\\My Drive\\SKANN_SSL\\ssl_data_50w\\window_pool_50.csv"
"""

import argparse, os, sys, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# ── Model (identical to diagnose_v5.py) ──────────────────────────────────────

def _norm_1d(ch, g=8): return nn.GroupNorm(min(g, ch), ch)

class SKConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, ks=(7,15,31,63,127,255,511,1023),
                 stride=1, reduction=16, residual=False):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(in_ch, out_ch, k, stride=stride, padding=k//2, bias=False)
            for k in ks])
        self.n = len(ks); self.out_ch = out_ch
        hid = max(out_ch//reduction, 8)
        self.fc1 = nn.Linear(out_ch, hid)
        self.fc2 = nn.Linear(hid, out_ch*self.n)
        self.norm = _norm_1d(out_ch); self.act = nn.GELU()
        self.residual = residual; self.match = None
        if residual and in_ch != out_ch:
            self.match = nn.Conv1d(in_ch, out_ch, 1, bias=False)
    def forward(self, x):
        feats = [b(x) for b in self.branches]
        U = torch.stack(feats, 1).sum(1)
        s = F.adaptive_avg_pool1d(U, 1).squeeze(-1)
        z = self.fc2(F.relu(self.fc1(s), inplace=False))
        a = F.softmax(z.view(z.size(0), self.n, self.out_ch), 1).unsqueeze(-1)
        V = (a * torch.stack(feats, 1)).sum(1)
        out = self.act(self.norm(V))
        if self.residual:
            out = out + (x if self.match is None else self.match(x))
        return out

class SKFilterbank(nn.Module):
    KERNELS = (7,15,31,63,127,255,511,1023)
    def __init__(self, out_ch=64):
        super().__init__()
        self.stem = SKConv1D(1, out_ch, self.KERNELS, residual=False)
        self.post_norm = _norm_1d(out_ch)
    def forward(self, x): return self.post_norm(self.stem(x))

def _cb(ic, oc, k=3, stride=(1,1)):
    return nn.Sequential(
        nn.Conv2d(ic, oc, k, stride=stride, padding=k//2, bias=False),
        nn.BatchNorm2d(oc), nn.ReLU(inplace=False))

class HybridSKEncoderV5(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.sk_frontend = SKFilterbank(out_ch=64)
        self.l1 = _cb(1,64,stride=(1,1)); self.l2 = _cb(64,128,stride=(1,2))
        self.l3 = _cb(128,256,stride=(1,2)); self.l4 = _cb(256,512,stride=(2,1))
        self.l5 = _cb(512,512,stride=(2,2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Sequential(
            nn.Linear(512,2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=False),
            nn.Linear(2048,2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=False),
            nn.Linear(2048, latent_dim))
    def forward(self, x, return_features=False):
        if x.dim()==1:   x=x.unsqueeze(0).unsqueeze(0)
        elif x.dim()==2: x=x.unsqueeze(1)
        x = self.sk_frontend(x); x = x.unsqueeze(1)
        x = self.l1(x); x = self.l2(x); x = self.l3(x)
        x = self.l4(x); x = self.l5(x)
        h = self.pool(x).flatten(1); z = self.projector(h)
        return (h, z) if return_features else z

# ── Classifiers ───────────────────────────────────────────────────────────────

def classify_euclidean(h, centroids):
    dists  = np.array([np.linalg.norm(h - c) for c in centroids])
    scores = 1.0 / (dists + 0.8)
    return scores / scores.sum()

def classify_cosine(h, centroids_norm):
    """centroids_norm: L2-normalised centroid matrix (n_clusters, 512)"""
    h_n = h / (np.linalg.norm(h) + 1e-9)
    sims = centroids_norm @ h_n          # dot product = cosine similarity
    # softmax with temperature to convert sims to probs
    temp = 10.0
    s = sims * temp - (sims * temp).max()
    e = np.exp(s)
    return e / e.sum()

# ── Embed ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def embed_npy(model, path, device):
    x = np.load(path).astype(np.float32).flatten()[:15360]
    if len(x) < 15360: x = np.pad(x, (0, 15360-len(x)))
    t = torch.from_numpy(x).float().unsqueeze(0).to(device)
    h, _ = model(t, return_features=True)
    return h.detach().cpu().numpy().squeeze()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle",     required=True)
    ap.add_argument("--tensor_dir", required=True)
    ap.add_argument("--pool",       required=True)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    # Load bundle
    bundle   = joblib.load(args.bundle)
    h_all    = bundle["embeddings_h"]          # (141, 512)
    c_labels = bundle["cluster_labels"]        # (141,)  — original Euclidean KMeans
    eids     = bundle["event_ids"]
    N_CL     = int(c_labels.max()) + 1

    print(f"Training events: {len(h_all)}  |  Clusters: {N_CL}")
    print(f"h_all norm stats: min={np.linalg.norm(h_all,axis=1).min():.4f}  "
          f"max={np.linalg.norm(h_all,axis=1).max():.4f}  "
          f"mean={np.linalg.norm(h_all,axis=1).mean():.4f}")

    # ── Cosine recluster on stored h_all ─────────────────────────────────────
    print("\n── Cosine recluster of stored h_all ──")
    h_norm = normalize(h_all, norm='l2')   # unit-sphere projection
    km_cos = KMeans(n_clusters=N_CL, random_state=42, n_init=20)
    cos_labels = km_cos.fit_predict(h_norm)

    # Normalised cosine centroids
    cos_centroids_norm = normalize(km_cos.cluster_centers_, norm='l2')

    # Agreement with original Euclidean labels (up to permutation)
    from scipy.optimize import linear_sum_assignment
    # Build confusion matrix between original and cosine labels
    conf = np.zeros((N_CL, N_CL), dtype=int)
    for a, b in zip(c_labels, cos_labels):
        conf[a, b] += 1
    row_ind, col_ind = linear_sum_assignment(-conf)
    agreement = conf[row_ind, col_ind].sum()
    print(f"  Cosine cluster sizes: {[int((cos_labels==k).sum()) for k in range(N_CL)]}")
    print(f"  Original Euclidean:   {[int((c_labels==k).sum()) for k in range(N_CL)]}")
    print(f"  Label agreement (best permutation): {agreement}/{len(h_all)} "
          f"({100*agreement/len(h_all):.1f}%)")

    # Recovery: stored h_all → cosine centroid
    correct_cos = sum(
        int(np.argmax(classify_cosine(h_all[i], cos_centroids_norm))) == cos_labels[i]
        for i in range(len(h_all))
    )
    print(f"  Stored h → cosine centroid recovery: {correct_cos}/{len(h_all)}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("\n── Loading model ──")
    latent_dim = bundle["metadata"].get("latent_dim", 256)
    model = HybridSKEncoderV5(latent_dim=latent_dim)
    state = {k.replace("module.",""): v for k, v in bundle["model_state"].items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    device = torch.device("cpu")
    print(f"  OK")

    # ── Load pool ─────────────────────────────────────────────────────────────
    import csv
    pool = {}
    with open(args.pool) as f:
        for row in csv.DictReader(f):
            pool[row["tensor_file"]] = int(row["event_id"])

    # Map event_id → original cluster label
    eid_to_orig = {int(eids[i]): int(c_labels[i]) for i in range(len(eids))}
    eid_to_cos  = {int(eids[i]): int(cos_labels[i]) for i in range(len(eids))}

    # ── Sample tensors ────────────────────────────────────────────────────────
    npy_files = sorted([f for f in os.listdir(args.tensor_dir)
                        if f.endswith(".npy") and "aug" not in f])
    random.seed(42)
    sample = npy_files[:args.n] if len(npy_files) <= args.n \
             else random.sample(npy_files, args.n)
    sample.sort()

    # Original Euclidean centroids
    euc_centroids = np.array([h_all[c_labels==k].mean(axis=0) for k in range(N_CL)])

    print(f"\n── Fresh embed: {len(sample)} tensors ──")
    print(f"  {'Tensor':<45} {'GT':>4}  "
          f"{'Euc pred':>8} {'Euc✓':>5}  "
          f"{'Cos pred':>8} {'Cos✓':>5}")
    print("  " + "-"*80)

    euc_correct = cos_correct = total_gt = 0
    for fname in sample:
        fpath    = os.path.join(args.tensor_dir, fname)
        h        = embed_npy(model, fpath, device)
        eid      = pool.get(fname)
        gt_orig  = eid_to_orig.get(eid, -1)
        gt_cos   = eid_to_cos.get(eid, -1)

        pred_euc = int(np.argmax(classify_euclidean(h, euc_centroids)))
        pred_cos = int(np.argmax(classify_cosine(h, cos_centroids_norm)))

        if gt_orig >= 0:
            total_gt += 1
            euc_correct += (pred_euc == gt_orig)
            cos_correct += (pred_cos == gt_cos)

        gt_str   = f"C{gt_orig}" if gt_orig >= 0 else "?"
        euc_mark = "✓" if pred_euc == gt_orig else "✗"
        cos_mark = "✓" if pred_cos == gt_cos  else "✗"

        print(f"  {fname:<45} {gt_str:>4}  "
              f"C{pred_euc:>1} {euc_mark:>5}     "
              f"C{pred_cos:>1} {cos_mark:>5}")

    print(f"\n── Recovery summary ──")
    if total_gt > 0:
        print(f"  Euclidean: {euc_correct}/{total_gt} ({100*euc_correct/total_gt:.1f}%)")
        print(f"  Cosine:    {cos_correct}/{total_gt} ({100*cos_correct/total_gt:.1f}%)")
        print()
        if cos_correct > euc_correct:
            print("  RESULT: Cosine classification is superior.")
            print("  ACTION: Switch demo to cosine-similarity classify + cosine recluster.")
        elif cos_correct == euc_correct:
            print("  RESULT: No difference — both metrics produce similar recovery.")
            print("  The fundamental issue is inter-centroid distance, not metric choice.")
            print("  V5 embeddings are compressed by tonal leak regardless of metric.")
        else:
            print("  RESULT: Euclidean is better — unexpected.")

    # ── Cosine similarity between training h_all ──────────────────────────────
    print(f"\n── Cosine similarity stats (stored h_all) ──")
    same, diff = [], []
    for i in range(len(h_all)):
        for j in range(i+1, len(h_all)):
            hi = h_all[i] / (np.linalg.norm(h_all[i]) + 1e-9)
            hj = h_all[j] / (np.linalg.norm(h_all[j]) + 1e-9)
            sim = float(hi @ hj)
            if c_labels[i] == c_labels[j]: same.append(sim)
            else: diff.append(sim)
    print(f"  Same-cluster cosine sim:  mean={np.mean(same):.4f}  "
          f"std={np.std(same):.4f}  min={np.min(same):.4f}")
    print(f"  Diff-cluster cosine sim:  mean={np.mean(diff):.4f}  "
          f"std={np.std(diff):.4f}  min={np.min(diff):.4f}")
    print(f"  Cosine sim gap (same-diff): {np.mean(same)-np.mean(diff):.4f}")
    print()
    if np.mean(same) - np.mean(diff) < 0.05:
        print("  VERDICT: Cosine sim gap < 0.05 — embeddings not separable by any")
        print("  clustering metric. This is the tonal leak signature.")
        print("  V5 classification is fundamentally limited. V6 (notch filter) required.")
    else:
        print("  VERDICT: Meaningful cosine gap exists — recluster with cosine should help.")

if __name__ == "__main__":
    main()
