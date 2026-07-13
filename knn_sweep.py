"""
knn_sweep.py -- SELF-CONTAINED. Tests proposal (A) and its controls.

WHAT THIS ANSWERS
-----------------
The rank sweep established that plain distance-to-class-mean (the r=0 ball, AUROC
0.8396) beats every anisotropic reweighting -- shrinking, deleting, isolating, or
amplifying the principal directions ALL made it worse. Second-order (covariance)
structure carries no correctness signal.

Two things the ball still cannot see, and this script tests both:

  1. LOCAL / NON-CONVEX structure. The ball is a unimodal parametric summary. k-NN
     is nonparametric. A class can be perfectly isotropic in aggregate covariance
     (so ellipsoids buy nothing) yet still have local density structure.

  2. CONTRASTIVENESS. Every gate tested so far asks "how close is this point to its
     own class?" The ratio d_in / d_out asks "is it closer to its OWN class than to
     OTHER classes?" -- which is far more directly aligned with pseudo-label
     CORRECTNESS, since a label is wrong exactly when another class fits better.
     This is the genuinely new signal type, and I expect it to be where the gain
     comes from (if there is one).

THE BUILT-IN CONTROL
--------------------
For L2-normalized hypervectors, the mean SQUARED distance to all n class points is

    E_i[ ||h - x_i||^2 ] = ||h - mu||^2 + trace(Sigma_c)

so as k -> n, the in-class kNN score converges to the BALL SCORE plus a per-class
constant. The ball is therefore the k=n limit of `knn_in`, recovered for free at the
top of the sweep. If `knn_in` at large k does NOT approach the ball's AUROC, there is
a bug.

THE ABLATION THAT DECIDES THE MECHANISM
---------------------------------------
    knn_in     -> local density only          (tests "non-convex shape")
    knn_ratio  -> d_in / d_out                (the full proposal: + contrastive)
    margin     -> top1 - top2 prototype sim   (the CHEAPEST contrastive score)

If `margin` matches `knn_ratio`, the win is entirely "contrastive helps" and you do
NOT need k-NN -- you need a two-line change to the existing gate, with no 85k-point
neighbor bank to carry at test time. That is the outcome to hope for.

Usage:
    CUDA_VISIBLE_DEVICES=3 uv run knn_sweep.py
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset.kitti.parser import Parser
from modules.HDC_utils import EllipsoidModel

DATA_DIR = "/mnt/alpha/jmfleming/KITTI"
KITTIC_DIR = "/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C"
CONFIG_ARCH = "config/arch/senet-2048p.yml"
CONFIG_LABELS = "config/labels/semantic-kitti-all.yaml"
MODEL_DIR = "logs/kitti_pretrain"
PRETRAINED = "logs/kitti_pretrain/hdc_sub.pth"
NUM_CLASSES = 17

CORRUPTIONS = ["snow", "cross_sensor"]     # the two that survived the sanity filter
SEVERITY = 3
KS = (1, 5, 10, 25, 50, 100, 500, 2000)
SRC_PER_CLASS = 3000          # neighbor bank size per class (memory/compute knob)
TGT_PER_CLASS = 3000
CHUNK = 1024                  # target points scored at a time (OOM control)
OUT = "knn_sweep.json"


# ============================================================== scores

@torch.no_grad()
def ball_score(H, mu):
    """||h - mu||. The r=0 baseline: AUROC 0.8396."""
    return (H - mu).norm(dim=1)


@torch.no_grad()
def knn_dists(H, bank, k, chunk=CHUNK):
    """Mean cosine distance from each row of H to its k nearest neighbours in `bank`.

    Hypervectors are L2-normalized, so cos_dist = 1 - h.x  and kNN-cosine is
    order-equivalent to kNN-Euclidean. Chunked to keep the |H| x |bank| matrix off
    the GPU all at once.
    """
    out = torch.empty(H.shape[0], device=H.device, dtype=H.dtype)
    kk = min(k, bank.shape[0])
    for i in range(0, H.shape[0], chunk):
        h = H[i:i + chunk]
        sims = h @ bank.T                       # (chunk, n_bank)
        topk = sims.topk(kk, dim=1).values      # nearest = HIGHEST similarity
        out[i:i + chunk] = (1.0 - topk).mean(dim=1)
    return out


# ============================================================== data

@torch.no_grad()
def collect_source(model, loader, device, cap):
    buckets = {c: [] for c in range(model.num_classes)}
    counts = {c: 0 for c in range(model.num_classes)}
    for batch in loader:
        x = batch[0].to(device); y = batch[2].to(device).view(-1)
        if x.shape[1] == 0:
            continue
        enc, idx, _ = model.encode(x)
        h = F.normalize(enc)
        lab = y[idx] if idx is not None else y
        v = (lab >= 0) & (lab < model.num_classes)
        if not v.any():
            continue
        h, lab = h[v], lab[v]
        for c in lab.unique().tolist():
            if counts[c] >= cap:
                continue
            hc = h[lab == c]
            take = min(hc.shape[0], cap - counts[c])
            buckets[c].append(hc[:take].cpu())
            counts[c] += take
        if all(counts[c] >= cap for c in range(1, model.num_classes)):
            break
    return {c: torch.cat(t) for c, t in buckets.items() if t}


@torch.no_grad()
def collect_target(model, loader, device, cap):
    """Returns H, preds, correct, and the top-2 prototype sims (for the margin arm)."""
    H, P, C, M = [], [], [], []
    protos = F.normalize(model.classify.weight)
    counts = {c: 0 for c in range(model.num_classes)}
    for batch in loader:
        x = batch[0].to(device); y = batch[2].to(device).view(-1)
        if x.shape[1] == 0:
            continue
        enc, idx, _ = model.encode(x)
        h = F.normalize(enc)
        lab = y[idx] if idx is not None else y
        v = (lab >= 0) & (lab < model.num_classes)
        if not v.any():
            continue
        h, lab = h[v], lab[v]
        sims = h.to(protos.dtype) @ protos.T
        top2 = sims.topk(2, dim=1).values
        preds = sims.argmax(dim=1)
        margin = top2[:, 0] - top2[:, 1]

        for c in lab.unique().tolist():
            if counts[c] >= cap:
                continue
            m = lab == c
            take = min(int(m.sum()), cap - counts[c])
            H.append(h[m][:take].cpu())
            P.append(preds[m][:take].cpu())
            C.append((preds[m][:take] == c).cpu())
            M.append(margin[m][:take].cpu())
            counts[c] += take
        if all(counts[c] >= cap for c in range(1, model.num_classes)):
            break
    return torch.cat(H), torch.cat(P), torch.cat(C), torch.cat(M)


def per_class_auroc(scores, correct, preds, valid_classes, min_n=50):
    """Macro-average AUROC over predicted classes. Higher score = more trustworthy.

    Per-class (not pooled) because a pooled AUROC is dominated by WHICH class was
    predicted -- easy classes are both more accurate and differently scaled -- which
    swamps the within-class geometry we are actually testing.
    """
    out = []
    for c in np.unique(preds):
        if c not in valid_classes:
            continue
        m = preds == c
        if m.sum() < min_n:
            continue
        cc = correct[m]
        if len(np.unique(cc)) < 2:
            continue
        out.append(roc_auc_score(cc, scores[m]))
    return (float(np.mean(out)), len(out)) if out else (float("nan"), 0)


# ============================================================== main

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ARCH = yaml.safe_load(open(CONFIG_ARCH))
    DATA = yaml.safe_load(open(CONFIG_LABELS))

    parser = Parser(
        root=DATA_DIR, train_sequences=DATA["split"]["train"],
        valid_sequences=DATA["split"]["valid"], test_sequences=None,
        labels=DATA["labels"], color_map=DATA.get("color_map", {}),
        learning_map=DATA["learning_map"], learning_map_inv=DATA["learning_map_inv"],
        sensor=ARCH["dataset"]["sensor"], max_points=ARCH["dataset"]["max_points"],
        batch_size=1, workers=0, gt=True, shuffle_train=False)
    clean_ds = parser.validloader.dataset

    model = EllipsoidModel(ARCH, MODEL_DIR, "rp", 0, 0, NUM_CLASSES, device)
    sd = torch.load(PRETRAINED, map_location=device, weights_only=False)
    sd = sd.state_dict() if isinstance(sd, torch.nn.Module) else sd
    if "subclusters" in sd and hasattr(model, "subclusters"):
        n_sub = sd["subclusters"].shape[0]
        if model.subclusters.shape[0] != n_sub:
            model.subclusters = torch.nn.Parameter(
                torch.zeros(n_sub, model.hd_dim, device=device))
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    print("Model loaded.")

    print("Collecting source bank...")
    src = collect_source(model, DataLoader(clean_ds, batch_size=1, num_workers=0),
                         device, SRC_PER_CLASS)
    src = {c: v.to(device) for c, v in src.items()}
    mus = {c: v.mean(0) for c, v in src.items()}
    valid = set(src.keys())
    print(f"  classes: {sorted(valid)}  bank sizes: "
          f"{ {c: v.shape[0] for c, v in src.items()} }")

    tgt = {}
    for cond in CORRUPTIONS:
        sev = {1: "light", 3: "moderate", 5: "heavy"}[SEVERITY]
        root = os.path.join(KITTIC_DIR, cond, sev)
        s08 = os.path.join(root, "sequences", "08")
        if not os.path.exists(s08):
            os.makedirs(os.path.dirname(s08), exist_ok=True)
            os.symlink("..", s08)
        p = Parser(root=root, train_sequences=DATA["split"]["valid"],
                   valid_sequences=DATA["split"]["valid"], test_sequences=None,
                   labels=DATA["labels"], color_map=DATA.get("color_map", {}),
                   learning_map=DATA["learning_map"],
                   learning_map_inv=DATA["learning_map_inv"],
                   sensor=ARCH["dataset"]["sensor"],
                   max_points=ARCH["dataset"]["max_points"],
                   batch_size=1, workers=0, gt=True, shuffle_train=False)
        H, P, C, M = collect_target(model, p.get_valid_set(), device, TGT_PER_CLASS)
        print(f"{cond}: n={len(C)}  acc={C.float().mean():.3f}")
        tgt[cond] = (H.to(device), P.to(device), C.numpy().astype(int),
                     M.numpy())

    results = {}

    # ---------------- non-kNN arms (cheap; these are the controls) -------------
    print("\n" + "=" * 76)
    print("BASELINE ARMS")
    print("=" * 76)
    protos = F.normalize(model.classify.weight)
    for arm in ("ball", "prototype", "margin"):
        row = {}
        for cond, (H, P, C, M) in tgt.items():
            if arm == "ball":
                s = torch.full((H.shape[0],), -1e9, device=device, dtype=H.dtype)
                for c in P.unique().tolist():
                    if c not in valid:
                        continue
                    m = P == c
                    s[m] = -ball_score(H[m], mus[c])       # negate: distance -> trust
                s = s.cpu().numpy()
            elif arm == "prototype":
                sims = H.to(protos.dtype) @ protos.T
                s = sims.gather(1, P.unsqueeze(1)).squeeze(1).cpu().numpy()
            else:  # margin = top1 - top2 prototype similarity
                s = M
            a, nc = per_class_auroc(s, C, P.cpu().numpy(), valid)
            row[cond] = a
        results[arm] = row
        print(f"{arm:>12}  " + "  ".join(f"{k}={v:.4f}" for k, v in row.items())
              + f"   mean={np.mean(list(row.values())):.4f}")

    # ---------------- kNN arms -------------------------------------------------
    print("\n" + "=" * 76)
    print("kNN ARMS   (knn_in should approach the BALL as k -> bank size)")
    print("=" * 76)
    print(f"{'k':>6} {'knn_in':>18} {'knn_ratio':>18}")

    for k in KS:
        row_in, row_ratio = {}, {}
        for cond, (H, P, C, M) in tgt.items():
            s_in = torch.full((H.shape[0],), 1e9, device=device, dtype=H.dtype)
            s_ratio = torch.full((H.shape[0],), 1e9, device=device, dtype=H.dtype)
            for c in P.unique().tolist():
                if c not in valid:
                    continue
                m = P == c
                Hc = H[m]
                d_in = knn_dists(Hc, src[c], k)
                # out-of-class bank: every OTHER class's source points
                out_bank = torch.cat([src[o] for o in valid if o != c], dim=0)
                d_out = knn_dists(Hc, out_bank, k)
                s_in[m] = d_in
                s_ratio[m] = d_in / d_out.clamp_min(1e-8)
            # negate: these are DISTANCES / distance-ratios (lower = more trustworthy)
            a_in, _ = per_class_auroc((-s_in).cpu().numpy(), C, P.cpu().numpy(), valid)
            a_rt, _ = per_class_auroc((-s_ratio).cpu().numpy(), C, P.cpu().numpy(), valid)
            row_in[cond] = a_in
            row_ratio[cond] = a_rt

        results[f"knn_in_k{k}"] = row_in
        results[f"knn_ratio_k{k}"] = row_ratio
        mi = np.mean(list(row_in.values()))
        mr = np.mean(list(row_ratio.values()))
        print(f"{k:>6} {mi:>18.4f} {mr:>18.4f}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {OUT}")

    # ---------------- verdict --------------------------------------------------
    ball = np.mean(list(results["ball"].values()))
    marg = np.mean(list(results["margin"].values()))
    best_ratio_k = max(KS, key=lambda k: np.mean(list(results[f"knn_ratio_k{k}"].values())))
    best_ratio = np.mean(list(results[f"knn_ratio_k{best_ratio_k}"].values()))
    best_in_k = max(KS, key=lambda k: np.mean(list(results[f"knn_in_k{k}"].values())))
    best_in = np.mean(list(results[f"knn_in_k{best_in_k}"].values()))

    print("\n" + "=" * 76)
    print(f"ball                 {ball:.4f}   <- the number to beat")
    print(f"margin (top1-top2)   {marg:.4f}   <- cheapest contrastive score")
    print(f"knn_in   (k={best_in_k})       {best_in:.4f}   <- local density only")
    print(f"knn_ratio(k={best_ratio_k})       {best_ratio:.4f}   <- proposal A")
    print("=" * 76)
    print("""
HOW TO READ IT:
  knn_ratio > margin > ball   -> contrastive is the mechanism AND kNN adds something
                                 on top of it. Proposal A is justified in full.
  knn_ratio ~= margin > ball  -> contrastive is the WHOLE mechanism. Drop the kNN
                                 bank entirely and just use top1-top2 margin. This is
                                 the best outcome: a two-line gate, no neighbor bank
                                 at test time.
  knn_in > ball               -> local/non-convex density DOES carry signal that the
                                 unimodal ball misses.
  nothing beats ball          -> distance-to-mean is already the best available gate.
                                 That is itself a clean, defensible result: HDC's
                                 representation is isotropic in the directions that
                                 matter, and no geometric refinement helps.
""")


if __name__ == "__main__":
    main()
