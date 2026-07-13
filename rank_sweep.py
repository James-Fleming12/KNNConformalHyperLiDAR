import json
import math
import importlib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset.kitti.parser import Parser
from modules.HDC_utils import EllipsoidModel

import os

DATA_DIR = "/mnt/alpha/jmfleming/KITTI"
KITTIC_DIR = "/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C"
CONFIG_ARCH = "config/arch/senet-2048p.yml"
CONFIG_LABELS = "config/labels/semantic-kitti-all.yaml"
MODEL_DIR = "logs/kitti_pretrain"
PRETRAINED = "logs/kitti_pretrain/hdc_sub.pth"
NUM_CLASSES = 17

CORRUPTIONS = ["fog", "snow", "motion", "beam", "crosstalk", "echo", "cross_sensor"]
SEVERITY = 3
RANKS = (0, 4, 8, 16, 32, 64, 128, 256)
COVERAGE = 0.90
MAX_PER_CLASS = 5000
RADIUS_QUANTILE = 0.99
OUT = "rank_sweep.json"

@torch.no_grad()
def score_ellipsoid(H, mu, V, d):
    """|| M^{-1/2} (h - mu) ||_2 for each row. Lower = more in-distribution."""
    delta = H.float() - mu
    sq = delta.pow(2).sum(dim=1)
    if V.shape[1] > 0:
        proj = delta @ V
        sq = sq - (1.0 - 1.0 / d) * proj.pow(2).sum(dim=1)
    return sq.clamp_min(0).sqrt()

@torch.no_grad()
def fit_ellipsoid(Y, d, rank, coverage=COVERAGE):
    """Fit to source hypervectors Y of ONE class. rank=0 -> ball at the centroid."""
    Y = Y.float()
    n = Y.shape[0]
    mu = Y.mean(dim=0)
    delta = Y - mu

    if rank == 0:
        V = torch.zeros(d, 0, device=Y.device)
    else:
        q = max(1, int(min(rank, n - 1, d)))
        _, S, Vfull = torch.pca_lowrank(delta, q=q, center=False, niter=4)
        V = Vfull[:, :rank].contiguous()

    R = torch.quantile(score_ellipsoid(Y, mu, V, d), coverage).item()
    return {"mu": mu, "V": V, "R": R, "r": V.shape[1], "d": d}

def log_volume(e):
    """log vol(E), up to the constant log(c_d) shared by every set.
    vol = c_d * R^d * det(M)^{1/2}, and log det(M) = r * log(d), since high-variance
    dirs contribute log(d) each and low-variance dirs contribute log(1) = 0."""
    return e["d"] * math.log(max(e["R"], 1e-12)) + 0.5 * e["r"] * math.log(e["d"])

@torch.no_grad()
def collect_source(model, loader, device, max_per_class=MAX_PER_CLASS):
    buckets = {c: [] for c in range(model.num_classes)}
    counts = {c: 0 for c in range(model.num_classes)}
    for batch in loader:
        x = batch[0].to(device)
        y = batch[2].to(device).view(-1)
        enc, idx, _ = model.encode(x)
        h = F.normalize(enc)
        lab = y[idx] if idx is not None else y
        v = (lab >= 0) & (lab < model.num_classes)
        if not v.any():
            continue
        h, lab = h[v], lab[v]
        for c in lab.unique().tolist():
            if counts[c] >= max_per_class:
                continue
            hc = h[lab == c]
            take = min(hc.shape[0], max_per_class - counts[c])
            buckets[c].append(hc[:take].cpu())
            counts[c] += take
        if all(counts[c] >= max_per_class for c in range(model.num_classes)):
            break
    return {c: torch.cat(t) for c, t in buckets.items() if t}

@torch.no_grad()
def collect_target(model, loader, device, max_per_class=MAX_PER_CLASS):
    """(H, preds, correct) for one corrupted stream, using the frozen model."""
    H, P, C = [], [], []
    protos = F.normalize(model.classify.weight)
    counts = {c: 0 for c in range(model.num_classes)}
    for batch in loader:
        x = batch[0].to(device)
        y = batch[2].to(device).view(-1)
        if x.shape[1] == 0: continue
        enc, idx, _ = model.encode(x)
        h = F.normalize(enc)
        lab = y[idx] if idx is not None else y
        v = (lab >= 0) & (lab < model.num_classes)
        if not v.any():
            continue
        h, lab = h[v], lab[v]
        preds = (h.to(protos.dtype) @ protos.T).argmax(dim=1)
        
        for c in lab.unique().tolist():
            if counts[c] >= max_per_class:
                continue
            mask = (lab == c)
            hc = h[mask]
            predc = preds[mask]
            
            take = min(hc.shape[0], max_per_class - counts[c])
            H.append(hc[:take].cpu())
            P.append(predc[:take].cpu())
            C.append((predc[:take] == c).cpu())
            counts[c] += take
            
        # We can stop if we have collected enough points for all valid classes
        # Ignoring class 0 (unlabeled) which might not be used
        if all(counts[c] >= max_per_class for c in range(1, 17)):
            break
            
    if not H:
        return torch.zeros(0, model.hd_dim), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.bool)
        
    return torch.cat(H), torch.cat(P), torch.cat(C)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    ARCH = yaml.safe_load(open(CONFIG_ARCH))
    DATA = yaml.safe_load(open(CONFIG_LABELS))

    parser = Parser(
        root=DATA_DIR,
        train_sequences=DATA["split"]["train"],
        valid_sequences=DATA["split"]["valid"],
        test_sequences=None,
        labels=DATA["labels"], color_map=DATA.get("color_map", {}),
        learning_map=DATA["learning_map"], learning_map_inv=DATA["learning_map_inv"],
        sensor=ARCH["dataset"]["sensor"], max_points=ARCH["dataset"]["max_points"],
        batch_size=1, workers=0, gt=True, shuffle_train=False,
    )
    clean_ds = parser.validloader.dataset

    print(f"Loading pretrained model from {PRETRAINED}...")
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
    d = model.hd_dim
    print("Model loaded successfully.")

    print("\nCollecting source (clean) hypervectors...")
    src = collect_source(model, DataLoader(clean_ds, batch_size=1, num_workers=0), device)
    src = {c: v.to(device) for c, v in src.items()}
    print(f"  classes with data: {sorted(src.keys())}")

    tgt = {}
    for cond in CORRUPTIONS:
        print(f"Collecting target hypervectors: {cond} sev {SEVERITY}...")
        try:
            SEVERITY_MAP = {1: 'light', 3: 'moderate', 5: 'heavy'}
            sev_str = SEVERITY_MAP.get(SEVERITY, 'moderate')
            corruption_root = os.path.join(KITTIC_DIR, cond, sev_str)
            seq_dir = os.path.join(corruption_root, "sequences")
            if not os.path.exists(seq_dir):
                os.makedirs(seq_dir, exist_ok=True)
                os.symlink("..", os.path.join(seq_dir, "08"))
            
            parser_obj = Parser(
                root=corruption_root,
                train_sequences=DATA["split"]["valid"],
                valid_sequences=DATA["split"]["valid"],
                test_sequences=None,
                labels=DATA["labels"], color_map=DATA.get("color_map", {}),
                learning_map=DATA["learning_map"], learning_map_inv=DATA["learning_map_inv"],
                sensor=ARCH["dataset"]["sensor"], max_points=ARCH["dataset"]["max_points"],
                batch_size=1, workers=0, gt=True, shuffle_train=False,
            )
            ld = parser_obj.get_valid_set()
            H, P, C = collect_target(model, ld, device)
            acc = C.float().mean().item()
            print(f"  n={len(C)}  pseudo-label acc={acc:.3f}")
            if acc < 1.5 / NUM_CLASSES:
                print(f"  !! WARNING: accuracy is at/below random chance "
                      f"(~{1/NUM_CLASSES:.3f}) for {NUM_CLASSES} classes. This chunk is "
                      f"probably broken (label misalignment), and its AUROC will be "
                      f"meaningless. Exclude it from any conclusion.")
            tgt[cond] = (H, P, C)
        except Exception as e:
            print(f"  SKIPPED ({type(e).__name__}: {e})")

    print("\n" + "=" * 78)
    print(f"{'rank':>5} {'meanAUROC':>10} {'log_vol':>11}   per-corruption AUROC")
    print("=" * 78)

    results = {}
    for r in RANKS:
        ells = {c: fit_ellipsoid(Y, d, rank=r) for c, Y in src.items()}
        mlv = float(np.mean([log_volume(e) for e in ells.values()]))

        per = {}
        for cond, (H, P, C) in tgt.items():
            Hd, Pd = H.to(device), P.to(device)
            s = torch.full((Hd.shape[0],), -1e9, device=device)
            for c in Pd.unique().tolist():
                if c not in ells:
                    continue
                m = Pd == c
                e = ells[c]

                s[m] = -score_ellipsoid(Hd[m], e["mu"], e["V"], d)
            correct = C.numpy().astype(int)
            if len(np.unique(correct)) < 2:
                continue
            per[cond] = float(roc_auc_score(correct, s.cpu().numpy()))

        mean_auroc = float(np.mean(list(per.values()))) if per else float("nan")
        results[r] = {"mean_auroc": mean_auroc, "mean_log_volume": mlv,
                      "per_corruption": per}

        tag = "   <-- BALL BASELINE" if r == 0 else ""
        detail = " ".join(f"{k[:4]}={v:.3f}" for k, v in per.items())
        print(f"{r:>5} {mean_auroc:>10.4f} {mlv:>11.1f}   {detail}{tag}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {OUT}")

    base = results[0]["mean_auroc"]
    best_r = max(results, key=lambda k: results[k]["mean_auroc"])
    best = results[best_r]["mean_auroc"]
    delta = best - base

    print("\n" + "=" * 78)
    print(f"ball  (r=0):   AUROC={base:.4f}  log_vol={results[0]['mean_log_volume']:.1f}")
    print(f"best  (r={best_r}):  AUROC={best:.4f}  log_vol={results[best_r]['mean_log_volume']:.1f}")

    if delta > 0.02:
        print(f"\n=> SHAPE HELPS (+{delta:.4f} AUROC over the ball). The anisotropy")
        print(f"   hypothesis holds. Use r={best_r} as the operating point, then proceed")
        print("   to the union-of-k and adaptation experiments.")
    elif delta > 0.005:
        print(f"\n=> MARGINAL (+{delta:.4f}). Before believing it, check the per-corruption")
        print("   column: a gain driven by one corruption is not a result. It must be")
        print("   consistent across them.")
    else:
        print(f"\n=> SHAPE DOES NOT HELP (+{delta:.4f}). Anisotropy carries no information")
        print("   about pseudo-label correctness in this space. The premise is wrong --")
        print("   STOP before building the adaptation machinery.")

if __name__ == "__main__":
    main()