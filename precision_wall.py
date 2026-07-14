"""
precision_wall.py -- DOES THE METHOD HAVE A MATHEMATICAL FUTURE?

This script answers the three critical blockers before any new method is built:
1. THE PRECISION WALL: At what precision does the +5.21 oracle gain vanish?
2. PER-CLASS HEADROOM: Are the rare classes actually fixable, or is it a feature issue?
3. DRIFT DIAGNOSTIC: Why does drift hit 0.765 regardless of learning rate?
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import yaml

class Logger(object):
    def __init__(self, filename="precision_wall.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("precision_wall.log")
sys.stderr = sys.stdout

from dataset.kitti.parser import Parser
from modules.HDC_utils import set_knn_model

KITTIC_DIR = "/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C"
DATA_DIR = "/mnt/alpha/jmfleming/KITTI"
CONFIG_ARCH = "config/arch/senet-2048p.yml"
CONFIG_LABELS = "config/labels/semantic-kitti-all.yaml"
PRETRAINED = "logs/kitti_pretrain/hdc_sub.pth"
NUM_CLASSES = 17

N_FRAMES = 2000
SEEDS = [0, 1, 2]
PRECISIONS = [100, 98, 95, 90, 85, 82, 80, 75, 70]
LR = 0.01

def fast_hist(preds, labels, K):
    m = (labels >= 0) & (labels < K)
    return torch.bincount(K * labels[m].long() + preds[m].long(),
                          minlength=K ** 2).reshape(K, K)

def per_class_iou(hist):
    tp = torch.diag(hist).float()
    fp = hist.sum(0).float() - tp
    fn = hist.sum(1).float() - tp
    return tp / (tp + fp + fn + 1e-10)

def live_miou(iou, live):
    return float(np.mean([iou[c] for c in live])) * 100

@torch.no_grad()
def encode_frame(model, x, y):
    enc, _, _ = model.encode(x)
    valid = torch.any(
        x.permute(0, 2, 3, 1).contiguous().reshape(-1, x.shape[1]) != 0, dim=1)
    if not valid.any():
        return None
    h = F.normalize(enc[valid], dim=1).to(model.classify.weight.dtype)
    protos = F.normalize(model.classify.weight, dim=1)
    sims = h @ protos.T
    return h, sims, sims.argmax(1), y[valid]

@torch.no_grad()
def run_precision_oracle(model, loader, device, src, live, precision_pct, seed=0, mode="oracle"):
    """
    Simulates a gate with EXACTLY `precision_pct` (pooled) precision.
    We admit all points where `pred == label`, but we also admit real false positives 
    to exactly hit the target precision. This perfectly simulates the systematic bias 
    of real label errors (adjacent classes, ambiguous features).
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    is_eval = rng.rand(N_FRAMES) < 0.5
    model.classify.weight.data = src.clone()

    hist = torch.zeros((NUM_CLASSES, NUM_CLASSES), device=device)
    
    global_tp = 0
    global_fp = 0
    pc_tp = torch.zeros(NUM_CLASSES, device=device)
    pc_fp = torch.zeros(NUM_CLASSES, device=device)
    
    for i, b in enumerate(loader):
        if i >= N_FRAMES:
            break
        x, y = b[0].to(device), b[2].to(device).view(-1)
        if x.shape[1] == 0:
            continue
        out = encode_frame(model, x, y)
        if out is None:
            continue
        h, sims, preds, labels = out

        if is_eval[i]:
            hist += fast_hist(preds, labels, NUM_CLASSES)
            continue

        if mode == "frozen":
            continue
            
        correct = preds == labels
        wrong = ~correct
        
        admit = correct.clone()
        if precision_pct < 100:
            P = precision_pct / 100.0
            n_correct = int(correct.sum())
            if n_correct > 0:
                target_wrong = int(n_correct * (1.0 - P) / P)
                wrong_idx = wrong.nonzero(as_tuple=True)[0]
                if len(wrong_idx) > 0:
                    if len(wrong_idx) > target_wrong:
                        wrong_idx = wrong_idx[torch.randperm(len(wrong_idx), device=device)[:target_wrong]]
                    admit[wrong_idx] = True
                    
        global_tp += int(correct[admit].sum())
        global_fp += int(wrong[admit].sum())
        for c in range(NUM_CLASSES):
            m = preds == c
            pc_tp[c] += (m & correct & admit).sum()
            pc_fp[c] += (m & wrong & admit).sum()
                
        # Prototype Update
        for c in preds[admit].unique().tolist():
            m = (preds == c) & admit
            if m.sum() < 10:
                continue
            pull = h[m].mean(0)
            w_new = model.classify.weight[c] + LR * pull
            model.classify.weight[c] = F.normalize(w_new.unsqueeze(0), dim=1).squeeze(0)

    iou = per_class_iou(hist).cpu().numpy()
    drift = F.cosine_similarity(F.normalize(model.classify.weight, dim=1), F.normalize(src, dim=1), dim=1)
    
    eps = 1e-10
    pooled_prec = global_tp / (global_tp + global_fp + eps)
    pc_prec = pc_tp / (pc_tp + pc_fp + eps)
    macro_prec = float(np.mean([pc_prec[c].item() for c in live])) * 100
    
    return {
        "miou": live_miou(iou, live),
        "pc_iou": iou * 100,
        "mean_drift": float(drift.mean()),
        "pc_drift": drift.cpu().numpy(),
        "macro_prec": macro_prec,
        "pooled_prec": pooled_prec * 100,
    }

@torch.no_grad()
def run_supervised_ceiling(model, loader, device, src, live):
    """Refit prototypes FROM SCRATCH on target data using GT labels."""
    rng = np.random.RandomState(0)
    is_eval = rng.rand(N_FRAMES) < 0.5
    
    acc = torch.zeros_like(src)
    cnt = torch.zeros(NUM_CLASSES, device=device)
    for i, b in enumerate(loader):
        if i >= N_FRAMES:
            break
        if is_eval[i]:
            continue
        x, y = b[0].to(device), b[2].to(device).view(-1)
        if x.shape[1] == 0:
            continue
        out = encode_frame(model, x, y)
        if out is None:
            continue
        h, _, _, labels = out
        for c in labels.unique().tolist():
            if c < 0 or c >= NUM_CLASSES:
                continue
            m = labels == c
            acc[c] += h[m].sum(0)
            cnt[c] += m.sum()

    new = src.clone()
    for c in range(NUM_CLASSES):
        if cnt[c] > 100:
            new[c] = F.normalize(acc[c].unsqueeze(0), dim=1).squeeze(0)
    model.classify.weight.data = new

    hist = torch.zeros((NUM_CLASSES, NUM_CLASSES), device=device)
    for i, b in enumerate(loader):
        if i >= N_FRAMES:
            break
        if not is_eval[i]:
            continue
        x, y = b[0].to(device), b[2].to(device).view(-1)
        if x.shape[1] == 0:
            continue
        out = encode_frame(model, x, y)
        if out is None:
            continue
        _, _, preds, labels = out
        hist += fast_hist(preds, labels, NUM_CLASSES)
        
    iou = per_class_iou(hist).cpu().numpy()
    model.classify.weight.data = src.clone()
    return {"miou": live_miou(iou, live), "pc_iou": iou * 100}

def get_loader(cond, ARCH, DATA):
    corr, sev = cond.split("/")
    root = os.path.join(KITTIC_DIR, corr, sev)
    s08 = os.path.join(root, "sequences", "08")
    if not os.path.exists(s08):
        os.makedirs(os.path.dirname(s08), exist_ok=True)
        try:
            os.symlink("..", s08)
        except FileExistsError:
            pass
    p = Parser(root=root, train_sequences=DATA["split"]["valid"],
               valid_sequences=DATA["split"]["valid"], test_sequences=None,
               labels=DATA["labels"], color_map=DATA.get("color_map", {}),
               learning_map=DATA["learning_map"],
               learning_map_inv=DATA["learning_map_inv"],
               sensor=ARCH["dataset"]["sensor"],
               max_points=ARCH["dataset"]["max_points"],
               batch_size=1, workers=0, gt=True, shuffle_train=False)
    return p.validloader

def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ARCH = yaml.safe_load(open(CONFIG_ARCH))
    DATA = yaml.safe_load(open(CONFIG_LABELS))
    model = set_knn_model(ARCH, "logs/kitti_pretrain", "rp", 0, 0, NUM_CLASSES, dev)
    model.load_state_dict(torch.load(PRETRAINED, map_location=dev), strict=False)
    model.to(dev).eval()
    src = model.classify.weight.data.clone()

    loader = get_loader("snow/light", ARCH, DATA)

    print("Evaluating frozen baseline to determine live classes...")
    f0 = run_precision_oracle(model, loader, dev, src, list(range(NUM_CLASSES)), 100, mode="frozen", seed=0)
    live = [c for c in range(NUM_CLASSES) if f0["pc_iou"][c] >= 1.0]
    
    print("\n" + "=" * 78)
    print("1. DRIFT DIAGNOSTIC: Is the metric broken?")
    print("=" * 78)
    print(f"Frozen global drift: {f0['mean_drift']:.4f}")
    
    orc100 = run_precision_oracle(model, loader, dev, src, live, 100, mode="oracle", seed=0)
    print(f"Oracle global drift: {orc100['mean_drift']:.4f}")
    print("\nPer-class drift (Oracle vs Frozen):")
    print(f"{'Class':>5} | {'Frozen':>8} | {'Oracle':>8}")
    for c in live:
        print(f"{c:>5} | {f0['pc_drift'][c]:>8.4f} | {orc100['pc_drift'][c]:>8.4f}")
        
    print("\n(If Oracle drift is 0.99 for some classes and 0.40 for others, the mean drift")
    print("is dominated by a few violently oscillating classes, explaining why it's so high.)")

    print("\n" + "=" * 78)
    print("2. PER-CLASS SUPERVISED CEILING: Are rare classes learnable?")
    print("=" * 78)
    sup = run_supervised_ceiling(model, loader, dev, src, live)
    
    print(f"{'Class':>5} | {'Frozen':>8} | {'Ceiling':>8} | {'Headroom':>8}")
    for c in live:
        f_iou = f0["pc_iou"][c]
        s_iou = sup["pc_iou"][c]
        print(f"{c:>5} | {f_iou:>8.2f} | {s_iou:>8.2f} | {s_iou - f_iou:>+8.2f}")
        
    print(f"\nLive mIoU -> Frozen: {f0['miou']:.2f} | Ceiling: {sup['miou']:.2f}")

    print("\n" + "=" * 78)
    print("3. THE PRECISION WALL: Where does the gain vanish?")
    print("=" * 78)
    
    fz_runs = [run_precision_oracle(model, loader, dev, src, live, 100, mode="frozen", seed=s)["miou"] for s in SEEDS]
    FROZEN_MEAN = np.mean(fz_runs)
    NOISE = np.std(fz_runs)
    
    print(f"Frozen mIoU: {FROZEN_MEAN:.2f} ± {NOISE:.2f}")
    print(f"{'Target P':>8} | {'Macro %':>7} | {'mIoU mean':>10} | {'std':>5} | {'Gain':>7} | {'Status':>8}")
    
    for p in PRECISIONS:
        res = [run_precision_oracle(model, loader, dev, src, live, p, mode="oracle", seed=s) for s in SEEDS]
        miou_vals = [r["miou"] for r in res]
        macro_vals = [r["macro_prec"] for r in res]
        
        m = np.mean(miou_vals)
        s = np.std(miou_vals)
        macro_mean = np.mean(macro_vals)
        gain = m - FROZEN_MEAN
        status = "PASS" if gain > 2 * NOISE else "FAIL"
        
        print(f"{p:>7}% | {macro_mean:>6.1f}% | {m:>10.2f} | {s:>5.2f} | {gain:>+7.2f} | {status:>8}")
        
    print("\nConclusion: If the gain fails (drops below 2*noise) at 95%, you need 98% precision.")
    print("If it survives to 90% or 85%, your target is achievable.")

if __name__ == "__main__":
    main()
