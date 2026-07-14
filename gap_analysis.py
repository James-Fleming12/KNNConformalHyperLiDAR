import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import yaml

import sys

class Logger(object):
    def __init__(self, filename="gap_analysis.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("gap_analysis.log")

from dataset.kitti.parser import Parser
from modules.HDC_utils import set_knn_model

KITTIC_DIR = "/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C"
DATA_DIR = "/mnt/alpha/jmfleming/KITTI"
CONFIG_ARCH = "config/arch/senet-2048p.yml"
CONFIG_LABELS = "config/labels/semantic-kitti-all.yaml"
PRETRAINED = "logs/kitti_pretrain/hdc_sub.pth"
NUM_CLASSES = 17

CONDITIONS = ["snow/light", "snow/heavy", "cross_sensor/heavy", "wet_ground/moderate"]
COVERAGES = [0.05, 0.10, 0.25, 0.50, 0.75]
LR = 0.01
N_ADAPT = 1000   # Increased from 300 to give rare classes time to move
N_EVAL = 500     # Increased from 100 for more robust mIoU
OUT = "gap_analysis.json"


def fast_hist(preds, labels, K):
    m = (labels >= 0) & (labels < K)
    return torch.bincount(K * labels[m].long() + preds[m].long(),
                          minlength=K ** 2).reshape(K, K)


def per_class_iou(hist):
    tp = torch.diag(hist).float()
    fp = hist.sum(0).float() - tp
    fn = hist.sum(1).float() - tp
    iou = tp / (tp + fp + fn + 1e-10)
    seen = (hist.sum(1) + hist.sum(0)) > 0
    return iou, seen


@torch.no_grad()
def encode_frame(model, x, y, device):
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
def calibrate(model, loader, device, coverage, mondrian, n=500):
    """Return threshold(s) on CLEAN SOURCE data.
    Increased to 500 frames and uses trainloader to get stable quantiles for rare classes.
    """
    scores = {c: [] for c in range(NUM_CLASSES)}
    allsc = []
    for i, b in enumerate(loader):
        if i >= n:
            break
        x = b[0].to(device); y = b[2].to(device).view(-1)
        if x.shape[1] == 0:
            continue
        out = encode_frame(model, x, y, device)
        if out is None:
            continue
        _, sims, preds, _ = out
        s = sims.gather(1, preds.unsqueeze(1)).squeeze(1)
        allsc.append(s.cpu())
        for c in preds.unique().tolist():
            scores[c].append(s[preds == c].cpu())

    if not mondrian:
        a = torch.cat(allsc).float()
        q = torch.quantile(a, 1.0 - coverage).item()
        return {c: q for c in range(NUM_CLASSES)}

    # Mondrian: calibrate WITHIN each class
    qs = {}
    glob = torch.quantile(torch.cat(allsc).float(), 1.0 - coverage).item()
    
    sizes = {c: sum(t.numel() for t in scores[c]) if scores[c] else 0 for c in range(NUM_CLASSES)}
    valid_qs = {}
    
    # Establish valid anchors
    for c in range(NUM_CLASSES):
        if sizes[c] >= 200:
            v = torch.cat(scores[c]).float()
            valid_qs[c] = torch.quantile(v, 1.0 - coverage).item()
            
    fallback_count = 0
    for c in range(NUM_CLASSES):
        if sizes[c] >= 200:
            qs[c] = valid_qs[c]
        else:
            fallback_count += 1
            if not valid_qs:
                qs[c] = glob
            else:
                # Borrow threshold from nearest valid class size to prevent starvation fallback
                nearest_c = min(valid_qs.keys(), key=lambda k: abs(sizes[k] - sizes[c]))
                qs[c] = valid_qs[nearest_c]
                
    if fallback_count > 0:
        print(f"      [Mondrian cov={coverage}] {fallback_count} classes fell back to nearest-size neighbor (<200 pts)")
    return qs


@torch.no_grad()
def run(model, loader, device, src_proto, qs, soft=False,
        n_adapt=N_ADAPT, n_eval=300, num_eval_windows=3):
    """Adapt with the given per-class thresholds, then evaluate across multiple windows."""
    model.classify.weight.data = src_proto.clone()
    fire_n = torch.zeros(NUM_CLASSES, device=device)
    seen_n = torch.zeros(NUM_CLASSES, device=device)
    prec_n = torch.zeros(NUM_CLASSES, device=device)
    gt_n_adapt = torch.zeros(NUM_CLASSES, device=device)

    qv = torch.tensor([qs[c] for c in range(NUM_CLASSES)], device=device)

    for i, b in enumerate(loader):
        if i >= n_adapt:
            break
        x = b[0].to(device); y = b[2].to(device).view(-1)
        if x.shape[1] == 0:
            continue
        out = encode_frame(model, x, y, device)
        if out is None:
            continue
        h, sims, preds, labels = out
        s = sims.gather(1, preds.unsqueeze(1)).squeeze(1)
        thr = qv[preds]
        admit = s >= thr

        for c in labels.unique().tolist():
            gt_n_adapt[c] += (labels == c).sum()

        for c in preds.unique().tolist():
            m = preds == c
            seen_n[c] += m.sum()
            a = m & admit
            fire_n[c] += a.sum()
            prec_n[c] += (a & (preds == labels)).sum()

        if admit.any():
            for c in preds[admit].unique().tolist():
                m = (preds == c) & admit
                # Noise limit: ignore updates from less than 10 points
                if m.sum() < 10:
                    continue
                if soft:
                    w = (s[m] - qs[c]).clamp_min(0).unsqueeze(1)
                    pull = (h[m] * w).sum(0) / w.sum().clamp_min(1e-8)
                else:
                    pull = h[m].mean(0)
                nw = model.classify.weight[c] + LR * pull
                model.classify.weight[c] = F.normalize(nw.unsqueeze(0), dim=1).squeeze(0)

    # evaluate in multiple disjoint windows to capture variance
    eval_hists = []
    current_hist = torch.zeros((NUM_CLASSES, NUM_CLASSES), device=device)
    total_eval_frames = n_eval * num_eval_windows
    frames_in_window = 0
    
    for i, b in enumerate(loader):
        if i < n_adapt:
            continue
        if i >= n_adapt + total_eval_frames:
            break
        x = b[0].to(device); y = b[2].to(device).view(-1)
        if x.shape[1] == 0:
            continue
        out = encode_frame(model, x, y, device)
        if out is None:
            continue
        _, _, preds, labels = out
        current_hist += fast_hist(preds, labels, NUM_CLASSES)
        frames_in_window += 1
        
        if frames_in_window >= n_eval:
            eval_hists.append(current_hist.cpu())
            current_hist = torch.zeros((NUM_CLASSES, NUM_CLASSES), device=device)
            frames_in_window = 0

    if frames_in_window > 0:
        eval_hists.append(current_hist.cpu())

    fr = (fire_n / seen_n.clamp_min(1)).cpu().numpy()
    pc = (prec_n / fire_n.clamp_min(1)).cpu().numpy()
    
    # Store all hists so main can compute mean/std across windows
    return {"fire_rate": fr, "precision": pc,
            "n_seen": seen_n.cpu().numpy(), "gt_n_adapt": gt_n_adapt.cpu().numpy(),
            "eval_hists": eval_hists}

def get_live_miou(run_dict, live_classes):
    """Computes mean mIoU and std across multiple evaluation windows."""
    if not live_classes:
        return 0.0, 0.0
    window_mious = []
    for hist in run_dict["eval_hists"]:
        iou, seen = per_class_iou(hist)
        miou = np.nanmean([iou[c].item()*100 for c in live_classes if seen[c]])
        window_mious.append(miou)
    return np.mean(window_mious), np.std(window_mious)

def get_live_per_class_iou(run_dict):
    """Computes the aggregate per-class IoU over all windows combined."""
    total_hist = sum(run_dict["eval_hists"])
    iou, seen = per_class_iou(total_hist)
    return (iou * 100).cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ARCH = yaml.safe_load(open(CONFIG_ARCH))
    DATA = yaml.safe_load(open(CONFIG_LABELS))

    model = set_knn_model(ARCH, "logs/kitti_pretrain", "rp", 0, 0, NUM_CLASSES, device)
    model.load_state_dict(torch.load(PRETRAINED, map_location=device), strict=False)
    model.to(device).eval()
    src = model.classify.weight.data.clone()

    p = Parser(root=DATA_DIR, train_sequences=DATA["split"]["train"],
               valid_sequences=DATA["split"]["valid"], test_sequences=None,
               labels=DATA["labels"], color_map=DATA.get("color_map", {}),
               learning_map=DATA["learning_map"],
               learning_map_inv=DATA["learning_map_inv"],
               sensor=ARCH["dataset"]["sensor"],
               max_points=ARCH["dataset"]["max_points"],
               batch_size=1, workers=0, gt=True, shuffle_train=False)
    src_loader = p.validloader

    results = {}
    for cond in CONDITIONS:
        corr, sev = cond.split("/")
        root = os.path.join(KITTIC_DIR, corr, sev)
        s08 = os.path.join(root, "sequences", "08")
        if not os.path.exists(s08):
            os.makedirs(os.path.dirname(s08), exist_ok=True)
            try:
                os.symlink("..", s08)
            except FileExistsError:
                pass
        try:
            pp = Parser(root=root, train_sequences=DATA["split"]["valid"],
                        valid_sequences=DATA["split"]["valid"], test_sequences=None,
                        labels=DATA["labels"], color_map=DATA.get("color_map", {}),
                        learning_map=DATA["learning_map"],
                        learning_map_inv=DATA["learning_map_inv"],
                        sensor=ARCH["dataset"]["sensor"],
                        max_points=ARCH["dataset"]["max_points"],
                        batch_size=1, workers=0, gt=True, shuffle_train=False)
            loader = pp.validloader
        except Exception as e:
            print(f"skip {cond}: {e}")
            continue

        print(f"\n{'='*78}\n{cond}\n{'='*78}")
        results[cond] = {}

        # 1. Establish live classes using frozen model
        print("Evaluating frozen baseline to determine live classes...")
        frozen = run(model, loader, device, src, {c: 9.9 for c in range(NUM_CLASSES)})
        frozen_pc_iou = get_live_per_class_iou(frozen)
        
        # A class is 'live' if the frozen model achieves at least 1.0 IoU on it.
        live_classes = [c for c in range(NUM_CLASSES) if frozen_pc_iou[c] >= 1.0]
        frozen_live_miou, frozen_live_std = get_live_miou(frozen, live_classes)
        print(f"Found {len(live_classes)} live classes: {live_classes}")
        print(f"Frozen LIVE mIoU: {frozen_live_miou:.2f} ± {frozen_live_std:.2f}")

        # ---- D3/D4: coverage operating point on LIVE CLASSES ------------------
        print(f"\nD3/D4: coverage sweep  (marginal = global q; mondrian = per-class q)")
        print(f"{'cov':>6} {'marg mIoU':>12} {'mond mIoU':>12} {'mond-marg':>10} "
              f"{'marg minfire':>13} {'mond minfire':>13}")
        for cov in COVERAGES:
            q_m = calibrate(model, src_loader, device, cov, mondrian=False)
            q_c = calibrate(model, src_loader, device, cov, mondrian=True)
            rm = run(model, loader, device, src, q_m)
            rc = run(model, loader, device, src, q_c)
            
            rm_miou, rm_std = get_live_miou(rm, live_classes)
            rc_miou, rc_std = get_live_miou(rc, live_classes)
            
            # minfire computed strictly over live classes
            seen_m = np.array([True if c in live_classes and rm["n_seen"][c] > 100 else False for c in range(NUM_CLASSES)])
            mn_m = rm["fire_rate"][seen_m].min() * 100 if seen_m.any() else float("nan")
            mn_c = rc["fire_rate"][seen_m].min() * 100 if seen_m.any() else float("nan")
            
            print(f"{cov:>6.2f} {rm_miou:>5.2f}±{rm_std:>4.2f} {rc_miou:>5.2f}±{rc_std:>4.2f} "
                  f"{rc_miou-rm_miou:>+10.2f} {mn_m:>13.1f} {mn_c:>13.1f}")
            results[cond][f"cov{cov}"] = {
                "marginal_miou_live": rm_miou, "marginal_std": rm_std,
                "mondrian_miou_live": rc_miou, "mondrian_std": rc_std,
            }

        # ---- D1/D2: per-class starvation and IoU delta at 50% coverage ---------
        q_m = calibrate(model, src_loader, device, 0.50, mondrian=False)
        q_c = calibrate(model, src_loader, device, 0.50, mondrian=True)
        rm = run(model, loader, device, src, q_m)
        rc = run(model, loader, device, src, q_c)
        
        rm_pc_iou = get_live_per_class_iou(rm)
        rc_pc_iou = get_live_per_class_iou(rc)

        print(f"\nD1/D2: PER-CLASS BREAKDOWN (Live Classes Only, 50% Coverage)")
        print(f"{'cls':>4} {'GT_pts':>9} {'Pred_pts':>9} {'fire_marg':>10} {'fire_mond':>10} "
              f"{'IoU_froz':>9} {'IoU_marg':>9} {'IoU_mond':>9} {'delta':>8}")
        
        # Sort live classes by true GT point count descending
        order = sorted(live_classes, key=lambda c: frozen['gt_n_adapt'][c], reverse=True)
        for c in order:
            gt_pts = int(frozen['gt_n_adapt'][c])
            pred_pts = int(rm['n_seen'][c])
            fm = rm['fire_rate'][c]*100
            fc = rc['fire_rate'][c]*100
            ifz = frozen_pc_iou[c]
            im = rm_pc_iou[c]
            ic = rc_pc_iou[c]
            delta = ic - im
            
            print(f"{c:>4} {gt_pts:>9} {pred_pts:>9} "
                  f"{fm:>9.1f}% {fc:>9.1f}% "
                  f"{ifz:>9.1f} {im:>9.1f} {ic:>9.1f} {delta:>+8.1f}")
                  
        rm_miou, rm_std = get_live_miou(rm, live_classes)
        rc_miou, rc_std = get_live_miou(rc, live_classes)
        print(f"\n  LIVE mIoU:  frozen={frozen_live_miou:.2f}±{frozen_live_std:.2f}  "
              f"marginal={rm_miou:.2f}±{rm_std:.2f}  mondrian={rc_miou:.2f}±{rc_std:.2f}  "
              f"(Delta: {rc_miou - rm_miou:+.2f})")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
