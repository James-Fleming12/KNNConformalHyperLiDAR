import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import yaml

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
N_ADAPT = 300
N_EVAL = 100
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
def calibrate(model, loader, device, coverage, mondrian, n=50):
    """Return threshold(s) on CLEAN SOURCE data.

    mondrian=False -> one global threshold  (marginal coverage; what everyone does)
    mondrian=True  -> one threshold PER CLASS (class-conditional coverage; the proposal)
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

    # Mondrian: calibrate WITHIN each class, so every class admits its own top X%
    qs = {}
    glob = torch.quantile(torch.cat(allsc).float(), 1.0 - coverage).item()
    for c in range(NUM_CLASSES):
        if scores[c]:
            v = torch.cat(scores[c]).float()
            qs[c] = torch.quantile(v, 1.0 - coverage).item() if v.numel() > 20 else glob
        else:
            qs[c] = glob
    return qs


@torch.no_grad()
def run(model, loader, device, src_proto, qs, soft=False,
        n_adapt=N_ADAPT, n_eval=N_EVAL):
    """Adapt with the given per-class thresholds, then evaluate. Tracks per-class firing."""
    model.classify.weight.data = src_proto.clone()
    fire_n = torch.zeros(NUM_CLASSES, device=device)
    seen_n = torch.zeros(NUM_CLASSES, device=device)
    prec_n = torch.zeros(NUM_CLASSES, device=device)

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

        for c in preds.unique().tolist():
            m = preds == c
            seen_n[c] += m.sum()
            a = m & admit
            fire_n[c] += a.sum()
            prec_n[c] += (a & (preds == labels)).sum()

        if admit.any():
            for c in preds[admit].unique().tolist():
                m = (preds == c) & admit
                if soft:
                    # SOFT: weight by how far above threshold (HyperDUM-style weighting,
                    # and the project's own "purification beats filtration" lesson)
                    w = (s[m] - qs[c]).clamp_min(0).unsqueeze(1)
                    pull = (h[m] * w).sum(0) / w.sum().clamp_min(1e-8)
                else:
                    pull = h[m].mean(0)
                nw = model.classify.weight[c] + LR * pull
                model.classify.weight[c] = F.normalize(nw.unsqueeze(0), dim=1).squeeze(0)

    # eval
    hist = torch.zeros((NUM_CLASSES, NUM_CLASSES), device=device)
    for i, b in enumerate(loader):
        if i < n_adapt:
            continue
        if i >= n_adapt + n_eval:
            break
        x = b[0].to(device); y = b[2].to(device).view(-1)
        if x.shape[1] == 0:
            continue
        out = encode_frame(model, x, y, device)
        if out is None:
            continue
        _, _, preds, labels = out
        hist += fast_hist(preds, labels, NUM_CLASSES)

    iou, seen = per_class_iou(hist)
    miou = iou[seen].mean().item() * 100
    fr = (fire_n / seen_n.clamp_min(1)).cpu().numpy()
    pc = (prec_n / fire_n.clamp_min(1)).cpu().numpy()
    return {"miou": miou, "per_class_iou": (iou * 100).cpu().numpy(),
            "seen": seen.cpu().numpy(), "fire_rate": fr, "precision": pc,
            "n_seen": seen_n.cpu().numpy()}


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

        # ---- D3: coverage operating point, marginal vs mondrian ---------------
        print(f"\nD3/D4: coverage sweep  (marginal = one global q; "
              f"mondrian = one q PER CLASS)")
        print(f"{'cov':>6} {'marg mIoU':>10} {'mond mIoU':>10} {'mond-marg':>10} "
              f"{'marg minfire':>13} {'mond minfire':>13}")
        for cov in COVERAGES:
            q_m = calibrate(model, src_loader, device, cov, mondrian=False)
            q_c = calibrate(model, src_loader, device, cov, mondrian=True)
            rm = run(model, loader, device, src, q_m)
            rc = run(model, loader, device, src, q_c)
            # the KEY number: the LEAST-fired class. If marginal starves some class to
            # ~0 while mondrian keeps it near `cov`, that is the mechanism.
            seen_m = rm["n_seen"] > 100
            mn_m = rm["fire_rate"][seen_m].min() * 100 if seen_m.any() else float("nan")
            mn_c = rc["fire_rate"][seen_m].min() * 100 if seen_m.any() else float("nan")
            print(f"{cov:>6.2f} {rm['miou']:>10.2f} {rc['miou']:>10.2f} "
                  f"{rc['miou']-rm['miou']:>+10.2f} {mn_m:>13.1f} {mn_c:>13.1f}")
            results[cond][f"cov{cov}"] = {
                "marginal_miou": rm["miou"], "mondrian_miou": rc["miou"],
                "marginal_fire": rm["fire_rate"].tolist(),
                "mondrian_fire": rc["fire_rate"].tolist(),
            }

        # ---- D1/D2: per-class starvation at the standard 50% operating point ---
        q_m = calibrate(model, src_loader, device, 0.50, mondrian=False)
        q_c = calibrate(model, src_loader, device, 0.50, mondrian=True)
        rm = run(model, loader, device, src, q_m)
        rc = run(model, loader, device, src, q_c)
        frozen = run(model, loader, device, src, {c: 9.9 for c in range(NUM_CLASSES)})

        print(f"\nD1/D2: per-class, at 50% marginal coverage")
        print(f"{'cls':>4} {'n_pts':>9} {'fire_marg':>10} {'fire_mond':>10} "
              f"{'IoU_frozen':>11} {'IoU_marg':>9} {'IoU_mond':>9}")
        order = np.argsort(-rm["n_seen"])
        for c in order:
            if rm["n_seen"][c] < 100:
                continue
            print(f"{c:>4} {int(rm['n_seen'][c]):>9} "
                  f"{rm['fire_rate'][c]*100:>9.1f}% {rc['fire_rate'][c]*100:>9.1f}% "
                  f"{frozen['per_class_iou'][c]:>11.1f} "
                  f"{rm['per_class_iou'][c]:>9.1f} {rc['per_class_iou'][c]:>9.1f}")
        print(f"\n  mIoU  frozen={frozen['miou']:.2f}  "
              f"marginal={rm['miou']:.2f}  mondrian={rc['miou']:.2f}")

        # ---- D5: soft weighting vs hard gate ----------------------------------
        rs = run(model, loader, device, src, q_c, soft=True)
        print(f"\nD5: hard gate={rc['miou']:.2f}   soft weight={rs['miou']:.2f}")
        results[cond]["soft_miou"] = rs["miou"]

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nsaved -> {OUT}")

    print("""
================================================================================
WHAT EACH OUTCOME MEANS
================================================================================
D1 (per-class firing under MARGINAL coverage)
   If rare classes fire at ~0% while common classes fire at 60-80%, the gate is
   allocating adaptation signal by CLASS FREQUENCY -- which is exactly backwards for
   mIoU. That is the gap, and it is structural, not a tuning issue.

D4 (mondrian - marginal)
   If mondrian > marginal, class-conditional conformal calibration is the contribution.
   It is novel for TTA, conformal-native (Vovk's Mondrian CP), and it targets the
   metric you actually report. The `minfire` columns are the mechanism: marginal should
   starve some class to near 0 while mondrian holds every class near `cov`.

D3 (coverage sweep)
   NOBODY HAS SWEPT THIS. Precision at 10% coverage was 99.2%. If mIoU peaks at 10-25%
   rather than 50%, then every adaptive-threshold idea so far (ACI, set-size) was
   loosening a gate that should have been TIGHTENED. That alone reframes the story.

D5 (soft vs hard)
   HyperDUM weights rather than gates, and this project's own early finding was
   "purification beats filtration". If soft > hard, the gate should be a WEIGHT, and
   the conformal p-value is the natural weight.

IF MONDRIAN WINS, THE PAPER IS:
   "Marginal conformal calibration is the wrong objective for segmentation TTA, because
   coverage is marginal while the metric is class-balanced. We give class-conditional
   conformal pseudo-label gating, which equalizes adaptation signal across classes and
   directly optimizes the reported metric. We further show that set-valued CP (as in
   ConformalHDC) is vacuous in HDC, because well-separated prototypes make prediction
   sets degenerate (|C| in {0,1})."
""")


if __name__ == "__main__":
    main()
