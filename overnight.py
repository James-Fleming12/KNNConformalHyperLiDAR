"""
overnight.py -- DOES THIS METHOD WORK AT ALL, AND IF SO, WHAT MAKES IT WORK?

Structured as a series of GATES. Each tier only runs if the previous one passes.
The point is to fail FAST and CHEAPLY if the method cannot work, rather than spend
another week optimizing a gate on a problem with no headroom.

--------------------------------------------------------------------------------
WHY THIS STRUCTURE
--------------------------------------------------------------------------------
Current state (snow/light, live classes only):
    frozen                45.23
    8 adaptation runs     41.09 .. 47.94   (only 3 of 8 beat frozen)
    marginal vs mondrian  +3.89, -1.00, +3.58, -5.74   <- no trend. This is NOISE.

We cannot currently distinguish "the method helps" from "the method does nothing,
noisily." So before comparing ANY gates, we must establish two numbers:

    NOISE FLOOR:  how much does mIoU vary from measurement alone?
    CEILING:      how much could adaptation POSSIBLY gain, with a PERFECT gate?

If (ceiling - frozen) < 2 * noise, there is no headroom, and NO gating scheme can
ever help. That is a pivot signal, not a tuning problem, and it is the single most
important thing to learn tonight.

--------------------------------------------------------------------------------
TIERS
--------------------------------------------------------------------------------
T0  MEASUREMENT VALIDITY + HEADROOM      <- the decision gate. Everything hinges here.
    T0.1  noise floor: frozen mIoU across disjoint eval windows -> std
    T0.2  ORACLE gate: update prototypes with GROUND-TRUTH labels (100% precision)
    T0.3  SUPERVISED ceiling: refit prototypes from scratch on target w/ GT labels
    => if the ceiling is within noise of frozen, STOP. The classifier is not the
       bottleneck; the features are. Pivot.

T1  DOES PSEUDO-LABEL ADAPTATION WORK?   <- only if T0 shows headroom
    LR x coverage grid, interleaved eval, multiple seeds, error bars
    => if no config beats frozen by > 2*noise, gating is not the problem. Pivot.

T2  WHAT MAKES IT WORK?                  <- only if T1 passes
    marginal vs mondrian; hard vs soft; anchor on/off; per-class LR normalization

T3  DOES IT GENERALIZE?                  <- only if T2 finds a winner
    across corruptions and severities

DIAGNOSTICS logged throughout: prototype drift, per-class firing, admitted precision,
and the mIoU TRAJECTORY (not just the endpoint).

--------------------------------------------------------------------------------
TWO CONFOUNDS THIS FIXES
--------------------------------------------------------------------------------
1. INTERLEAVED EVAL. Previously: adapt on frames 0-999, eval on 1000-1499. But that is
   a DIFFERENT PART OF THE DRIVE -- different scene, different class balance. So "mIoU
   after adaptation" partly measured "how hard is the second half of sequence 08."
   Here: adapt on EVEN frames, evaluate on ODD frames. Same scene distribution, no
   drift confound. (This is the same artifact that produced the phantom +0.06 mIoU
   gains earlier in this project.)

2. ERROR BARS. Every previous number was a single run. We now repeat each config over
   several frame orderings and report mean +/- std, so a "+3.9" can be judged against
   the noise it has to clear.

Usage:
    CUDA_VISIBLE_DEVICES=3 nohup uv run overnight.py > overnight.log 2>&1 &
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import yaml

class Logger(object):
    def __init__(self, filename="overnight.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true", help="Run a rapid 10-frame test to catch crash bugs")
args = parser.parse_args()

sys.stdout = Logger("overnight.log")
sys.stderr = sys.stdout  # Catch tracebacks in the log too

from dataset.kitti.parser import Parser
from modules.HDC_utils import set_knn_model

KITTIC_DIR = "/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C"
DATA_DIR = "/mnt/alpha/jmfleming/KITTI"
CONFIG_ARCH = "config/arch/senet-2048p.yml"
CONFIG_LABELS = "config/labels/semantic-kitti-all.yaml"
PRETRAINED = "logs/kitti_pretrain/hdc_sub.pth"
NUM_CLASSES = 17

PRIMARY = "snow/light"                    # the tier-0/1/2 workbench
GENERALIZE = ["snow/heavy", "wet_ground/moderate", "cross_sensor/heavy",
              "motion_blur/moderate", "beam_missing/moderate"]

SEEDS = [0, 1, 2]
LRS = [0.0001, 0.001, 0.01, 0.1]
COVERAGES = [0.05, 0.10, 0.25, 0.50]
N_FRAMES = 2000
CALIB_N = 200
CALIB_MIN_N = 200

if args.dry_run:
    print(">>> DRY RUN MODE ACTIVATED: Testing pipeline with 20 frames... <<<")
    SEEDS = [0]
    LRS = [0.01]
    COVERAGES = [0.50]
    N_FRAMES = 20
    CALIB_N = 5
    CALIB_MIN_N = 1
    GENERALIZE = ["snow/heavy"]

OUT = "overnight.json"


# ============================================================ metrics / helpers

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
def calibrate(model, loader, device, coverage, mondrian, n=None, min_n=None):
    """Thresholds from CLEAN SOURCE.

    min_n=200 (not 5): a 90th-percentile estimate from a handful of samples is
    essentially random, and it is exactly the RARE classes that hit that path -- so a
    bad estimate there kills the very classes Mondrian exists to protect.

    Classes below min_n fall back to the global q AND ARE REPORTED, because if half the
    live classes fall back, you are not actually running Mondrian.
    """
    n = n if n is not None else CALIB_N
    min_n = min_n if min_n is not None else CALIB_MIN_N
    
    per, allsc = {c: [] for c in range(NUM_CLASSES)}, []
    for i, b in enumerate(loader):
        if i >= n:
            break
        x, y = b[0].to(device), b[2].to(device).view(-1)
        if x.shape[1] == 0:
            continue
        out = encode_frame(model, x, y)
        if out is None:
            continue
        _, sims, preds, _ = out
        s = sims.gather(1, preds.unsqueeze(1)).squeeze(1)
        allsc.append(s.cpu())
        for c in preds.unique().tolist():
            per[c].append(s[preds == c].cpu())

    glob = torch.quantile(torch.cat(allsc).float(), 1.0 - coverage).item()
    if not mondrian:
        return {c: glob for c in range(NUM_CLASSES)}, 0

    qs, fallbacks = {}, 0
    for c in range(NUM_CLASSES):
        v = torch.cat(per[c]).float() if per[c] else torch.tensor([])
        if v.numel() >= min_n:
            qs[c] = torch.quantile(v, 1.0 - coverage).item()
        else:
            qs[c] = glob
            fallbacks += 1
    return qs, fallbacks


# ============================================================ the core run

@torch.no_grad()
def adapt_and_eval(model, loader, device, src, live, *,
                   mode="pseudo",          # pseudo | oracle | frozen
                   qs=None, lr=0.01, soft=False, anchor=0.0,
                   per_class_lr=False, seed=0, n_frames=N_FRAMES,
                   track=False):
    """Adapt on EVEN frames, evaluate on ODD frames.

    Interleaving is the key fix: adapt and eval now see the SAME scene distribution, so
    the result cannot be contaminated by "the second half of the drive is harder."

    mode="oracle": gate on GROUND TRUTH (admit iff the prediction is actually correct).
                   This is a PERFECT gate -- 100% precision. It is the CEILING for any
                   pseudo-label gating scheme. NOT deployable; it exists only to tell us
                   whether ANY gate could ever help.
    """
    torch.manual_seed(seed)
    model.classify.weight.data = src.clone()
    qv = (torch.tensor([qs[c] for c in range(NUM_CLASSES)], device=device)
          if qs else None)

    hist = torch.zeros((NUM_CLASSES, NUM_CLASSES), device=device)
    fire = torch.zeros(NUM_CLASSES, device=device)
    seen = torch.zeros(NUM_CLASSES, device=device)
    hit = torch.zeros(NUM_CLASSES, device=device)
    traj = []

    for i, b in enumerate(loader):
        if i >= n_frames:
            break
        x, y = b[0].to(device), b[2].to(device).view(-1)
        if x.shape[1] == 0:
            continue
        out = encode_frame(model, x, y)
        if out is None:
            continue
        h, sims, preds, labels = out

        # ---- ODD frames: EVALUATE ONLY (never adapted on) ----
        if i % 2 == 1:
            hist += fast_hist(preds, labels, NUM_CLASSES)
            if track and i % 50 == 1:
                traj.append({
                    "frame": i,
                    "miou": live_miou(per_class_iou(hist).cpu().numpy(), live),
                    "drift": float(F.cosine_similarity(
                        F.normalize(model.classify.weight, dim=1),
                        F.normalize(src, dim=1), dim=1).mean()),
                })
            continue

        # ---- EVEN frames: ADAPT ----
        if mode == "frozen":
            continue

        if mode == "oracle":
            admit = preds == labels          # PERFECT gate. Ceiling only.
        else:
            s = sims.gather(1, preds.unsqueeze(1)).squeeze(1)
            admit = s >= qv[preds]

        for c in preds.unique().tolist():
            m = preds == c
            seen[c] += m.sum()
            a = m & admit
            fire[c] += a.sum()
            hit[c] += (a & (preds == labels)).sum()

        if not admit.any():
            continue
        for c in preds[admit].unique().tolist():
            m = (preds == c) & admit
            n_adm = int(m.sum())
            if n_adm < 10:
                # a 10-point mean in 10k dims is noise; updating on it injects
                # exactly the noise we are trying to keep out of rare classes
                continue
            if soft and qs is not None:
                w = (sims.gather(1, preds.unsqueeze(1)).squeeze(1)[m]
                     - qs[c]).clamp_min(0).unsqueeze(1)
                pull = (h[m] * w).sum(0) / w.sum().clamp_min(1e-8)
            else:
                pull = h[m].mean(0)

            eff_lr = lr
            if per_class_lr:
                # rare classes contribute few, noisy points -> damp their step so we do
                # not inject noise into the classes we are trying to protect
                eff_lr = lr * float(np.sqrt(min(n_adm, 1000) / 1000.0))

            w_new = model.classify.weight[c] + eff_lr * pull
            if anchor > 0:
                w_new = w_new + anchor * eff_lr * (src[c] - model.classify.weight[c])
            model.classify.weight[c] = F.normalize(
                w_new.unsqueeze(0), dim=1).squeeze(0)

    iou = per_class_iou(hist).cpu().numpy()
    return {
        "miou": live_miou(iou, live),
        "per_class_iou": (iou * 100).tolist(),
        "fire": (fire / seen.clamp_min(1)).cpu().numpy().tolist(),
        "precision": (hit / fire.clamp_min(1)).cpu().numpy().tolist(),
        "drift": float(F.cosine_similarity(
            F.normalize(model.classify.weight, dim=1),
            F.normalize(src, dim=1), dim=1).mean()),
        "traj": traj,
    }


# ============================================================ supervised ceiling

@torch.no_grad()
def supervised_ceiling(model, loader, device, src, live, n_frames=N_FRAMES):
    """Refit prototypes FROM SCRATCH on target data using GT labels (even frames only),
    then evaluate on odd frames.

    This is the ABSOLUTE ceiling for a prototype classifier on this domain. If this is
    barely above frozen, then the prototype layer is ALREADY near-optimal for these
    features, there is no headroom, and no gating scheme -- however clever -- can matter.
    That is the pivot signal.
    """
    acc = torch.zeros_like(src)
    cnt = torch.zeros(NUM_CLASSES, device=device)
    for i, b in enumerate(loader):
        if i >= n_frames:
            break
        if i % 2 == 1:
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
        if i >= n_frames:
            break
        if i % 2 == 0:
            continue
        x, y = b[0].to(device), b[2].to(device).view(-1)
        if x.shape[1] == 0:
            continue
        out = encode_frame(model, x, y)
        if out is None:
            continue
        _, _, preds, labels = out
        hist += fast_hist(preds, labels, NUM_CLASSES)
    model.classify.weight.data = src.clone()
    return live_miou(per_class_iou(hist).cpu().numpy(), live)


# ============================================================ main

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

    p = Parser(root=DATA_DIR, train_sequences=DATA["split"]["train"],
               valid_sequences=DATA["split"]["valid"], test_sequences=None,
               labels=DATA["labels"], color_map=DATA.get("color_map", {}),
               learning_map=DATA["learning_map"],
               learning_map_inv=DATA["learning_map_inv"],
               sensor=ARCH["dataset"]["sensor"],
               max_points=ARCH["dataset"]["max_points"],
               batch_size=1, workers=0, gt=True, shuffle_train=False)
    src_loader = p.validloader

    R = {}
    loader = get_loader(PRIMARY, ARCH, DATA)

    # ---- live classes -------------------------------------------------------
    fz = adapt_and_eval(model, loader, dev, src, list(range(NUM_CLASSES)),
                        mode="frozen")
    live = [c for c in range(NUM_CLASSES) if fz["per_class_iou"][c] >= 1.0]
    print(f"live classes ({len(live)}): {live}")

    # =========================================================================
    print("\n" + "=" * 78)
    print("T0 -- MEASUREMENT VALIDITY AND HEADROOM   (the decision gate)")
    print("=" * 78)

    # T0.1 noise floor
    fz_runs = [adapt_and_eval(model, loader, dev, src, live, mode="frozen",
                              seed=s)["miou"] for s in SEEDS]
    # frozen is deterministic, so instead measure window-to-window variation
    win = []
    window_size = 400 if not args.dry_run else 5
    for w in range(4):
        hist = torch.zeros((NUM_CLASSES, NUM_CLASSES), device=dev)
        for i, b in enumerate(loader):
            if i >= (w + 1) * window_size:
                break
            if i < w * window_size:
                continue
            x, y = b[0].to(dev), b[2].to(dev).view(-1)
            if x.shape[1] == 0:
                continue
            out = encode_frame(model, x, y)
            if out is None:
                continue
            _, _, preds, labels = out
            hist += fast_hist(preds, labels, NUM_CLASSES)
        win.append(live_miou(per_class_iou(hist).cpu().numpy(), live))
    NOISE = float(np.std(win))
    FROZEN = float(np.mean(fz_runs))
    print(f"T0.1  frozen mIoU (interleaved)  = {FROZEN:.2f}")
    print(f"      frozen across 4 windows    = {[f'{v:.1f}' for v in win]}")
    print(f"      NOISE FLOOR (std)          = {NOISE:.2f}")
    print(f"      => any claimed gain must exceed ~{2*NOISE:.2f} to be real")

    # T0.2 oracle gate
    orc = {}
    for lr in LRS:
        r = adapt_and_eval(model, loader, dev, src, live, mode="oracle", lr=lr)
        orc[lr] = r["miou"]
        print(f"T0.2  ORACLE gate  lr={lr:<7} mIoU={r['miou']:.2f}  "
              f"drift={r['drift']:.3f}")
    ORACLE = max(orc.values())

    # T0.3 supervised ceiling
    SUP = supervised_ceiling(model, loader, dev, src, live)
    print(f"T0.3  SUPERVISED ceiling (refit prototypes on target w/ GT) = {SUP:.2f}")

    print(f"\n  frozen   {FROZEN:.2f}")
    print(f"  oracle   {ORACLE:.2f}   (headroom {ORACLE-FROZEN:+.2f})")
    print(f"  ceiling  {SUP:.2f}   (headroom {SUP-FROZEN:+.2f})")
    print(f"  noise    {NOISE:.2f}")

    R["T0"] = {"frozen": FROZEN, "noise": NOISE, "oracle": orc,
               "supervised": SUP, "windows": win, "live": live}

    HEADROOM = max(ORACLE, SUP) - FROZEN
    if HEADROOM < 2 * NOISE:
        print(f"""
################################################################################
STOP. NO HEADROOM.

Even a PERFECT gate (ground-truth labels) and a fully SUPERVISED refit cannot beat
the frozen model by more than the measurement noise ({HEADROOM:.2f} vs {2*NOISE:.2f}).

This means the prototype layer is ALREADY near-optimal for these features. The
bottleneck is NOT pseudo-label selection -- it is the representation. No gating
scheme, conformal or otherwise, can help, and every gate comparison run so far was
measuring noise.

PIVOT. Options:
  - adapt the FEATURE EXTRACTOR, not the classifier (but that needs backprop)
  - change the benchmark: find a shift where prototypes DO have headroom
    (cross_sensor is the obvious candidate -- frozen mIoU there is far lower)
  - reframe the paper around the negative result + the n<<d explanation
################################################################################
""")
        json.dump(R, open(OUT, "w"), indent=2, default=float)
        return

    print(f"\n  => headroom {HEADROOM:.2f} > 2*noise {2*NOISE:.2f}. Proceed to T1.")

    # =========================================================================
    print("\n" + "=" * 78)
    print("T1 -- DOES PSEUDO-LABEL ADAPTATION WORK?   (lr x coverage, with error bars)")
    print("=" * 78)
    print(f"{'lr':>8} {'cov':>6} {'mIoU mean':>10} {'std':>6} {'vs frozen':>10} "
          f"{'fire':>7} {'prec':>7} {'drift':>7}")

    T1, best = {}, (None, -1e9)
    for lr in LRS:
        for cov in COVERAGES:
            qs, _ = calibrate(model, src_loader, dev, cov, mondrian=False)
            runs = [adapt_and_eval(model, loader, dev, src, live, qs=qs, lr=lr,
                                   seed=s) for s in SEEDS]
            m = float(np.mean([r["miou"] for r in runs]))
            sd = float(np.std([r["miou"] for r in runs]))
            fr = float(np.mean([np.mean([r["fire"][c] for c in live]) for r in runs]))
            pr = float(np.mean([np.mean([r["precision"][c] for c in live])
                                for r in runs]))
            dr = float(np.mean([r["drift"] for r in runs]))
            flag = " *" if m - FROZEN > 2 * NOISE else ""
            print(f"{lr:>8} {cov:>6.2f} {m:>10.2f} {sd:>6.2f} {m-FROZEN:>+10.2f} "
                  f"{fr*100:>6.1f}% {pr*100:>6.1f}% {dr:>7.3f}{flag}")
            T1[f"lr{lr}_cov{cov}"] = {"miou": m, "std": sd, "fire": fr,
                                      "prec": pr, "drift": dr}
            if m > best[1]:
                best = ((lr, cov), m)
    R["T1"] = T1

    if best[1] - FROZEN < 2 * NOISE:
        print(f"""
################################################################################
STOP. PSEUDO-LABEL GATING DOES NOT WORK HERE.

There IS headroom (oracle/supervised beat frozen), but NO pseudo-label configuration
gets any of it. Best = {best[1]:.2f} vs frozen {FROZEN:.2f}, noise {NOISE:.2f}.

So the gap is not the CALIBRATION of the gate -- it is that pseudo-labels are simply
not good enough at any operating point. Compare the oracle ({ORACLE:.2f}) to the best
pseudo-label run ({best[1]:.2f}): that gap is the cost of not knowing the labels, and
it is apparently unbridgeable by thresholding alone.

PIVOT. Options:
  - improve PSEUDO-LABEL QUALITY, not selection (e.g. temporal/spatial consistency,
    multi-view agreement -- things that make the label better, not the filter tighter)
  - accept the negative result and write it up with the n<<d explanation
################################################################################
""")
        json.dump(R, open(OUT, "w"), indent=2, default=float)
        return

    (BLR, BCOV), BM = best
    print(f"\n  => best: lr={BLR} cov={BCOV} -> {BM:.2f} "
          f"({BM-FROZEN:+.2f} vs frozen). Proceed to T2.")

    # =========================================================================
    print("\n" + "=" * 78)
    print("T2 -- WHAT MAKES IT WORK?   (at the best lr/cov from T1)")
    print("=" * 78)
    q_marg, _ = calibrate(model, src_loader, dev, BCOV, mondrian=False)
    q_mond, nfb = calibrate(model, src_loader, dev, BCOV, mondrian=True)
    print(f"  (mondrian fell back to global q for {nfb} classes -- if that is most of "
          f"the live classes, you are not really running Mondrian)")

    variants = {
        "marginal":            dict(qs=q_marg),
        "mondrian":            dict(qs=q_mond),
        "mondrian+soft":       dict(qs=q_mond, soft=True),
        "mondrian+anchor":     dict(qs=q_mond, anchor=0.1),
        "mondrian+perclasslr": dict(qs=q_mond, per_class_lr=True),
        "oracle(ceiling)":     dict(mode="oracle"),
    }
    print(f"{'variant':>22} {'mIoU':>8} {'std':>6} {'vs frozen':>10} {'vs marg':>9}")
    T2 = {}
    base = None
    for name, kw in variants.items():
        runs = [adapt_and_eval(model, loader, dev, src, live, lr=BLR, seed=s, **kw)
                for s in SEEDS]
        m = float(np.mean([r["miou"] for r in runs]))
        sd = float(np.std([r["miou"] for r in runs]))
        if name == "marginal":
            base = m
        print(f"{name:>22} {m:>8.2f} {sd:>6.2f} {m-FROZEN:>+10.2f} "
              f"{(m-base) if base else 0:>+9.2f}")
        T2[name] = {"miou": m, "std": sd}
    R["T2"] = T2

    # =========================================================================
    print("\n" + "=" * 78)
    print("T3 -- GENERALIZATION   (best config across corruptions)")
    print("=" * 78)
    winner = max((k for k in T2 if k != "oracle(ceiling)"), key=lambda k: T2[k]["miou"])
    print(f"  carrying forward: {winner} @ lr={BLR} cov={BCOV}\n")
    print(f"{'condition':>26} {'frozen':>8} {'adapted':>8} {'delta':>8} {'oracle':>8}")
    T3 = {}
    for cond in [PRIMARY] + GENERALIZE:
        try:
            ld = get_loader(cond, ARCH, DATA)
        except Exception as e:
            print(f"{cond:>26}  skip ({e})")
            continue
        f0 = adapt_and_eval(model, ld, dev, src, list(range(NUM_CLASSES)),
                            mode="frozen")
        lv = [c for c in range(NUM_CLASSES) if f0["per_class_iou"][c] >= 1.0]
        fzm = adapt_and_eval(model, ld, dev, src, lv, mode="frozen")["miou"]
        kw = variants[winner]
        ad = float(np.mean([adapt_and_eval(model, ld, dev, src, lv, lr=BLR, seed=s,
                                           **kw)["miou"] for s in SEEDS]))
        orm = adapt_and_eval(model, ld, dev, src, lv, mode="oracle", lr=BLR)["miou"]
        print(f"{cond:>26} {fzm:>8.2f} {ad:>8.2f} {ad-fzm:>+8.2f} {orm:>8.2f}")
        T3[cond] = {"frozen": fzm, "adapted": ad, "oracle": orm, "live": lv}
    R["T3"] = T3

    json.dump(R, open(OUT, "w"), indent=2, default=float)
    print(f"\nsaved -> {OUT}")

    print("""
================================================================================
READING THE RESULT
================================================================================
The whole run hinges on T0. Everything else is conditional on it.

  ORACLE ~= FROZEN         -> no gate can ever help. The classifier is already
                              optimal for these features. PIVOT.
  ORACLE >> FROZEN,
    but best pseudo ~= FROZEN -> the problem is pseudo-label QUALITY, not selection.
                              Tightening the filter cannot fix labels that are wrong.
                              Work on making labels better (consistency, multi-view),
                              not on filtering them harder.
  ORACLE >> best pseudo
    >> FROZEN              -> gating works AND there is room left. This is the good
                              case: report the gap to oracle as the remaining headroom,
                              and T2 tells you which mechanism closes it.

The oracle-to-pseudo gap is the single most informative number in the whole table. It
is exactly "what it costs not to know the labels", and it bounds what ANY gating paper
can possibly contribute.
""")


if __name__ == "__main__":
    main()
