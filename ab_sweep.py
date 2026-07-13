import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset.kitti.parser import Parser
from modules.HDC_utils import set_efficient_knn_model

DATA_DIR = "/mnt/alpha/jmfleming/KITTI"
KITTIC_DIR = "/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C"
CONFIG_ARCH = "config/arch/senet-2048p.yml"
CONFIG_LABELS = "config/labels/semantic-kitti-all.yaml"
PRETRAINED = "logs/kitti_pretrain/hdc_sub.pth"
NUM_CLASSES = 17

CORRUPTION = "snow"
SEVERITY = 3
CAP = 3000

@torch.no_grad()
def collect_source(model, loader, device, cap):
    buckets = {c: [] for c in range(model.num_classes)}
    counts = {c: 0 for c in range(model.num_classes)}
    for batch in loader:
        x = batch[0].to(device); y = batch[2].to(device).view(-1)
        if x.shape[1] == 0: continue
        enc, idx, _ = model.encode(x)
        lab = y[idx] if idx is not None else y
        v = (lab >= 0) & (lab < model.num_classes)
        if not v.any(): continue
        enc, lab = enc[v], lab[v]
        for c in lab.unique().tolist():
            if counts[c] >= cap: continue
            hc = enc[lab == c]
            take = min(hc.shape[0], cap - counts[c])
            buckets[c].append(hc[:take].cpu())
            counts[c] += take
        if all(counts[c] >= cap for c in range(1, model.num_classes)):
            break
    return {c: torch.cat(t) for c, t in buckets.items() if t}

@torch.no_grad()
def collect_target(model, loader, device, cap):
    H, P, C = [], [], []
    counts = {c: 0 for c in range(model.num_classes)}
    protos = F.normalize(model.classify.weight)
    for batch in loader:
        x = batch[0].to(device); y = batch[2].to(device).view(-1)
        if x.shape[1] == 0: continue
        enc, idx, _ = model.encode(x)
        lab = y[idx] if idx is not None else y
        v = (lab >= 0) & (lab < model.num_classes)
        if not v.any(): continue
        enc, lab = enc[v], lab[v]
        sims = F.normalize(enc).to(protos.dtype) @ protos.T
        preds = sims.argmax(dim=1)
        for c in lab.unique().tolist():
            if counts[c] >= cap: continue
            m = lab == c
            take = min(int(m.sum()), cap - counts[c])
            H.append(enc[m][:take].cpu())
            P.append(preds[m][:take].cpu())
            C.append((preds[m][:take] == c).cpu())
            counts[c] += take
        if all(counts[c] >= cap for c in range(1, model.num_classes)):
            break
    return torch.cat(H), torch.cat(P), torch.cat(C)

def per_class_auroc(scores, correct, preds, min_n=50):
    out = []
    for c in np.unique(preds):
        m = preds == c
        if m.sum() < min_n: continue
        cc = correct[m]
        if len(np.unique(cc)) < 2: continue
        out.append(roc_auc_score(cc, scores[m]))
    return float(np.mean(out)) if out else float("nan")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ARCH = yaml.safe_load(open(CONFIG_ARCH))
    DATA = yaml.safe_load(open(CONFIG_LABELS))
    
    # 1. Base Model setup to extract data
    print("Initializing base model and extracting data...")
    base_model = set_efficient_knn_model(ARCH, "logs/kitti_pretrain", 'rp', 0, 0, NUM_CLASSES, device)
    base_model.load_state_dict(torch.load(PRETRAINED, map_location=device), strict=False)
    base_model.to(device).eval()
    
    parser = Parser(
        root=DATA_DIR, train_sequences=DATA["split"]["train"],
        valid_sequences=DATA["split"]["valid"], test_sequences=None,
        labels=DATA["labels"], color_map=DATA.get("color_map", {}),
        learning_map=DATA["learning_map"], learning_map_inv=DATA["learning_map_inv"],
        sensor=ARCH["dataset"]["sensor"], max_points=ARCH["dataset"]["max_points"],
        batch_size=1, workers=4, gt=True, shuffle_train=False
    )
    src_dict = collect_source(base_model, parser.trainloader, device, CAP)
    print(f"Collected Source Data: {[v.shape[0] for k, v in src_dict.items()]}")

    tgt_dir = os.path.join(KITTIC_DIR, CORRUPTION, "moderate" if SEVERITY == 2 else ("heavy" if SEVERITY == 3 else "extreme"))
    if not os.path.exists(os.path.join(tgt_dir, "sequences")):
        os.symlink("..", os.path.join(tgt_dir, "sequences"))
        
    tgt_parser = Parser(
        root=tgt_dir, train_sequences=DATA["split"]["valid"],
        valid_sequences=DATA["split"]["valid"], test_sequences=None,
        labels=DATA["labels"], color_map=DATA.get("color_map", {}),
        learning_map=DATA["learning_map"], learning_map_inv=DATA["learning_map_inv"],
        sensor=ARCH["dataset"]["sensor"], max_points=ARCH["dataset"]["max_points"],
        batch_size=1, workers=4, gt=True, shuffle_train=False
    )
    H_tgt, P_tgt, C_tgt = collect_target(base_model, tgt_parser.validloader, device, CAP)
    print(f"Collected Target Data: H={H_tgt.shape}, Acc={C_tgt.float().mean().item():.3f}")

    configs = [
        {"name": "Original (3k)", "use_coreset": False, "use_pca": False, "use_binary": False},
        {"name": "1. Coreset (M=64)", "use_coreset": True, "coreset_size": 64, "use_pca": False, "use_binary": False},
        {"name": "2. PCA (D=128)", "use_coreset": False, "use_pca": True, "pca_dims": 128, "use_binary": False},
        {"name": "3. Binary", "use_coreset": False, "use_pca": False, "use_binary": True},
        {"name": "Stacked (64x128b)", "use_coreset": True, "coreset_size": 64, "use_pca": True, "pca_dims": 128, "use_binary": True}
    ]

    print("\n" + "="*50)
    print("Ablation Sweep: Efficiency vs AUROC")
    print("="*50)
    
    # Random seed to keep RP fixed if PCA is used
    torch.manual_seed(42)

    for cfg in configs:
        name = cfg.pop("name")
        model = set_efficient_knn_model(ARCH, "logs/kitti_pretrain", 'rp', 0, 0, NUM_CLASSES, device, **cfg)
        model.load_state_dict(torch.load(PRETRAINED, map_location=device), strict=False)
        model.to(device).eval()
        
        # Manually invoke update_bank on src_dict so it goes through transformations
        for c, encs in src_dict.items():
            model.update_bank(encs.to(device), torch.full((encs.shape[0],), c, device=device))
            
        scores = []
        # Chunk to avoid OOM
        chunk = 1024
        for i in range(0, H_tgt.shape[0], chunk):
            h_chunk = H_tgt[i:i+chunk].to(device)
            p_chunk = P_tgt[i:i+chunk].to(device)
            conf = model.get_confidence(h_chunk, p_chunk)
            scores.append(conf.cpu())
            
        scores = torch.cat(scores).numpy()
        correct = C_tgt.numpy()
        preds = P_tgt.numpy()
        
        auroc = per_class_auroc(scores, correct, preds)
        
        # Calculate theoretical bytes per element in bank
        num_items = 64 if cfg.get("use_coreset") else CAP
        dim = 128 if cfg.get("use_pca") else 2048
        bytes_per = 0.125 if cfg.get("use_binary") else 2
        bank_kb = (num_items * dim * bytes_per * NUM_CLASSES) / 1024
        
        print(f"{name:<20} | AUROC: {auroc:.4f} | Bank Size: {bank_kb:,.1f} KB")

if __name__ == "__main__":
    main()
