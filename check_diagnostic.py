import sys
import torch
import torch.nn.functional as F
import yaml
from dataset.kitti.parser import Parser
from modules.HDC_utils import set_knn_model

DATA_DIR = "/mnt/alpha/jmfleming/KITTI"
KITTIC_DIR = "/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C"
CONFIG_ARCH = "config/arch/senet-2048p.yml"
CONFIG_LABELS = "config/labels/semantic-kitti-all.yaml"
PRETRAINED = "logs/kitti_pretrain/hdc_sub.pth"
BANK_PATH = "logs/knn_bank.pt"
NUM_CLASSES = 17

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ARCH = yaml.safe_load(open(CONFIG_ARCH))
    DATA = yaml.safe_load(open(CONFIG_LABELS))
    
    print("Loading Model...")
    model = set_knn_model(ARCH, "logs/kitti_pretrain", 'rp', 0, 0, NUM_CLASSES, device)
    model.load_state_dict(torch.load(PRETRAINED, map_location=device), strict=False)
    
    print(f"Loading Bank from {BANK_PATH}...")
    model.bank = torch.load(BANK_PATH, map_location=device)
    
    print("Calibrating Thresholds (Coverage 50%)...")
    model.calibrate_thresholds(coverage=0.50)
    
    tgt_dir = f"{KITTIC_DIR}/snow/heavy"
    print(f"Loading one frame from {tgt_dir}...")
    
    tgt_parser = Parser(
        root=tgt_dir, train_sequences=DATA["split"]["valid"],
        valid_sequences=DATA["split"]["valid"], test_sequences=None,
        labels=DATA["labels"], color_map=DATA.get("color_map", {}),
        learning_map=DATA["learning_map"], learning_map_inv=DATA["learning_map_inv"],
        sensor=ARCH["dataset"]["sensor"], max_points=ARCH["dataset"]["max_points"],
        batch_size=1, workers=1, gt=True, shuffle_train=False
    )
    
    loader = tgt_parser.validloader
    batch = next(iter(loader))
    
    proj_in = batch[0].to(device)
    proj_labels = batch[2].to(device).view(-1)
    
    model.eval()
    with torch.no_grad():
        enc, _, _ = model.encode(proj_in)
        valid = torch.any(proj_in.permute(0, 2, 3, 1).contiguous().reshape(-1, proj_in.shape[1]) != 0, dim=1)
        
        enc_norm = F.normalize(enc[valid], dim=1).to(model.classify.weight.dtype)
        preds = model.classify(enc_norm).argmax(dim=1)
        conf = model.get_confidence(enc_norm, preds)
        ratios = -conf
        
        true_labels = proj_labels.view(-1)[valid]
        correct = (preds == true_labels)
        
        thresh = model.knn_threshold
        admitted = conf > thresh
        
        print("\n" + "="*40)
        print("--- KNN Gate Diagnostics ---")
        print(f"Ratio (d_in/d_out) Range: min={ratios.min().item():.4f}, median={ratios.median().item():.4f}, max={ratios.max().item():.4f}")
        print(f"Threshold Set: {thresh:.4f} (Ratio threshold: {-thresh:.4f})")
        print(f"Firing Rate: {admitted.float().mean().item()*100:.1f}%")
        print(f"Frame Accuracy: {correct.float().mean().item()*100:.1f}%")
        if admitted.any():
            print(f"Admitted Accuracy: {correct[admitted].float().mean().item()*100:.1f}%")
        print("="*40 + "\n")
        
        # Also print prototype diagnostic
        sims = F.linear(enc_norm, F.normalize(model.classify.weight, dim=1))
        max_sims, _ = sims.max(dim=1)
        p_thresh = model.prototype_threshold
        p_admitted = max_sims > p_thresh
        
        print("--- Prototype Gate Diagnostics ---")
        print(f"Cosine Similarity Range: min={max_sims.min().item():.4f}, median={max_sims.median().item():.4f}, max={max_sims.max().item():.4f}")
        print(f"Threshold Set: {p_thresh:.4f}")
        print(f"Firing Rate: {p_admitted.float().mean().item()*100:.1f}%")
        if p_admitted.any():
            print(f"Admitted Accuracy: {correct[p_admitted].float().mean().item()*100:.1f}%")
        print("="*40 + "\n")

if __name__ == "__main__":
    main()
