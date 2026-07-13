import copy
import json
import os
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from dataset.kitti.parser import Parser
from modules.HDC_utils import Model, EllipsoidModel

from tqdm import tqdm

from unsup_main import train_extractor, train_hdc, init_sub, test_hdc_model

import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from dataset.waymo_data import WaymoDataset
import torch.utils.data as torchdata

MODEL_DIR = "logs"
DATA_DIR = "/mnt/bravo/jmfleming/waymo_skitti"
LOG_DIR = "logs"
NUM_CLASSES = 13

MAX_HDC_EPOCHS = 10
FEATURE_EXTRACTOR_EPOCHS = 80

HD_DIM = 10000

HDC_SAVE_PATH = "logs/hdc.pth"
HDC_SUB_PATH = "logs/hdc_sub.pth"

USE_ENTROPY_MINIMIZATION = False

ALL_CONDITIONS = ["sunny", "rain", "night"]
ADVERSE_CONDITIONS = [c for c in ALL_CONDITIONS if c != "sunny"]

CONDITION_COLORS = {
    "sunny": "#F5C518",
    "rain":  "#4C9BE8",
    "night": "#7B4EA0",
}
DEFAULT_COLOR = "#AAAAAA"

ABLATION_CONFIGS = [
    {
        "name": "Baseline",
        "online_subclusters": False,
        "thresholds": [0.45, 0.80],
    },
    {
        "name": "Online subclusters",
        "online_subclusters": True,
        "thresholds": [0.45, 0.80],
    },
    {
        "name": "High-conf (0.65)",
        "online_subclusters": False,
        "thresholds": [0.65, 0.90],
    },
    {
        "name": "High-conf (0.75)",
        "online_subclusters": False,
        "thresholds": [0.75, 0.95],
    },
]

def get_loader(ARCH, DATA, sequences, shuffle=True, weather_filter=None):
    """
    Return a Parser initialised in Waymo mode for the given sequences.
    weather_filter : list of condition strings to include, or None for all.
    """
    return Parser(
        mode="waymo",
        root=DATA_DIR,
        train_sequences=sequences,
        valid_sequences=DATA["split"]["valid"],
        test_sequences=None,
        labels=DATA["labels"],
        color_map=DATA["color_map"],
        learning_map=DATA["learning_map"],
        learning_map_inv=DATA["learning_map_inv"],
        sensor=ARCH["dataset"]["sensor"],
        max_points=ARCH["dataset"]["max_points"],
        batch_size=1,
        workers=ARCH["train"]["workers"],
        gt=True,
        shuffle_train=shuffle,
    )

def get_condition_loaders(ARCH, DATA, sequences, batch_size=1, shuffle=False, conditions=None):
    """
    Build one DataLoader per weather condition for the given sequences.

    Returns a dict:  { "sunny": DataLoader, "rain": DataLoader, ... }
    Only conditions that have at least one frame in `sequences` are included.
    """
    if conditions is None:
        conditions = ALL_CONDITIONS

    common_kwargs = dict(
        root=DATA_DIR,
        sequences=sequences,
        labels=DATA["labels"],
        color_map=DATA["color_map"],
        learning_map=DATA["learning_map"],
        learning_map_inv=DATA["learning_map_inv"],
        sensor=ARCH["dataset"]["sensor"],
        max_points=ARCH["dataset"]["max_points"],
        transform=False,
        gt=True,
    )

    loaders = {}
    for cond in conditions:
        ds = WaymoDataset(**common_kwargs, weather_filter=[cond])
        if len(ds) == 0:
            print(f"  [get_condition_loaders] '{cond}' - 0 frames in these sequences, skipping.")
            continue
        loaders[cond] = torchdata.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=ARCH["train"]["workers"],
            drop_last=False,
        )
        print(f"  [get_condition_loaders] '{cond}' - {len(ds)} frames")

    return loaders

def pretrain_pipeline(ARCH, DATA, return_trainer=False, skip_extractor=False):
    print(f"--- Starting Pretraining on ALL sunny scenarios ---")

    PRE_DATA = copy.deepcopy(DATA)
    PRE_DATA["weather_filter"] = ["sunny"]

    if not skip_extractor:
        ARCH["train"]["batch_size"] = 24
        print("Training Feature Extractor (sunny only)...")
        trainer = train_extractor(ARCH, PRE_DATA, data_dir=DATA_DIR, epochs=FEATURE_EXTRACTOR_EPOCHS, return_trainer=True)

        if USE_ENTROPY_MINIMIZATION:
            print("Running target entropy minimization on adverse conditions...")
            adverse_loaders = get_condition_loaders(ARCH, DATA, DATA["split"]["train"], batch_size=ARCH["train"]["batch_size"], shuffle=True, conditions=ADVERSE_CONDITIONS)

            if adverse_loaders:
                target_dataset = torch.utils.data.ConcatDataset([loader.dataset for loader in adverse_loaders.values()])
                target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=ARCH["train"]["batch_size"], shuffle=True, num_workers=ARCH["train"]["workers"], drop_last=True)

                trainer.run_target_entropy_minimization(target_loader, epochs=3, lr=1e-5)

                checkpoint_path = os.path.join(MODEL_DIR, "SENet_valid_best")
                state = {'state_dict': trainer.model.state_dict()}
                torch.save(state, checkpoint_path)
                print(f"Saved adapted feature extractor weights to {checkpoint_path}")
            else:
                print("  No adverse condition frames found, skipping entropy minimization.")
        else:
            print("Skipping entropy minimization (USE_ENTROPY_MINIMIZATION=False)")
    else:
        print("Skipping Feature Extractor Pretraining... using existing weights.")
        trainer = None

    ARCH["train"]["batch_size"] = 6

    print("Training HDC Density Model (sunny only)...")
    model, _ = train_hdc(ARCH, PRE_DATA, data_dir=DATA_DIR, epochs=MAX_HDC_EPOCHS, return_extractor=True)

    sunny_loaders = get_condition_loaders(ARCH, PRE_DATA, PRE_DATA["split"]["train"], batch_size=ARCH["train"]["batch_size"], shuffle=True, conditions=["sunny"])

    if "sunny" not in sunny_loaders:
        raise RuntimeError("No sunny frames found in pretraining sequences.")

    print("Initializing Subclusters...")
    model.init_subclusters(sunny_loaders["sunny"])

    torch.save(model.state_dict(), HDC_SUB_PATH)
    print(f"Pretraining complete. Model saved to {HDC_SUB_PATH}")

    if return_trainer:
        return model, trainer
    return model

def incremental_update_test(ARCH, DATA, pretrained_path="logs/hdc_sub.pth", compare=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_seqs = DATA["split"]["train"]
    valid_seqs = DATA["split"]["valid"]

    print("Building per-condition validation loaders...")
    val_loaders = get_condition_loaders(
        ARCH, DATA, valid_seqs,
        batch_size=1, shuffle=False,
        conditions=ALL_CONDITIONS)

    if not val_loaders:
        raise RuntimeError("No validation frames found for any condition.")

    train_loaders = get_condition_loaders(
        ARCH, DATA, train_seqs,
        batch_size=1, shuffle=True,
        conditions=ADVERSE_CONDITIONS)

    if not train_loaders:
        print("No adverse condition frames found - skipping.")
        return

    history = {
        "steps_labels": [],
        "conditions": [],
        "acc_pairs": [],
        "miou_pairs": [],
        "config_names": [],
    }

    if "sunny" in val_loaders:
        print(f"\n{'='*60}")
        print(f"Condition: [SUNNY BASELINE]")
        if compare:
            model_base = load_d3ctta_model(pretrained_path)
        else:
            model_base = EllipsoidModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device, subcluster_type='continuous')
            model_base.load_state_dict(torch.load(pretrained_path, map_location=device))
            model_base.to(device)
        acc_sunny, miou_sunny = test_hdc_model(model_base, val_loaders["sunny"])
        print(f"    Baseline - acc: {acc_sunny:.4f}  mIoU: {miou_sunny:.4f}")

    for cond in ADVERSE_CONDITIONS:
        if cond not in train_loaders:
            continue

        print(f"\n{'='*60}")
        print(f"Condition: [{cond.upper()}]")

        val_loader_for_cond = val_loaders.get(cond, next(iter(val_loaders.values())))

        configs_to_run = [ABLATION_CONFIGS[0]] if compare else ABLATION_CONFIGS
        for cfg in configs_to_run:
            if compare:
                model = load_d3ctta_model(pretrained_path)
                # D3CTTA does not support online subclusters logic easily, so we just use inference_update
                update_fn = model.inference_update
            else:
                model = EllipsoidModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device, subcluster_type='continuous')
                model.load_state_dict(torch.load(pretrained_path, map_location=device))
                model.to(device)
                update_fn = model.inference_update_with_subcluster_pull if cfg["online_subclusters"] else model.inference_update

            print(f"  Config: {cfg['name']}")

            acc_pre, miou_pre = test_hdc_model(model, val_loader_for_cond)
            print(f"    Pre  - acc: {acc_pre:.4f}  mIoU: {miou_pre:.4f}")

            model.train()
            
            for _, batch_data in enumerate(tqdm(train_loaders[cond], desc=f"    update [{cond}|{cfg['name']}]", leave=False)):
                proj_in = batch_data[0]
                if proj_in.shape[1] > 0:
                    proj_in = proj_in.to(device)
                    if compare:
                        with torch.no_grad():
                            logits, sims, indices, h = model(proj_in)
                            predictions = torch.argmax(logits, dim=1)
                        proj_xyz = batch_data[10].to(device) if len(batch_data) > 10 else None
                        update_fn(
                            h=h,
                            predictions=predictions,
                            xyz=proj_xyz
                        )
                    else:
                        update_fn(
                            proj_in,
                            learning_rate=0.001,
                            distance_sensitivity=3.0,
                            thresholds=cfg["thresholds"]
                        )

            acc_post, miou_post = test_hdc_model(model, val_loader_for_cond)
            print(f"    Post - acc: {acc_post:.4f}  mIoU: {miou_post:.4f}  Δ mIoU: {miou_post - miou_pre:+.4f}")

            history["steps_labels"].append(f"{cond.capitalize()}")
            history["conditions"].append(cond)
            history["acc_pairs"].append((acc_pre, acc_post))
            history["miou_pairs"].append((miou_pre, miou_post))
            history["config_names"].append(cfg["name"])
    suffix = "_condition_split_d3ctta" if compare else "_condition_split"
    
    json_path = f'ablation_dumbbell{suffix}.json'
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f'Ablation metrics saved to {json_path}')
    
    save_multi_step_dumbbell_ug(history, DATA, file_suffix=suffix)

def save_multi_step_dumbbell_ug(history, DATA=None, file_suffix="", sunny_baseline=None):
    labels = history["steps_labels"]
    conditions = history["conditions"]
    acc_pairs = np.array(history["acc_pairs"])
    miou_pairs = np.array(history["miou_pairs"])

    fig = plt.figure(figsize=(18, max(8, len(labels) * 0.85 + 3)))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)

    y_pos = np.arange(len(labels))

    COLOR_PRE = '#4C9BE8'
    COLOR_POST = '#E8574C'

    def row_bg(yi):
        cond = conditions[yi] if yi < len(conditions) else "sunny"
        base = CONDITION_COLORS.get(cond, DEFAULT_COLOR)
        return base + "33"

    def draw_ax(ax, pairs, title):
        for yi in range(len(pairs)):
            ax.axhspan(yi - 0.45, yi + 0.45, color=row_bg(yi), zorder=0, alpha=0.8)

        ax.hlines(y_pos, pairs[:, 0], pairs[:, 1], color='#AAAAAA', alpha=0.6, linewidth=2, zorder=1)
        ax.scatter(pairs[:, 0], y_pos, color=COLOR_PRE,  s=130, label='Pre-Update',  zorder=3, edgecolors='white', linewidths=0.8)
        ax.scatter(pairs[:, 1], y_pos, color=COLOR_POST, s=130, label='Post-Update', zorder=3, edgecolors='white', linewidths=0.8)

        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.grid(axis='x', linestyle='--', alpha=0.35)
        ax.set_xlabel("Metric Value", fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='lower right', fontsize=9)

    if len(acc_pairs) > 0:
        draw_ax(ax1, acc_pairs, "Accuracy Gain per Condition")
        draw_ax(ax2, miou_pairs, "mIoU Gain per Condition")

        ax1.set_yticks(y_pos)
        tick_labels = ax1.set_yticklabels(labels, fontsize=8)
        for tick, cond in zip(tick_labels, conditions):
            tick.set_color(CONDITION_COLORS.get(cond, "black"))

        ax2.tick_params(labelleft=False)

        cond_patches = [
            mpatches.Patch(color=CONDITION_COLORS[c], label=c.capitalize())
            for c in ALL_CONDITIONS
            if c in conditions
        ]
        ax1.legend(
            handles=cond_patches,
            title="Condition", loc='upper left',
            fontsize=8, title_fontsize=8,
            bbox_to_anchor=(0, -0.06), ncol=len(cond_patches),
            frameon=True, framealpha=0.9,
        )

    subtitle = ""
    if sunny_baseline is not None:
        subtitle = (f"Baseline sunny performance (no adaptation): acc {sunny_baseline['acc']:.4f}  |  mIoU {sunny_baseline['miou']:.4f}")
    
    plt.suptitle("Impact of Incremental Unsupervised Inference Updates", fontsize=16, fontweight='bold', y=1.03)
    if subtitle:
        fig.text(0.5, 1.005, subtitle, ha='center', fontsize=10, color='#666666')

    plt.tight_layout()

    out_path = f"incremental_dumbbell_results{file_suffix}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Dumbbell plot saved to {out_path}")

def save_ablation_dumbbell(ablation_histories, sunny_baseline=None, file_suffix=""):
    active_histories = [h for h in ablation_histories if len(h["conditions"]) > 0]
    if not active_histories:
        print("No active ablation histories to plot.")
        return
        
    conditions = active_histories[0]["conditions"]
    n_cond = len(conditions)
    n_abl = len(active_histories)

    band_height = 1.0
    y_spread = 0.15

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(6, n_cond * 1.8 + 3)), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    LINESTYLES = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--']
    ABLATION_COLORS = ['#555555', '#2266CC', '#CC6622', '#999999', '#9b59b6', '#1abc9c', '#e74c3c', '#34495e', '#f1c40f', '#8e44ad']
    MARKERS = ['o','^','s','D','v','p','*','h','<','>']
    COLOR_PRE = '#4C9BE8'
    COLOR_POST = '#E8574C'

    def draw_ax(ax, pairs_key, title):
        for ci, cond in enumerate(conditions):
            y_center = ci
            bg = CONDITION_COLORS.get(cond, DEFAULT_COLOR) + '33'
            ax.axhspan(y_center - 0.45, y_center + 0.45, color=bg, zorder=0, alpha=0.9)

            for ai, hist in enumerate(active_histories):
                if ci >= len(hist[pairs_key]):
                    continue
                pre, post = hist[pairs_key][ci]
                y = y_center + (ai - (n_abl - 1) / 2) * y_spread
                
                c_idx = ai % len(ABLATION_COLORS)
                m_idx = ai % len(MARKERS)

                ax.hlines(y, pre, post, color=ABLATION_COLORS[c_idx], linestyle=LINESTYLES[c_idx], linewidth=1.6, alpha=0.75, zorder=1)
                ax.scatter(pre,  y, color=COLOR_PRE,  s=70, zorder=3, edgecolors='white', linewidths=0.6, marker=MARKERS[m_idx])
                ax.scatter(post, y, color=COLOR_POST, s=70, zorder=3, edgecolors='white', linewidths=0.6, marker=MARKERS[m_idx])

                delta = post - pre
                x_mid = (pre + post) / 2

                delta_color = '#2a9d2a' if delta >= 0 else '#cc3333'
                ax.text(x_mid, y + 0.055, f'{delta:+.3f}', ha='center', va='bottom', fontsize=7.5, color=delta_color, fontweight='bold')

        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        ax.set_xlabel('Metric value', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)

    draw_ax(ax1, 'acc_pairs', 'Accuracy gain per condition')
    draw_ax(ax2, 'miou_pairs', 'mIoU gain per condition')

    y_pos = np.arange(n_cond)
    ax1.set_yticks(y_pos)
    ticks = ax1.set_yticklabels(conditions, fontsize=9)
    for tick, cond in zip(ticks, conditions):
        tick.set_color(CONDITION_COLORS.get(cond, 'black'))
    ax2.tick_params(labelleft=False)

    abl_handles = [plt.Line2D([0], [0], color=ABLATION_COLORS[i % len(ABLATION_COLORS)], linestyle=LINESTYLES[i % len(LINESTYLES)], linewidth=1.8, marker=MARKERS[i % len(MARKERS)], markersize=6, markerfacecolor='white', markeredgecolor=ABLATION_COLORS[i % len(ABLATION_COLORS)], label=hist['name']) for i, hist in enumerate(active_histories)]
    ax1.legend(handles=abl_handles, title='Ablation config', fontsize=8, title_fontsize=8, loc='lower left', bbox_to_anchor=(0, -0.18), ncol=2, frameon=True, framealpha=0.9)

    pre_post = [plt.scatter([], [], color=COLOR_PRE,  s=60, label='Pre-update'), plt.scatter([], [], color=COLOR_POST, s=60, label='Post-update'),]
    ax2.legend(handles=pre_post, fontsize=8, loc='lower right')

    title_str = 'Impact of incremental unsupervised inference updates | Ablation Study'
    if sunny_baseline:
        sub = (f"Baseline sunny (no adaptation): acc {sunny_baseline['acc']:.4f}  |  mIoU {sunny_baseline['miou']:.4f}")
        plt.suptitle(title_str, fontsize=14, fontweight='bold', y=0.98)
        fig.text(0.5, 0.91, sub, ha='center', fontsize=9.5, color='#666666')
    else:
        plt.suptitle(title_str, fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    out_path = f'ablation_dumbbell{file_suffix}.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Ablation dumbbell plot saved to {out_path}')

def save_final_plot(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history["steps"], history["miou"], 'r-s', label='mIoU')
    plt.plot(history["steps"], history["acc"],  'b-o', label='Accuracy')
    plt.xlabel('Condition Update Step')
    plt.ylabel('Performance Metrics')
    plt.title('HDC Model Improvement via Incremental Inference Updates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('incremental_update_test.png', dpi=300)
    plt.close()
    print("Plot saved to incremental_update_test.png")

import argparse

def load_d3ctta_model(path):
    print(f"Loading pretrained feature extractor for D3CTTA from {path}...")
    from modules.network.ResNet import ResNet_34
    from modules.D3CTTA import D3CTTA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES = 13
    feature_extractor = ResNet_34(NUM_CLASSES, aux=False, use_adaptor=True)
    
    se_path = os.path.join(os.path.dirname(path), "SENet_valid_best")
    if os.path.exists(se_path):
        w_dict = torch.load(se_path, map_location=device)
        feature_extractor.load_state_dict(w_dict['state_dict'], strict=False)
    feature_extractor.to(device)
    feature_extractor.eval()
    
    model = D3CTTA(feature_extractor, num_classes=NUM_CLASSES)
    model.to(device)
    return model

def main():
    parser = argparse.ArgumentParser(description="Test Unsupervised Updates on Waymo UGW")
    parser.add_argument('--pretrain', action='store_true', help='Pretrain the model on sunny conditions')
    parser.add_argument('--skip_extractor', action='store_true', help='Skip feature extractor pretraining and only retrain the HDC model')
    parser.add_argument('--pretrained_path', type=str, default='logs/hdc_sub.pth', help='Path to load pretrained model')
    parser.add_argument('--compare', action='store_true', help='Use D3CTTA with pretrained feature extractor instead of HDC')
    args = parser.parse_args()
    
    try:
        ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    except Exception as e:
        print(f"Error opening arch yaml file. {e}")
        quit()
    try:
        DATA = yaml.safe_load(open("config/labels/waymo.yaml", 'r'))
    except Exception as e:
        print(f"Error opening data yaml file. {e}")
        quit()

    if args.pretrain:
        model = pretrain_pipeline(ARCH, DATA, skip_extractor=args.skip_extractor)
    
    incremental_update_test(ARCH, DATA, args.pretrained_path, args.compare)

if __name__ == "__main__":
    main()