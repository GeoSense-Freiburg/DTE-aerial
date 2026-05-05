#!/usr/bin/env python3
import argparse
import json
import os
import warnings
import torch
import pandas as pd

from collections import OrderedDict
from safetensors.torch import load_file

from src.dataset import build_loader_inference
from src.model import build_model
from src.utils import (
    get_config,
    get_logger,
    OrthoSegmentationMetrics,
    set_random_seed,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ============================
# Args
# ============================
def parse_args():
    parser = argparse.ArgumentParser("Segmentation evaluation")

    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='results.json')

    return parser.parse_args()


# ============================
# Load model
# ============================
def load_model(checkpoint, device, config):
    model = build_model(config.model).to(device)

    ckpt = load_file(checkpoint, device="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())

    model.load_state_dict(state_dict)
    model = model.to(memory_format=torch.channels_last, device=device)
    model.eval()

    return model


# ============================
# Validation
# ============================
@torch.inference_mode()
def validate(config, data_loader, model, logger):

    device = next(model.parameters()).device
    num_classes = config.model.num_classes

    logger.info("Starting inference...")
    logger.info(f"Dataset size: {len(data_loader)}")

    # -------------------
    # Metrics
    # -------------------

    ortho_metrics = OrthoSegmentationMetrics(
        num_classes=num_classes,
        ignore_index=255,
        device=device
    )

    records = []

    for idx, samples in enumerate(data_loader):

        images = samples["image"].to(device=device, dtype=torch.float32)
        masks_true = samples["mask"].to(device=device, dtype=torch.long).squeeze(1)

        ortho_ids = samples["ortho_id"]
        image_paths = samples["path"]
        indices = samples["idx"]

        logits = model(images)
        preds = torch.softmax(logits, dim=1).argmax(dim=1)

        # -------- per-sample --------
        for b in range(preds.shape[0]):

            pred_b = preds[b].view(-1)
            gt_b = masks_true[b].view(-1)

            valid = gt_b != 255
            pred_b = pred_b[valid]
            gt_b = gt_b[valid]

            if pred_b.numel() == 0:
                continue

            K = num_classes

            indices_cm = (gt_b * K + pred_b).to(torch.int64)
            cm = torch.bincount(indices_cm, minlength=K * K).reshape(K, K).float()

            tp = cm.diag()
            fp = cm.sum(dim=0) - tp
            fn = cm.sum(dim=1) - tp

            # -------- store --------
            record = {
                "idx": int(indices[b]),
                "ortho_id": str(ortho_ids[b]),
                "image_path": image_paths[b],
                "num_pixels": int(gt_b.numel()),
            }

            for c in range(K):
                record[f"tp_{c}"] = float(tp[c])
                record[f"fp_{c}"] = float(fp[c])
                record[f"fn_{c}"] = float(fn[c])

            records.append(record)

            # -------- macro metric update --------
            ortho_metrics.update(
                preds[b],
                masks_true[b],
                ortho_id=ortho_ids[b]
            )

        if idx % config.print_freq == 0:
            mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            logger.info(f"Iter {idx} | GPU Mem: {mem:.0f} MB")

    # ============================
    # Save parquet (KEY OUTPUT)
    # ============================
    df_records = pd.DataFrame(records)

    parquet_path = os.path.join(
        os.path.dirname(config.output) if hasattr(config, "output") else ".",
        "per_sample_metrics.parquet"
    )

    df_records.to_parquet(parquet_path, index=False)

    logger.info(f"Saved per-sample metrics → {parquet_path}")

    # ============================
    # Compute metrics
    # ============================
    ortho_results = ortho_metrics.compute()

    # -------------------
    # Aggregate
    # -------------------


   
    
    log_dict = {
        "deadtree_macro_f1": float(ortho_results["macro_f1_per_class"][2]),
        "forest_cover_macro_f1": float(ortho_results["macro_f1_per_class"][1]),
    }
    
    logger.info(
        f"DEADWOOD MACRO → F1: {log_dict['deadtree_macro_f1']:.2f}, "
        f"FOREST COVER MACRO → F1: {log_dict['forest_cover_macro_f1']:.2f}"
    )


    return log_dict


# ============================
# Main
# ============================
def main():
    args = parse_args()

    config = get_config(args)
    logger = get_logger(config)

    set_random_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = build_loader_inference(config)

    model = load_model(args.checkpoint, device, config)

    stats = validate(config, data_loader, model, logger)

    # -------------------
    # Save JSON
    # -------------------
    output_path = args.output

    # if user passes a directory → append default filename
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "results.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)

    logger.info(f"Saved results → {output_path}")


if __name__ == "__main__":
    main()