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
from src.utils.metric import GlobalSegmentationMetrics
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
    parser.add_argument('--output', type=str, default='results')

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
# Grouped F1 helper
# ============================
def compute_grouped_f1(df: pd.DataFrame, group_col: str, num_classes: int) -> dict:
    """
    For each group, sum TP/FP/FN across samples then apply the same formula
    as GlobalSegmentationMetrics.compute() (identical epsilon, identical averaging).
    """
    eps = 1e-8
    results = {}

    for group_val, grp in df.groupby(group_col):
        tp = torch.tensor([grp[f"tp_{c}"].sum() for c in range(num_classes)], dtype=torch.float32)
        fp = torch.tensor([grp[f"fp_{c}"].sum() for c in range(num_classes)], dtype=torch.float32)
        fn = torch.tensor([grp[f"fn_{c}"].sum() for c in range(num_classes)], dtype=torch.float32)

        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)

        f1_per_class = [round(float(v), 4) for v in f1]

        results[str(group_val)] = {
            "n_samples":   int(len(grp)),
            "f1_per_class": f1_per_class,
            "deadtree_f1": f1_per_class[2] if num_classes > 2 else None,
            "forest_f1":   f1_per_class[1] if num_classes > 1 else None,
        }

    return results


def log_grouped_f1(logger, grouped: dict, group_label: str):
    logger.info(f"\n{'='*60}")
    logger.info(f"  Per-{group_label} Macro F1")
    logger.info(f"{'='*60}")
    for name, m in sorted(grouped.items()):
        logger.info(
            f"  {group_label}={name:<30s} | "
            f"macro={m['macro_f1']:.4f} | "
            f"deadtree={m['deadtree_f1']:.4f} | "
            f"forest={m['forest_f1']:.4f}"
        )
    logger.info(f"{'='*60}\n")


# ============================
# Validation
# ============================
@torch.inference_mode()
def validate(config, data_loader, model, logger, meta_csv_path: str):

    device = next(model.parameters()).device
    num_classes = config.model.num_classes

    logger.info("Starting inference...")
    logger.info(f"Dataset size: {len(data_loader)}")

    # ============================
    # Load metadata for group lookup
    # ============================
    meta_df = pd.read_csv(meta_csv_path)
    meta_df["_tile_basename"] = meta_df["tile_path"].apply(os.path.basename)
    tile_to_biome      = dict(zip(meta_df["_tile_basename"], meta_df["biome"]))
    tile_to_resolution = dict(zip(meta_df["_tile_basename"], meta_df["resolution"]))

    # -------------------
    # Metrics
    # -------------------
    ortho_metrics = OrthoSegmentationMetrics(
        num_classes=num_classes,
        ignore_index=255,
        device=device
    )

    # group metric dicts — one OrthoSegmentationMetrics per group,
    # so F1 is averaged across orthos within each biome / resolution
    biome_metrics      = {}   # biome_str      -> OrthoSegmentationMetrics
    resolution_metrics = {}   # resolution_str -> OrthoSegmentationMetrics

    def get_or_create(d, key):
        if key not in d:
            d[key] = OrthoSegmentationMetrics(num_classes=num_classes, ignore_index=255, device=device)
        return d[key]

    records = []

    for idx, samples in enumerate(data_loader):

        images = samples["image"].to(device=device, dtype=torch.float32)
        masks_true = samples["mask"].to(device=device, dtype=torch.long).squeeze(1)

        ortho_ids = samples["ortho_id"]
        image_paths = samples["path"]
        sample_indices = samples["idx"]

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
                "idx": int(sample_indices[b]),
                "ortho_id": str(ortho_ids[b]),
                "image_path": image_paths[b],
                "num_pixels": int(gt_b.numel()),
            }

            for c in range(K):
                record[f"tp_{c}"] = float(tp[c])
                record[f"fp_{c}"] = float(fp[c])
                record[f"fn_{c}"] = float(fn[c])

            records.append(record)

            # -------- global metric update --------
            ortho_metrics.update(
                preds[b],
                masks_true[b],
                ortho_id=ortho_ids[b]
            )

            # -------- grouped metric update --------
            tile_basename = os.path.basename(image_paths[b])
            biome      = tile_to_biome.get(tile_basename)
            resolution = tile_to_resolution.get(tile_basename)

            if biome is not None:
                get_or_create(biome_metrics, str(biome)).update(preds[b], masks_true[b], ortho_id=ortho_ids[b])
            if resolution is not None:
                get_or_create(resolution_metrics, str(resolution)).update(preds[b], masks_true[b], ortho_id=ortho_ids[b])

        if idx % config.print_freq == 0:
            mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            logger.info(f"Iter {idx} | GPU Mem: {mem:.0f} MB")

    # ============================
    # Build per-sample DataFrame & save parquet
    # ============================
    df_records = pd.DataFrame(records)

    df_records["_tile_basename"] = df_records["image_path"].apply(os.path.basename)
    df_records = df_records.merge(
        meta_df[["_tile_basename", "biome", "resolution"]],
        on="_tile_basename",
        how="left",
    )

    missing = df_records["biome"].isna().sum()
    if missing:
        logger.warning(
            f"{missing} sample(s) could not be joined to metadata "
            f"and will be excluded from grouped metrics."
        )

    output_dir = os.path.dirname(config.output) if hasattr(config, "output") else "."
    parquet_path = os.path.join(output_dir, "per_sample_metrics.parquet")
    df_records.to_parquet(parquet_path, index=False)
    logger.info(f"Saved per-sample metrics → {parquet_path}")

    # ============================
    # Compute & log overall metrics
    # ============================
    ortho_results = ortho_metrics.compute()

    log_dict = {
        "deadtree_macro_f1":     float(ortho_results["macro_f1_per_class"][2]),
        "forest_cover_macro_f1": float(ortho_results["macro_f1_per_class"][1]),
    }

    logger.info(
        f"DEADWOOD MACRO → F1: {log_dict['deadtree_macro_f1']:.4f}, "
        f"FOREST COVER MACRO → F1: {log_dict['forest_cover_macro_f1']:.4f}"
    )

    # ============================
    # Compute & log per-biome metrics
    # ============================
    def compute_and_log_group(group_metrics: dict, group_label: str) -> dict:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Per-{group_label} F1 (macro over orthos within group)")
        logger.info(f"{'='*60}")
        out = {}
        for name, metric in sorted(group_metrics.items()):
            m = metric.compute()
            f1_per_class = [round(float(v), 4) for v in m["macro_f1_per_class"]]
            n_orthos = len(metric.data)
            out[name] = {
                "n_orthos":     n_orthos,
                "f1_per_class": f1_per_class,
                "deadtree_f1":  f1_per_class[2] if num_classes > 2 else None,
                "forest_f1":    f1_per_class[1] if num_classes > 1 else None,
            }
            logger.info(
                f"  {group_label}={name:<30s} | n_orthos={n_orthos:>4d} | "
                f"deadtree={out[name]['deadtree_f1']:.4f} | "
                f"forest={out[name]['forest_f1']:.4f}"
            )
        logger.info(f"{'='*60}\n")
        return out

    log_dict["per_biome"]      = compute_and_log_group(biome_metrics,      "biome")
    log_dict["per_resolution"] = compute_and_log_group(resolution_metrics, "resolution")

    return log_dict


# ============================
# Main
# ============================
def main():
    print("Starting evaluation...")
    args = parse_args()
    print(f"Using config: {args.cfg}")

    config = get_config(args)
    logger = get_logger(config)

    set_random_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = build_loader_inference(config)

    model = load_model(args.checkpoint, device, config)

    # Pass the test meta CSV so we can join biome / resolution
    meta_csv_path = config.data.meta.test
    stats = validate(config, data_loader, model, logger, meta_csv_path)

    # -------------------
    # Save JSON
    # -------------------
    output_path = args.output
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(stats, f, indent=4)

    logger.info(f"Saved results → {output_path}")


if __name__ == "__main__":
    main()
