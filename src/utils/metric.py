#import pandas as pd
import torch

class GlobalSegmentationMetrics:
    def __init__(self, num_classes, ignore_index=255, device='cpu'):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device
        self.reset()

    def reset(self):
        self.tp = torch.zeros(self.num_classes, dtype=torch.float32, device=self.device)
        self.fp = torch.zeros_like(self.tp)
        self.fn = torch.zeros_like(self.tp)

    @torch.no_grad()
    def update(self, preds, targets):
        preds = preds.to(self.device).view(-1).long()
        targets = targets.to(self.device).view(-1).long()

        valid = targets != self.ignore_index
        preds = preds[valid]
        targets = targets[valid]

        if preds.numel() == 0:
            return

        K = self.num_classes
        indices = (targets * K + preds).to(torch.int64)
        cm = torch.bincount(indices, minlength=K*K).reshape(K, K).float()

        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp

        self.tp += tp
        self.fp += fp
        self.fn += fn

    def compute(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall    = self.tp / (self.tp + self.fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        iou       = self.tp / (self.tp + self.fp + self.fn + 1e-8)

        return {
            "precision": precision.cpu(),
            "recall": recall.cpu(),
            "f1": f1.cpu(),
            "iou": iou.cpu(),
            "mean_f1": f1.mean().item(),
            "mean_iou": iou.mean().item()
        }

    def reduce(self, accelerator):
        self.tp = accelerator.reduce(self.tp, reduction="sum")
        self.fp = accelerator.reduce(self.fp, reduction="sum")
        self.fn = accelerator.reduce(self.fn, reduction="sum")
        
        
class OrthoSegmentationMetrics:
    """
    Computes macro-averaged precision, recall, F1-score, and IoU for multi-class
    semantic segmentation, averaged across orthophotos.
    Returns:
        {
            "macro_precision_per_class": Tensor[C],
            "macro_recall_per_class": Tensor[C],
            "macro_f1_per_class": Tensor[C],
            "macro_iou_per_class": Tensor[C],
            "macro_precision": float,
            "macro_recall": float,
            "macro_f1": float,
            "macro_iou": float
        }
    
        
    """
    def __init__(self, num_classes, ignore_index=255, device="cpu"):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device
        self.reset()

    def reset(self):
        self.data = {}  # ortho_id -> counts

    @torch.no_grad()
    def update(self, preds, targets, ortho_id):
        preds = preds.to(self.device).view(-1).long()
        targets = targets.to(self.device).view(-1).long()

        # mask ignored pixels
        valid = targets != self.ignore_index
        preds = preds[valid]
        targets = targets[valid]

        # skip empty ortho
        if preds.numel() == 0:
            return

        if ortho_id not in self.data:
            self.data[ortho_id] = {
                "tp": torch.zeros(self.num_classes, device=self.device, dtype=torch.float32),
                "fp": torch.zeros(self.num_classes, device=self.device, dtype=torch.float32),
                "fn": torch.zeros(self.num_classes, device=self.device, dtype=torch.float32),
            }

        K = self.num_classes

        # -------- vectorized confusion matrix --------
        indices = (targets * K + preds).to(torch.int64)
        cm = torch.bincount(indices, minlength=K * K).reshape(K, K).float()

        # -------- derive tp, fp, fn --------
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp

        # -------- accumulate --------
        self.data[ortho_id]["tp"] += tp
        self.data[ortho_id]["fp"] += fp
        self.data[ortho_id]["fn"] += fn
        
    def compute(self):
        precisions = []
        recalls = []
        f1s = []
        ious = []

        for d in self.data.values():
            tp, fp, fn = d["tp"], d["fp"], d["fn"]

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            iou = tp / (tp + fp + fn + 1e-8)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            ious.append(iou)

        precisions = torch.stack(precisions)  # [N_orthos, C]
        recalls = torch.stack(recalls)
        f1s = torch.stack(f1s)
        ious = torch.stack(ious)

        return {
            "macro_precision_per_class": precisions.mean(dim=0),
            "macro_recall_per_class": recalls.mean(dim=0),
            "macro_f1_per_class": f1s.mean(dim=0),
            "macro_iou_per_class": ious.mean(dim=0),
            "macro_precision": precisions.mean().item(),
            "macro_recall": recalls.mean().item(),
            "macro_f1": f1s.mean().item(),
            "macro_iou": ious.mean().item(),
        }
