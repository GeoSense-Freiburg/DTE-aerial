import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig


class TverskyFocalLoss(nn.Module):
    """
    Tversky Focal Loss for multi-class semantic segmentation.

    This implementation ignores pixels with `ignore_index` by masking them out
    during TP/FP/FN computation.

    Supports per-class weighting via alpha, beta, and gamma.
    

    Args:
        num_classes (int): Number of classes.
        alpha (float or list): Weight for false positives.
        beta (float or list): Weight for false negatives.
        gamma (float or list): Focal exponent.
        smooth (float): Numerical stability term.
        ignore_index (int): Label to ignore.
    """

    def __init__(
        self,
        num_classes,
        alpha=0.5,
        beta=0.5,
        gamma=1.0,
        smooth=1e-6,
        ignore_index=255,
        logits=True
    ):
        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

        def to_tensor(param):
            if isinstance(param, (list, tuple, ListConfig)):
                return torch.tensor(param).view(num_classes)
            else:
                return torch.full((num_classes,), float(param))

        self.alpha = to_tensor(alpha)
        self.beta = to_tensor(beta)
        self.gamma = to_tensor(gamma)
        self.logits = logits    

    def forward(self, pred, targets):
        
        """
        Args:
            pred: [B, C, H, W] (logits)
            targets: [B, H, W] (integer labels)
        """

        # -------------------------
        # 1. Create valid mask
        # -------------------------
        valid_mask = (targets != self.ignore_index)  # [B, H, W]

        # -------------------------
        # 2. Softmax probabilities
        # -------------------------
        if self.logits:
            probs = F.softmax(pred, dim=1)  # [B, C, H, W]
        else:
            probs = pred

        # -------------------------
        # 3. One-hot targets (safe)
        # -------------------------
        targets_clamped = targets.clone()
        targets_clamped[~valid_mask] = 0  # Clamp ignored labels to 0 a

        targets_one_hot = F.one_hot(targets_clamped, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # -------------------------
        # 4. Expand mask
        # -------------------------
        valid_mask = valid_mask.unsqueeze(1).float()  # [B, 1, H, W]

        # -------------------------
        # 5. Compute TP / FP / FN
        # -------------------------
        dims = (0, 2, 3)

        TP = torch.sum(probs * targets_one_hot * valid_mask, dim=dims)
        FP = torch.sum(probs * (1 - targets_one_hot) * valid_mask, dim=dims)
        FN = torch.sum((1 - probs) * targets_one_hot * valid_mask, dim=dims)

        # -------------------------
        # 6. Move params to device
        # -------------------------
        alpha = self.alpha.to(pred.device)
        beta = self.beta.to(pred.device)
        gamma = self.gamma.to(pred.device)

        # -------------------------
        # 7. Tversky index
        # -------------------------
        tversky = (TP + self.smooth) / (
            TP + alpha * FP + beta * FN + self.smooth
        )

        # -------------------------
        # 8. Focal modulation
        # -------------------------
        loss_per_class = torch.pow((1 - tversky), gamma)

        
        # -------------------------
        # 9. Mask absent classes
        # -------------------------
        class_present = (targets_one_hot * valid_mask).sum(dim=dims) > 0

        #-------------------------
        # 10. Average only over present classes to avoid bias from absent ones
        #-------------------------
        
        # avoid empty case (rare but safe)
        if class_present.any():
            loss = loss_per_class[class_present].mean()
        else:
            loss = loss_per_class.mean()  # fallback

        return loss