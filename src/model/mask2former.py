import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor

from .builder import MODELS
from src.utils import get_logger


@MODELS.register_module()
class Mask2Former(nn.Module):
    """
    Mask2Former wrapper for semantic segmentation.

    Notes:
    - Expects input tensor: (B, C, H, W)
    - Outputs logits: (B, num_classes, H, W)
    """

    def __init__(
        self,
        weights,
        num_classes=3,
        num_channels=3
    ):
        super().__init__()

        self.logger = get_logger()
        self.use_local = os.getenv("HF_FORCE_LOCAL") is not None
        self.num_channels = num_channels

        self.logger.info(f"Loading Mask2Former: {weights}")

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained_model_name_or_path=weights,
            num_labels=num_classes,
            ignore_mismatched_sizes=True, 
        )


    def forward(self, image):

        if image.shape[1] > 3:
            pixel_values = image[:, :3]
        else:
            pixel_values = image

        outputs = self.model(pixel_values=pixel_values)

        class_logits = outputs.class_queries_logits[..., :-1]   # [B, Q, C]
        mask_logits = outputs.masks_queries_logits              # [B, Q, H, W]

        # log-space fusion
        log_mask_prob = F.logsigmoid(mask_logits)               # [B, Q, H, W]

        class_logits = class_logits.permute(0, 2, 1)            # [B, C, Q]
        class_logits = class_logits.unsqueeze(-1).unsqueeze(-1) # [B, C, Q, 1, 1]

        fused = class_logits + log_mask_prob.unsqueeze(1)       # [B, C, Q, H, W]

        pixel_logits = torch.logsumexp(fused, dim=2)            # [B, C, H, W]

        # stability
        pixel_logits = torch.clamp(pixel_logits, -20, 20)

        return F.interpolate(
            pixel_logits,
            size=image.shape[-2:],
            mode='bilinear',
            align_corners=False
        )