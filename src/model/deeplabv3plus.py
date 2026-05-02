import os
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from .builder import MODELS
from src.utils import get_logger


@MODELS.register_module()
class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ wrapper using segmentation_models_pytorch.

    Notes:
    - Input: (B, C, H, W)
    - Output: (B, num_classes, H, W)
    - Uses ImageNet pretrained encoder (standard)
    """

    def __init__(
        self,
        encoder_name="resnet50",
        encoder_weights="imagenet",
        num_classes=3,
        in_channels=3,
        decoder_channels=256,
        encoder_output_stride=16,
        upsampling=4
    ):
        super().__init__()

        self.logger = get_logger()
        self.logger.info(f"Loading DeepLabV3+ ({encoder_name})")

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,  # ImageNet
            in_channels=in_channels,
            classes=num_classes,
            encoder_output_stride=encoder_output_stride,
            decoder_channels=decoder_channels,
            upsampling=upsampling,
        )

    def forward(self, image):
        """
        Args:
            image: (B, C, H, W)

        Returns:
            logits: (B, num_classes, H, W)
        """

        logits = self.model(image)

        # Safety resize (SMP usually already matches size, but keep consistent)
        logits = F.interpolate(
            logits,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return logits