import os
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from .builder import MODELS
from src.utils import get_logger


@MODELS.register_module()
class Segformer(nn.Module):
    """
    SegFormer wrapper for semantic segmentation.

    Notes:
    - Expects preprocessed tensors as input (B, C, H, W)
    - Outputs logits resized to input resolution
    """

    def __init__(
        self,
        weights,
        revision='main',
        num_classes=3,
        id2label=None,
        label2id=None,
        pre_classifier_dropout=None
    ):
        super().__init__()

        self.logger = get_logger()
        self.use_local = os.getenv("HF_FORCE_LOCAL") is not None

        if id2label is not None:
            id2label = dict(id2label)
        if label2id is not None:
            label2id = dict(label2id)

        self.logger.info(f"Loading SegFormer: {weights}")

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path=weights,
            revision=revision,
            local_files_only=self.use_local,
            use_safetensors=True,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )

        #Add dropout before the classifier if specified
        if pre_classifier_dropout is not None:
            self.model.decode_head.dropout = nn.Dropout(p=pre_classifier_dropout)
            self.logger.info(f"Decode head dropout set to {pre_classifier_dropout}")

    def forward(self, image):
        """
        Args:
            image: Tensor of shape (B, C, H, W)

        Returns:
            logits: Tensor of shape (B, num_classes, H, W)
        """

        logits = self.model(pixel_values=image).logits

        # Resize logits to match input image resolution using bilinear interpolation
        logits = F.interpolate(
            logits,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return logits