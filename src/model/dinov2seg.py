import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .builder import MODELS
from src.utils import get_logger


@MODELS.register_module()
class DINOv2Seg(nn.Module):
    """
    DINOv2-based segmentation model.

    Modes:
    - last layer only: use features from the last layer for segmentation.
    - multi-layer fusion: fuse features from multiple layers for better segmentation performance.
    """

    def __init__(
        self,
        weights="facebook/dinov2-base",
        num_classes=3,
        use_multilayer=True,
        selected_layers=[3, 6, 9, 12],
        embed_dim=128,
        patch_size=14
    ):
        super().__init__()

        self.logger = get_logger()
        self.use_multilayer = use_multilayer
        self.selected_layers = selected_layers
        self.patch_size = patch_size

        self.logger.info(f"Loading DINOv2: {weights}")

        self.backbone = AutoModel.from_pretrained(weights)
        
        #freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_dim = self.backbone.config.hidden_size

        # projection layers (only if multilayer)
        if self.use_multilayer:
            self.proj = nn.ModuleList([
                nn.Conv2d(hidden_dim, embed_dim, kernel_size=1)
                for _ in selected_layers
            ])
            decoder_in_dim = embed_dim * len(selected_layers)
        else:
            decoder_in_dim = hidden_dim

        # simple decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_in_dim, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def tokens_to_map(self, x, H, W):
        x = x[:, 1:, :]  # remove CLS token
        B, N, C = x.shape

        h = H // self.patch_size
        w = W // self.patch_size

        return x.transpose(1, 2).reshape(B, C, h, w)

    def forward(self, image):
        B, C, H, W = image.shape

        outputs = self.backbone(
            pixel_values=image,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states

        # -------------------------
        # LAST LAYER ONLY
        # -------------------------
        if not self.use_multilayer:
            x = hidden_states[-1]
            x = self.tokens_to_map(x, H, W)

        # -------------------------
        # MULTI-LAYER
        # -------------------------
        else:
            feats = [hidden_states[i] for i in self.selected_layers]

            maps = [self.tokens_to_map(f, H, W) for f in feats]
            maps = [self.proj[i](maps[i]) for i in range(len(maps))]

            x = torch.cat(maps, dim=1)

        # decode at low resolution
        logits = self.decoder(x)

        # upsample once
        logits = F.interpolate(
            logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        return logits