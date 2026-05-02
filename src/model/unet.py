import torch
import torch.nn as nn
from .builder import MODELS
from src.utils import get_logger
import segmentation_models_pytorch as smp
import torch

@MODELS.register_module()
class UNet(nn.Module):
    def __init__(self, model_encoder='resnet34', num_classes=3):
        super(UNet, self).__init__()

        self.logger = get_logger()
        
        #Encode is pretrained on imagenet, decoder is randomly initialized
        self.model = smp.Unet(
            encoder_name=model_encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        ).to(memory_format=torch.channels_last)

    def forward(self, image):
        return self.model(image)