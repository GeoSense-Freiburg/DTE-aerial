import math
import rasterio
from rasterio.windows import Window

import torch
from torch.utils.data import Dataset

from PIL import Image
from torchvision import transforms

class GeoTIFFInferenceDataset(Dataset):

    def __init__(
        self,
        image_path,
        tile_size=1024,
        padding=256,
    ):

        super().__init__()

        self.image_path = image_path

        self.src = rasterio.open(image_path)
        self.profile = self.src.profile.copy()
        self.transform = self.src.transform
        self.crs = self.src.crs

        self.height = self.src.height
        self.width = self.src.width

        self.tile_size = tile_size
        self.padding = padding

        self.crop_size = tile_size - 2 * padding
        
        self.image_to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.tiles = []

        for y in range(0, self.height, self.crop_size):
            for x in range(0, self.width, self.crop_size):

                self.tiles.append(
                    {
                        "x": x,
                        "y": y,
                    }
                )
                
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):

        tile = self.tiles[idx]

        x = tile["x"]
        y = tile["y"]

        # Read a larger tile so we only use the center during prediction
        read_x = x - self.padding
        read_y = y - self.padding

        window = Window(
            col_off=read_x,
            row_off=read_y,
            width=self.tile_size,
            height=self.tile_size,
        )

        image = self.src.read(
            window=window,
            boundless=True,
            fill_value=0,
        )

        # rasterio -> HWC
        image = image.transpose(1, 2, 0)

        # ToTensor + ImageNet normalization
        image = self.image_to_tensor(Image.fromarray(image))

        return {
            "image": image,
            "x": x,
            "y": y,
        }
        
    def close(self):
        self.src.close()