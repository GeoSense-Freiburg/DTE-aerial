import os
import numpy as np
import torch
from PIL import Image
import albumentations as A
from torchvision import transforms
from torchvision.transforms import functional as TF

from src.utils import get_logger



# -----------------------------
# DATASET
# -----------------------------
class DeadwoodDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataframe,
        input_dir,
        crop_size=(512, 512),
        is_train = False
    ):
        super().__init__()
        self.df = dataframe
        self.image_dir = input_dir
        self.logger = get_logger()

        self.is_train = is_train
        self.transform = None  # Albumentations transform, defined only if is_train=True

        if self.is_train:
            self.transform = A.Compose([
                A.RandomCrop(*crop_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ToGray(p=0.1),
                A.RandomGamma(p=0.3),
                A.ColorJitter(p=0.7),
            ])
        # Torch transforms (FINAL STEP)
        self.image_to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),  
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.mask_to_tensor = transforms.PILToTensor()  # keeps labels intact



    def __len__(self):
        return len(self.df)
    

    # -----------------------------
    # PIL LOADING
    # -----------------------------
    def _load_image_pil(self, path):
        return Image.open(path).convert("RGB")
    
    def _load_mask(self, mask_path):
        return np.array(
                Image.open(mask_path).convert("L"),
                dtype=np.int32,
        )
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = os.path.join(self.image_dir, str(row["tile_path"]))
        mask_path = os.path.join(self.image_dir, str(row["mask_path"]))
        ortho_id = row["id"]

        try:
            # 🔹 Load ONCE
            image_pil = self._load_image_pil(image_path)
            image_np = np.array(image_pil)

            # 🔹 Pass image_np to mask loader
            mask_np = self._load_mask(mask_path)

            # 🔹 Albumentations
            if self.is_train and self.transform is not None:
                augmented = self.transform(image=image_np, mask=mask_np)
                image_np = augmented["image"]
                mask_np = augmented["mask"]
                
            # 🔹 To torch tensors
            image = self.image_to_tensor(Image.fromarray(image_np))
            mask = (
                self.mask_to_tensor(Image.fromarray(mask_np))
                .squeeze(0)
                .long()
            )

            return {
                "image": image,
                "mask": mask,
                "path": image_path,
                "ortho_id": ortho_id,
                "idx": index,
            }

        except Exception as e:
            self.logger.exception(f"Failed at index {index}: {e}")
            raise
