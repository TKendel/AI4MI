from pathlib import Path
from typing import Callable, Union

import torch
import numpy as np
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# def make_dataset(root, subset) -> list[tuple[Path, Path]]:
#     assert subset in ['train', 'val', 'test']

#     root = Path(root)

#     img_path = root / subset / 'img'
#     full_path = root / subset / 'gt'

#     images = sorted(img_path.glob("*.png"))
#     full_labels = sorted(full_path.glob("*.png"))

#     return list(zip(images, full_labels))


# class SliceDataset(Dataset):
#     def __init__(self, subset, root_dir, img_transform=None,
#                  gt_transform=None, augment=False, equalize=False, debug=False):
#         self.root_dir: str = root_dir
#         self.img_transform: Callable = img_transform
#         self.gt_transform: Callable = gt_transform
#         self.augmentation: bool = augment
#         self.equalize: bool = equalize

#         self.files = make_dataset(root_dir, subset)
#         if debug:
#             self.files = self.files[:10]

#         print(f">> Created {subset} dataset with {len(self)} images...")

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
#         img_path, gt_path = self.files[index]

#         img: Tensor = self.img_transform(Image.open(img_path))
#         gt: Tensor = self.gt_transform(Image.open(gt_path))

#         _, W, H = img.shape
#         K, _, _ = gt.shape
#         assert gt.shape == (K, W, H)

#         return {"images": img,
#                 "gts": gt,
#                 "stems": img_path.stem}
    
# img_transform = transforms.Compose([
#     lambda img: img.convert('L'),  #converts to grayscale 
#     lambda img: np.array(img)[np.newaxis, ...],
#     lambda nd: nd / 255,  # max <= 1   # normalises
#     lambda nd: torch.tensor(nd, dtype=torch.float32)
# ])

img = Image.open("data\SEGTHOR\\train\img\Patient_03_0000.png").convert('L')
img = np.array(img)[np.newaxis, ...]
print(img)
print(img.dtype)
print(img.shape)

img.show()


