import torch
import numpy as np
import warnings
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import (
                   class2one_hot,
                   probs2class,
                   tqdm_,
                   save_images)
from operator import itemgetter
from typing import Callable, Union
from torch import Tensor
from torch.utils.data import Dataset


'''
Test performance of model
'''

def make_dataset(root, subset) -> list[tuple[Path, Path]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    images = sorted(img_path.glob("*.png"))
    full_labels = sorted(full_path.glob("*.png"))

    return list(zip(images, full_labels))


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        img: Tensor = self.img_transform(Image.open(img_path))

        _, W, H = img.shape

        return {"images": img,
                "stems": img_path.stem}

img_transform = transforms.Compose([
    lambda img: img.convert('L'),  #converts to grayscale 
    lambda img: np.array(img)[np.newaxis, ...],
    lambda nd: nd / 255,  # max <= 1   # normalises
    lambda nd: torch.tensor(nd, dtype=torch.float32)
])

gt_transform = transforms.Compose([
    lambda img: np.array(img)[...],
    # The idea is that the classes are mapped to {0, 255} for binary cases
    # {0, 85, 170, 255} for 4 classes
    # {0, 51, 102, 153, 204, 255} for 6 classes
    # Very sketchy but that works here and that simplifies visualization
    lambda nd: nd / (255 / (K - 1)) if K != 5 else nd / 63,  # max <= 1
    lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
    lambda t: class2one_hot(t, K=K),
    itemgetter(0)
    ])

model = torch.load('BER\\bestmodel.pkl', weights_only=False, map_location=torch.device('cpu'))
model.eval()

root_dir = 'data\SEGTHOR_TEST\\'
K = 5

test_set = SliceDataset('test',
                            root_dir,
                            img_transform=img_transform,
                            debug=False)

test_loader = DataLoader(test_set,
                            num_workers=0,
                            shuffle=True) 

tq_iter = tqdm_(enumerate(test_loader), total=len(test_loader))
for i, data in tq_iter:  # i = batch
    img = data['images']
    pred = model(img)
    pred_probs = F.softmax(1 * pred, dim=1)  # 1 is the temperature parameter

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        predicted_class: torch.Tensor = probs2class(pred_probs)

        mult: int = 63 if K == 5 else (255 / (K - 1))
        save_images(predicted_class * mult,
                    data['stems'],
                    Path("data\SEGTHOR_TEST\predictions"))
