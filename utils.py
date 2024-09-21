#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
from functools import partial
from multiprocessing import Pool
from contextlib import AbstractContextManager
from typing import Callable, Iterable, List, Set, Tuple, TypeVar, cast
import os
from collections import defaultdict


import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import Tensor, einsum

tqdm_ = partial(tqdm, dynamic_ncols=True,
                leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]')


class Dcm(AbstractContextManager):
    # Dummy Context manager
    def __exit__(self, *args, **kwargs):
        pass


# Functools
A = TypeVar("A")
B = TypeVar("B")


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return Pool().map(fn, iter)


def starmmap_(fn: Callable[[Tuple[A]], B], iter: Iterable[Tuple[A]]) -> List[B]:
    return Pool().starmap(fn, iter)


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def class2one_hot(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape

    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(1, seg[:, None, ...], 1)

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)

    return res


def probs2class(probs: Tensor) -> Tensor:
    b, _, *img_shape = probs.shape
    assert simplex(probs)

    res = probs.argmax(dim=1)  # selects the class with the highest probability for each pixel.
    assert res.shape == (b, *img_shape) #res will be a tensor of class labels for each pixel (B, H, W)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, K, *_ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), K)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


# Save the raw predictions
def save_images(segs: Tensor, names: Iterable[str], root: Path) -> None:
        for seg, name in zip(segs, names):
                save_path = (root / name).with_suffix(".png")
                save_path.parent.mkdir(parents=True, exist_ok=True)

                if len(seg.shape) == 2:
                        Image.fromarray(seg.detach().cpu().numpy().astype(np.uint8)).save(save_path)
                elif len(seg.shape) == 3:
                        np.save(str(save_path), seg.detach().cpu().numpy())
                else:
                        raise ValueError(seg.shape)


# Metrics
def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices


dice_coef = partial(meta_dice, "bk...->bk") #computes Dice coefficients per image (b dimension) and per class (k dimension).
dice_batch = partial(meta_dice, "bk...->k")  # used for 3d dice  ## dice_scores will have shape (k,), giving us the Dice coefficient for each class across the entire 3D volume



def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a & b
    assert sset(res, [0, 1])

    return res


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a | b
    assert sset(res, [0, 1])

    return res



def count_slices_per_patient(image_dir):
    """
    Count the number of slices for each patient based on filenames in a directory.
    Args:
    - image_dir (str): Path to the directory containing patient slice images.
    Returns:
    - slices_per_patient (dict): Dictionary where keys are patient IDs and values are the number of slices.
    """
    slices_per_patient = defaultdict(int)

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(".png"):
            # Assuming the format is like "Patient_03_000.png"
            patient_id = '_'.join(filename.split('_')[:2])
            slices_per_patient[patient_id] += 1

    return slices_per_patient


def volume_dice(predictions, gts, path_to_slices):
    """
    Computes Dice similarity coefficient for predicted and ground truth segmentations per patient.

    Parameters:
    - predictions (Tensor): Predicted segmentation volumes, shape (num_slices, num_classes, height, width).
    - gts (Tensor): Ground truth segmentation volumes, shape same as `predictions`.
    - path_to_slices (str): Path to directory containing patient slices, used to count slices per patient.

    Returns:
    - dice_scores_per_patient (dict): A dictionary where keys are patient IDs and values are tensors of Dice scores 
      for each class, shape (num_classes,).
    
    Workflow:
    1. Organizes slices into patient volumes.
    2. For each patient, computes the Dice score across all classes using `dice_batch`.
    3. Returns a dictionary mapping patient IDs to their corresponding Dice scores.
    """
    prediction_patient_volumes = {}
    gt_patient_volumes = {}
    dice_scores_per_patient = {}

    start = 0
    count_slices = count_slices_per_patient(path_to_slices)
    
    for patient, num_slices in count_slices.items():
        predictions_patient_slices = predictions[start:start + num_slices]
        gt_patient_slices = gts[start:start + num_slices]

        prediction_patient_volumes[patient] = predictions_patient_slices
        gt_patient_volumes[patient] = gt_patient_slices

        # {
        # "Patient_01": tensor of shape (nr slices, nr classes, 256, 256),
        # "Patient_02": tensor of shape (nr slices, nr classes, 256, 256),
        # "Patient_03": tensor of shape (nr slices, nr classes, 256, 256),
        # }
        
        # Update the start index for the next patient
        start += num_slices

    
    for (patient_pred, volumepred), (patient_gt, volumegt) in zip(prediction_patient_volumes.items(), gt_patient_volumes.items()):
        assert patient_pred == patient_gt  # Ensure that we are processing the same patient
        dicescores = dice_batch(volumegt, volumepred)
    
        # Store the Dice scores per patient
        dice_scores_per_patient[patient_pred] = dicescores
        print(dice_scores_per_patient)

    # Return the dictionary containing Dice scores for each patient
    return dice_scores_per_patient

  

