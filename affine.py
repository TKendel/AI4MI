#!/usr/bin/env python3.10

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

import numpy as np
from numpy import pi as π
import nibabel as nib
from scipy.ndimage import affine_transform  # To apply affine matrix
import os
import shutil

original_dir = "./data/OLD DATA/segthor_train_og"
new_dir = "./data/segthor_train"  ### MAKE SURE YOU RENAME THE OLD (ORIGINAL DATA) BECAUSE IT WILL BE OVERWRITTEN

if not os.path.exists(new_dir):
    shutil.copytree(original_dir, new_dir)
    print(f"Copied {original_dir} to {new_dir}")



# given transformation
TR = np.asarray([[1, 0, 0, 50],
                 [0,  1, 0, 40],  # noqa: E241
                 [0,             0,      1, 15],  # noqa: E241
                 [0,             0,      0, 1]])  # noqa: E241

DEG: int = 27
ϕ: float = - DEG / 180 * π
RO = np.asarray([[np.cos(ϕ), -np.sin(ϕ), 0, 0],  # noqa: E241, E201
                 [np.sin(ϕ),  np.cos(ϕ), 0, 0],  # noqa: E241
                 [     0,         0,     1, 0],  # noqa: E241, E201
                 [     0,         0,     0, 1]])  # noqa: E241, E201

X_bar: float = 275
Y_bar: float = 200
Z_bar: float = 0
C1 = np.asarray([[1, 0, 0, X_bar],
                 [0, 1, 0, Y_bar],
                 [0, 0, 1, Z_bar],
                 [0, 0, 0,    1]])  # noqa: E241
C2 = np.linalg.inv(C1)

AFF = C1 @ RO @ C2 @ TR
INV = np.linalg.inv(AFF)
print(f"{AFF=}")
print(f"{RO=}")
print(f"{AFF=}")
print(f"{INV=}")

for root, dirs, files in os.walk(new_dir):
    for file in files:
        if file.endswith("GT.nii.gz"):  # Process only GT files
            gt_path = os.path.join(root, file)
            print(f"Processing {gt_path}...")

            img = nib.load(gt_path)
            gt = img.get_fdata()
            original_affine = img.affine

            heart_segmentation = (gt == 2).astype(np.uint8)  # Binary mask for heart

            shifted_heart = affine_transform(heart_segmentation, INV[:3, :3], offset=INV[:3, 3])
            shifted_heart = np.round(shifted_heart).astype(np.uint8)  # Ensure binary mask


            transformed_data = np.copy(gt)
            transformed_data[gt == 2] = 0  # Remove original heart
            transformed_data[shifted_heart == 1] = 2  # Replace with transformed heart

            transformed_data = transformed_data.astype(np.uint8)
            aligned_img = nib.Nifti1Image(transformed_data, affine=original_affine)
            nib.save(aligned_img, gt_path)
            print(f"Saved transformed GT for {file} at {gt_path}")




