import torch
from utils import dice_batch, dice_coef
import numpy as np
from numpy import einsum
# To show the difference between 2D and 3D Dice coefficient calculations, let's generate
# an example of 3D volumetric segmentation and calculate both the 2D and 3D Dice coefficients.



# # Define a batch of 3 grayscale images (binary segmentation), each of size 2x2
# batch_size = 3
# image_size = (2, 2)

# # Example predictions (batch_size, height, width)
# predictions = torch.tensor([
#     [[1, 0], [0, 1]],
#     [[0, 1], [1, 0]],
#     [[1, 1], [0, 0]]
# ])

# # Example ground truth (batch_size, height, width)
# ground_truth = torch.tensor([
#     [[1, 0], [0, 1]],
#     [[1, 1], [0, 0]],
#     [[0, 1], [1, 0]]
# ])

# # Calculate Dice coefficient for positive class (per image and average over batch)
# dice_c = dice_coef(predictions, ground_truth)[:, 1].mean()

# # Calculate Dice coefficient for positive class (treating the entire batch as one)
# dice_b = dice_batch(predictions, ground_truth)

# # print("coef", dice_c)
# # print("batch", dice_b)






# Let's create two simple 3D volumes (ground truth and predicted) of size 4x4x3 (HxWxD)
# Ground truth (GT): a cube with ones in the middle
gt_volume =  torch.tensor([[[0, 0, 0, 0],
                       [0, 1, 1, 0],
                       [0, 1, 1, 0],
                       [0, 0, 0, 0]],

                      [[0, 0, 0, 0],
                       [0, 1, 1, 0],
                       [0, 1, 1, 0],
                       [0, 0, 0, 0]],

                      [[0, 0, 0, 0],
                       [0, 1, 1, 0],
                       [0, 1, 1, 0],
                       [0, 0, 0, 0]]])

# Predicted segmentation (Pred): almost the same but with a small error in the last slice
pred_volume = torch.tensor([[[0, 0, 0, 0],
                         [0, 1, 1, 0],
                         [0, 1, 1, 0],
                         [0, 0, 0, 0]],

                        [[0, 0, 0, 0],
                         [0, 1, 1, 0],
                         [0, 1, 1, 0],
                         [0, 0, 0, 0]],

                        [[0, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 1, 1, 0],
                         [0, 0, 0, 0]]])





# Function to calculate 2D Dice coefficient slice-by-slice and return the average
def dice_coefficient_2d_einsum(gt_slice, pred_slice, smooth=1e-8):
    intersection = einsum('ij,ij->', gt_slice, pred_slice)  # Sum of element-wise multiplication for intersection
    sum_gt_pred = einsum('ij->', gt_slice) + einsum('ij->', pred_slice)  # Sum of all elements
    dice = (2. * intersection + smooth) / (sum_gt_pred + smooth)  # Dice formula
    return dice

# Function to calculate Dice coefficient using einsum for 3D volume
def dice_coefficient_3d_einsum(gt_volume, pred_volume, smooth=1e-8):
    intersection = einsum('ijk,ijk->', gt_volume, pred_volume)  # Sum of element-wise multiplication for intersection
    sum_gt_pred = einsum('ijk->', gt_volume) + einsum('ijk->', pred_volume)  # Sum of all elements in 3D volume
    dice = (2. * intersection + smooth) / (sum_gt_pred + smooth)  # Dice formula
    return dice


# Calculate 2D Dice slice-by-slice using einsum
dice_2d_slices_einsum = [dice_coefficient_2d_einsum(gt_volume[i, :, :], pred_volume[i, :, :]) for i in range(gt_volume.shape[0])]
dice_2d_avg_einsum = np.mean(dice_2d_slices_einsum)

# Calculate 3D Dice using einsum
dice_3d_einsum = dice_coefficient_3d_einsum(gt_volume, pred_volume)

# Display the results
print("2d dices for each slice", dice_2d_slices_einsum)
print("2d dices av", dice_2d_avg_einsum) 
print("3d dice", dice_3d_einsum)


# Calculate Dice coefficient for positive class (per image and average over batch)
dice_c = dice_coef(pred_volume, gt_volume)[:, 1].mean()
# Calculate Dice coefficient for positive class (treating the entire batch as one)
dice_b = dice_batch(pred_volume, gt_volume)
print("coef", dice_c)
print("batch", dice_b)


