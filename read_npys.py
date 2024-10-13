import numpy as np
"""
This script performs the following tasks:
1. Reads in numpy arrays containing metrics from saved .npy files for segmentation performance evaluation.
2. Prints the shape of the arrays to verify the loaded data.
3. Computes whether the mathematical relationship between Dice and IoU metrics holds true, using the formula:
   IoU = Dice / (2 - Dice)
--> cna be used for other checks
"""

# Load and print train dice metrics
# dice_train = np.load('./results/segthor/finalfinal1010/dice_tra.npy')
# IOU_train = np.load('./results/segthor/finalfinal1010/iou_tra.npy')

# dice_val = np.load('./results/segthor/finalfinal1010/dice_val.npy')
# IOU_val = np.load('./results/segthor/finalfinal1010/iou_val.npy')

# dice3d_val = np.load('./results/segthor/finalfinal1010/3ddice_val.npy')
# iou3d_val = np.load('./results/segthor/finalfinal1010/3dIOU_val.npy')
# hausdorff_val = np.load('./results/segthor/finalfinal1010/HD.npy')
# haysdorff95_val = np.load('./results/segthor/finalfinal1010/95HD.npy')


# # Print neatly formatted results with spacing
# print(f"{'Metric':<30}{'Shape':<20}{'Mean Organs (except background)'}")
# print("-" * 70)
# print(f"{'Dice train':<30}{str(dice_train.shape):<20}{dice_train[:, :, 1:].mean():.4f}")
# print(f"{'IOU train':<30}{str(IOU_train.shape):<20}{IOU_train[:, :, 1:].mean():.4f}")

# print(f"{'Dice val':<30}{str(dice_val.shape):<20}{dice_val[:, :, 1:].mean():.4f}")
# print(f"{'IOU val':<30}{str(IOU_val.shape):<20}{IOU_val[:, :, 1:].mean():.4f}")

# print(f"{'Dice 3D val':<30}{str(dice3d_val.shape):<20}{dice3d_val[:, :, 1:].mean():.4f}")
# print(f"{'IOU 3D val':<30}{str(iou3d_val.shape):<20}{iou3d_val[:, :, 1:].mean():.4f}")


# print(f"{'HD val':<30}{str(hausdorff_val.shape):<20}{hausdorff_val[:, :, :].mean():.4f}")
# print(f"{'HD 95':<30}{str(haysdorff95_val.shape):<20}{haysdorff95_val[:, :, :].mean():.4f}")



# check fi the relation between IOU and Dice holds true
# Load the uploaded numpy arrays
dice_val_path = './results/segthor/BASELINE/3ddice_val.npy'
iou_val_path = './results/segthor/BASELINE/3diou_val.npy'
dice_val = np.load(dice_val_path)
iou_val = np.load(iou_val_path)
dice_shape = dice_val.shape
iou_shape = iou_val.shape
print(dice_shape, iou_shape)

computed_iou = dice_val / (2 - dice_val)
iou_check = np.isclose(iou_val, computed_iou)
percentage_true = np.mean(iou_check) * 100
print("Does the relation between dice and iou hold?", percentage_true)

# Check for epoch 19 specifically (index 18 in zero-indexed array)
epoch_19_dice = dice_val[19]
epoch_19_iou = iou_val[19]
computed_iou_epoch_19 = epoch_19_dice / (2 - epoch_19_dice)
iou_check_epoch_19 = np.isclose(epoch_19_iou, computed_iou_epoch_19)
percentage_true_epoch_19 = np.mean(iou_check_epoch_19) * 100
print("Does the relation between dice and iou hold in epoch 19?",percentage_true_epoch_19)

average_iou_per_class_epoch_19 = np.mean(epoch_19_iou, axis=0)
average_dice_per_class_epoch_19 = np.mean(epoch_19_dice, axis=0)
print("avg 3dice epoch 19", average_dice_per_class_epoch_19)
print("avg 3diou epoch 19", average_iou_per_class_epoch_19)


# If you average Dice and IoU across multiple volumes independently, you break the inherent connection because averaging non-linear functions (like 
# (Dice/2-Dice) does not preserve the original relationship.
computed_iou_per_class_3d = average_dice_per_class_epoch_19 / (2 - average_dice_per_class_epoch_19)
iou_check_per_class_3d = np.isclose(average_iou_per_class_epoch_19, computed_iou_per_class_3d)
print("Does the relation betwee dice and iou hold in ep0ch 19 after averaging over the 10 patients", iou_check_per_class_3d)
