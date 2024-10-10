import numpy as np

import numpy as np

# Load and print train dice metrics
dice_train = np.load('./results/segthor/finalfinal1010/dice_tra.npy')
IOU_train = np.load('./results/segthor/finalfinal1010/iou_tra.npy')

dice_val = np.load('./results/segthor/finalfinal1010/dice_val.npy')
IOU_val = np.load('./results/segthor/finalfinal1010/iou_val.npy')

dice3d_val = np.load('./results/segthor/finalfinal1010/3ddice_val.npy')
iou3d_val = np.load('./results/segthor/finalfinal1010/3dIOU_val.npy')
hausdorff_val = np.load('./results/segthor/finalfinal1010/HD.npy')
haysdorff95_val = np.load('./results/segthor/finalfinal1010/95HD.npy')


# Print neatly formatted results with spacing
print(f"{'Metric':<30}{'Shape':<20}{'Mean Organs (except background)'}")
print("-" * 70)
print(f"{'Dice train':<30}{str(dice_train.shape):<20}{dice_train[:, :, 1:].mean():.4f}")
print(f"{'IOU train':<30}{str(IOU_train.shape):<20}{IOU_train[:, :, 1:].mean():.4f}")

print(f"{'Dice val':<30}{str(dice_val.shape):<20}{dice_val[:, :, 1:].mean():.4f}")
print(f"{'IOU val':<30}{str(IOU_val.shape):<20}{IOU_val[:, :, 1:].mean():.4f}")

print(f"{'Dice 3D val':<30}{str(dice3d_val.shape):<20}{dice3d_val[:, :, 1:].mean():.4f}")
print(f"{'IOU 3D val':<30}{str(iou3d_val.shape):<20}{iou3d_val[:, :, 1:].mean():.4f}")


print(f"{'HD val':<30}{str(hausdorff_val.shape):<20}{hausdorff_val[:, :, :].mean():.4f}")
print(f"{'HD 95':<30}{str(haysdorff95_val.shape):<20}{haysdorff95_val[:, :, :].mean():.4f}")

