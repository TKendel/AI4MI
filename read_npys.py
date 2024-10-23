import numpy as np

import numpy as np

# Load and print train dice metrics
dice_train = np.load('./results/segthor/floormetric/dice_tra.npy')
dice_val = np.load('./results/segthor/floormetric/dice_val.npy')
dice3d_train = np.load('./results/segthor/floormetric/dice3d_tra.npy')
dice3d_val = np.load('./results/segthor/floormetric/dice3d_val.npy')

# Print neatly formatted results with spacing
print(f"{'Metric':<30}{'Shape':<20}{'Mean (except background)'}")
print("-" * 70)
print(f"{'Dice train':<30}{str(dice_train.shape):<20}{dice_train[:, :, 1:].mean():.4f}")
print(f"{'Dice 3D train':<30}{str(dice3d_train.shape):<20}{dice3d_train[:, :, 1:].mean():.4f}")
print(f"{'Dice val':<30}{str(dice_val.shape):<20}{dice_val[:, :, 1:].mean():.4f}")
print(f"{'Dice 3D val':<30}{str(dice3d_val.shape):<20}{dice3d_val[:, :, 1:].mean():.4f}")
