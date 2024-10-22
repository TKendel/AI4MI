# File to plot and compare the different implementations

#This was a quick comparison between the two models, however, it
# was never finished as the export from nnunet to npy files 
# during training continuosly failed

import os
import numpy as np

# E-net
e_folder = 'AI4MI/results/segthor/ce'

val_e = np.load(os.path.join(e_folder, 'loss_val.npy'))
tra_e = np.load(os.path.join(e_folder, 'loss_tra.npy'))
dive_tra_e = np.load(os.path.join(e_folder, 'dice_tra.npy'))
dice_val_e = np.load(os.path.join(e_folder, 'dice_val.npy'))

# nnU-Net
n_folder = 'Dataset010_SEGT/nnUNet_trained_models/Dataset555_Segthor/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0'

val_n = np.load(os.path.join(n_folder, 'loss_val.npy'))
tra_n = np.load(os.path.join(n_folder, 'loss_tra.npy'))
dive_tra_n = np.load(os.path.join(n_folder, 'dice_tra.npy'))
dice_val_n = np.load(os.path.join(n_folder, 'dice_val.npy'))